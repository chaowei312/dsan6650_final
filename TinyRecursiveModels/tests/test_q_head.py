"""
Unit tests for Q head (ACT halting head) functionality.

Tests:
1. Q head outputs are computed correctly
2. Gradients flow through Q head during training
3. Validity-based Q head loss works correctly
4. Q head weights update during training
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from tests.evaluation import count_prediction_validity


def test_q_head_output_shape():
    """Test that Q head outputs have correct shape."""
    print("=" * 60)
    print("TEST 1: Q Head Output Shape")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = {
        'batch_size': 1, 'seq_len': 16, 'vocab_size': 6,
        'hidden_size': 128, 'num_heads': 4, 'expansion': 4,
        'H_cycles': 2, 'L_cycles': 4, 'H_layers': 0, 'L_layers': 3,
        'pos_encodings': 'rope', 'rms_norm_eps': 1e-5, 'rope_theta': 10000.0,
        'puzzle_emb_ndim': 0, 'puzzle_emb_len': 0, 'num_puzzle_identifiers': 1,
        'halt_max_steps': 16, 'halt_exploration_prob': 0.0, 'forward_dtype': 'float32',
        'detach_early_cycles': False,
    }
    
    model = TinyRecursiveReasoningModel_ACTV1(config).to(device)
    
    # Create dummy input
    batch_size = 4
    puzzles = torch.randint(1, 6, (batch_size, 16)).to(device)
    batch = {
        'inputs': puzzles,
        'puzzle_identifiers': torch.zeros(batch_size, dtype=torch.long, device=device)
    }
    
    # Forward pass
    carry = model.initial_carry(batch)
    carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
    carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
    carry.halted = carry.halted.to(device)
    carry.steps = carry.steps.to(device)
    
    carry, outputs = model(carry, batch)
    
    # Check outputs exist
    assert 'q_halt_logits' in outputs, "q_halt_logits not in outputs!"
    assert 'q_continue_logits' in outputs, "q_continue_logits not in outputs!"
    
    # Check shapes
    q_halt = outputs['q_halt_logits']
    q_continue = outputs['q_continue_logits']
    
    print(f"  q_halt shape: {q_halt.shape}")
    print(f"  q_continue shape: {q_continue.shape}")
    
    assert q_halt.shape[0] == batch_size, f"Expected batch size {batch_size}, got {q_halt.shape[0]}"
    
    print("  ✓ Q head outputs have correct shape!")
    return True


def test_q_head_gradient_flow():
    """Test that gradients flow through Q head."""
    print("\n" + "=" * 60)
    print("TEST 2: Q Head Gradient Flow")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = {
        'batch_size': 1, 'seq_len': 16, 'vocab_size': 6,
        'hidden_size': 128, 'num_heads': 4, 'expansion': 4,
        'H_cycles': 2, 'L_cycles': 4, 'H_layers': 0, 'L_layers': 3,
        'pos_encodings': 'rope', 'rms_norm_eps': 1e-5, 'rope_theta': 10000.0,
        'puzzle_emb_ndim': 0, 'puzzle_emb_len': 0, 'num_puzzle_identifiers': 1,
        'halt_max_steps': 16, 'halt_exploration_prob': 0.0, 'forward_dtype': 'float32',
        'detach_early_cycles': False,
    }
    
    model = TinyRecursiveReasoningModel_ACTV1(config).to(device)
    
    # Get Q head parameters
    q_head_params = list(model.inner.q_head.parameters())
    print(f"  Q head has {len(q_head_params)} parameter tensors")
    for i, p in enumerate(q_head_params):
        print(f"    param {i}: shape {p.shape}, requires_grad={p.requires_grad}")
    
    # Store initial weights
    initial_weights = [p.clone().detach() for p in q_head_params]
    
    # Forward pass
    batch_size = 4
    puzzles = torch.randint(1, 6, (batch_size, 16)).to(device)
    batch = {
        'inputs': puzzles,
        'puzzle_identifiers': torch.zeros(batch_size, dtype=torch.long, device=device)
    }
    
    carry = model.initial_carry(batch)
    carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
    carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
    carry.halted = carry.halted.to(device)
    carry.steps = carry.steps.to(device)
    
    carry, outputs = model(carry, batch)
    
    # Compute Q head loss - ORIGINAL TRM DESIGN
    # Train q_halt to predict binary "sequence is exactly correct"
    q_halt = outputs['q_halt_logits']
    target = (torch.rand(batch_size, device=device) > 0.5).float()  # Binary targets
    q_loss = F.binary_cross_entropy_with_logits(q_halt.squeeze(-1), target)
    
    print(f"\n  Q head loss: {q_loss.item():.4f}")
    
    # Backward pass
    q_loss.backward()
    
    # Check gradients exist
    has_grad = all(p.grad is not None for p in q_head_params)
    print(f"  All Q head params have gradients: {has_grad}")
    
    if has_grad:
        for i, p in enumerate(q_head_params):
            grad_norm = p.grad.norm().item()
            print(f"    param {i} grad norm: {grad_norm:.6f}")
    
    assert has_grad, "Q head parameters have no gradients!"
    
    # Do optimizer step
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.step()
    
    # Check weights changed
    weights_changed = any(
        not torch.allclose(p, initial_weights[i])
        for i, p in enumerate(q_head_params)
    )
    print(f"\n  Q head weights changed after optimization: {weights_changed}")
    
    assert weights_changed, "Q head weights did not change!"
    
    print("  ✓ Gradients flow through Q head correctly!")
    return True


def test_validity_based_q_loss():
    """Test validity-based Q head loss computation."""
    print("\n" + "=" * 60)
    print("TEST 3: Validity-Based Q Loss")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = {
        'batch_size': 1, 'seq_len': 16, 'vocab_size': 6,
        'hidden_size': 128, 'num_heads': 4, 'expansion': 4,
        'H_cycles': 2, 'L_cycles': 4, 'H_layers': 0, 'L_layers': 3,
        'pos_encodings': 'rope', 'rms_norm_eps': 1e-5, 'rope_theta': 10000.0,
        'puzzle_emb_ndim': 0, 'puzzle_emb_len': 0, 'num_puzzle_identifiers': 1,
        'halt_max_steps': 16, 'halt_exploration_prob': 0.0, 'forward_dtype': 'float32',
        'detach_early_cycles': False,
    }
    
    model = TinyRecursiveReasoningModel_ACTV1(config).to(device)
    
    # Create a simple puzzle (mostly filled, easy to predict)
    # Token space: 1=empty, 2-5=digits 1-4
    batch_size = 2
    
    # Puzzle 1: Valid Sudoku with some empty cells
    puzzle1 = torch.tensor([2, 3, 4, 5,  # Row 1: 1,2,3,4
                            4, 5, 2, 3,  # Row 2: 3,4,1,2
                            3, 4, 5, 2,  # Row 3: 2,3,4,1
                            5, 2, 3, 4]) # Row 4: 4,1,2,3
    solution1 = puzzle1.clone()
    # Make some cells empty
    puzzle1[0] = 1  # Empty first cell
    puzzle1[5] = 1  # Empty another
    puzzle1[10] = 1  # Empty another
    
    puzzles = puzzle1.unsqueeze(0).repeat(batch_size, 1).to(device)
    solutions = solution1.unsqueeze(0).repeat(batch_size, 1).to(device)
    mask = (puzzles == 1)
    
    batch = {
        'inputs': puzzles,
        'puzzle_identifiers': torch.zeros(batch_size, dtype=torch.long, device=device)
    }
    
    # Forward pass
    carry = model.initial_carry(batch)
    carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
    carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
    carry.halted = carry.halted.to(device)
    carry.steps = carry.steps.to(device)
    
    carry, outputs = model(carry, batch)
    
    # Get predictions
    logits = outputs['logits']
    preds = logits.argmax(dim=-1)
    
    print(f"  Puzzle shape: {puzzles.shape}")
    print(f"  Predictions shape: {preds.shape}")
    print(f"  Empty cells per puzzle: {mask[0].sum().item()}")
    
    # Compute validity for each sample
    validity_scores = []
    for b in range(batch_size):
        completed = puzzles[b].clone()
        completed[mask[b]] = preds[b][mask[b]]
        completed_digits = completed - 1  # Token space to digit space
        
        valid, total = count_prediction_validity(completed_digits.cpu(), mask[b].cpu(), size=4)
        validity = valid / total if total > 0 else 0.0
        validity_scores.append(validity)
        print(f"  Sample {b}: {valid}/{total} valid = {validity:.2%}")
    
    validity_target = torch.tensor(validity_scores, device=device, dtype=torch.float32)
    
    # Compute Q head loss - ORIGINAL TRM DESIGN
    # Binary: is sequence exactly correct?
    q_halt = outputs['q_halt_logits']
    # For this test, use binary target (all correct or not)
    seq_is_correct = validity_target  # 0.0 since random predictions are wrong
    q_loss = F.binary_cross_entropy_with_logits(q_halt.squeeze(-1), seq_is_correct)
    
    print(f"\n  seq_is_correct targets: {seq_is_correct.tolist()}")
    print(f"  Q halt logits: {q_halt.squeeze(-1).tolist()}")
    print(f"  Q head loss: {q_loss.item():.4f}")
    
    # Verify loss is reasonable
    assert not torch.isnan(q_loss), "Q loss is NaN!"
    assert not torch.isinf(q_loss), "Q loss is Inf!"
    
    print("  ✓ Validity-based Q loss computed correctly!")
    return True


def test_q_head_training_loop():
    """Test Q head in a mini training loop."""
    print("\n" + "=" * 60)
    print("TEST 4: Q Head Training Loop")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = {
        'batch_size': 1, 'seq_len': 16, 'vocab_size': 6,
        'hidden_size': 64, 'num_heads': 2, 'expansion': 2,  # Smaller for speed
        'H_cycles': 1, 'L_cycles': 2, 'H_layers': 0, 'L_layers': 2,
        'pos_encodings': 'rope', 'rms_norm_eps': 1e-5, 'rope_theta': 10000.0,
        'puzzle_emb_ndim': 0, 'puzzle_emb_len': 0, 'num_puzzle_identifiers': 1,
        'halt_max_steps': 16, 'halt_exploration_prob': 0.0, 'forward_dtype': 'float32',
        'detach_early_cycles': False,
    }
    
    model = TinyRecursiveReasoningModel_ACTV1(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Store initial Q head weights
    initial_q_weight = model.inner.q_head.weight.clone().detach()
    initial_q_bias = model.inner.q_head.bias.clone().detach()
    
    print(f"  Initial Q head bias: {initial_q_bias.tolist()}")
    
    # Training loop
    num_steps = 10
    batch_size = 8
    losses = []
    
    for step in range(num_steps):
        # Random puzzles
        puzzles = torch.randint(1, 6, (batch_size, 16)).to(device)
        solutions = torch.randint(2, 6, (batch_size, 16)).to(device)
        mask = (puzzles == 1)
        
        batch = {
            'inputs': puzzles,
            'puzzle_identifiers': torch.zeros(batch_size, dtype=torch.long, device=device)
        }
        
        # Forward
        carry = model.initial_carry(batch)
        carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
        carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
        carry.halted = carry.halted.to(device)
        carry.steps = carry.steps.to(device)
        
        model.train()
        carry, outputs = model(carry, batch)
        
        # CE loss
        logits = outputs['logits']
        ce_loss = F.cross_entropy(logits[mask], solutions[mask])
        
        # Q head loss - ORIGINAL TRM DESIGN
        # Train q_halt to predict binary "sequence is exactly correct"
        q_halt = outputs['q_halt_logits']
        seq_is_correct = (torch.rand(batch_size, device=device) > 0.5).float()  # Random binary
        q_loss = F.binary_cross_entropy_with_logits(q_halt.squeeze(-1), seq_is_correct)
        
        # Combined loss
        total_loss = ce_loss + 0.5 * q_loss
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        losses.append(total_loss.item())
    
    # Check Q head weights changed
    final_q_weight = model.inner.q_head.weight.clone().detach()
    final_q_bias = model.inner.q_head.bias.clone().detach()
    
    weight_changed = not torch.allclose(initial_q_weight, final_q_weight)
    bias_changed = not torch.allclose(initial_q_bias, final_q_bias)
    
    print(f"\n  Training steps: {num_steps}")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Q head weight changed: {weight_changed}")
    print(f"  Q head bias changed: {bias_changed}")
    print(f"  Final Q head bias: {final_q_bias.tolist()}")
    
    assert weight_changed or bias_changed, "Q head did not update during training!"
    
    print("  ✓ Q head updates correctly during training loop!")
    return True


def test_act_halting_during_eval():
    """Test that ACT halting works during evaluation (no training)."""
    print("\n" + "=" * 60)
    print("TEST 5: ACT Halting During Evaluation")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = {
        'batch_size': 1, 'seq_len': 16, 'vocab_size': 6,
        'hidden_size': 64, 'num_heads': 2, 'expansion': 2,
        'H_cycles': 1, 'L_cycles': 2, 'H_layers': 0, 'L_layers': 2,
        'pos_encodings': 'rope', 'rms_norm_eps': 1e-5, 'rope_theta': 10000.0,
        'puzzle_emb_ndim': 0, 'puzzle_emb_len': 0, 'num_puzzle_identifiers': 1,
        'halt_max_steps': 16, 'halt_exploration_prob': 0.0, 'forward_dtype': 'float32',
        'detach_early_cycles': False,
    }
    
    model = TinyRecursiveReasoningModel_ACTV1(config).to(device)
    
    # Manually set Q head to halt immediately (q_halt > 0)
    # ORIGINAL TRM DESIGN: halt if q_halt > 0
    with torch.no_grad():
        model.inner.q_head.bias[0] = 10.0  # High q_halt → always halt
    
    print(f"  Q head bias set to: halt={model.inner.q_head.bias[0].item():.1f} (threshold=0)")
    
    # Set to eval mode
    model.eval()
    
    # Forward pass
    batch_size = 1
    puzzles = torch.randint(1, 6, (batch_size, 16)).to(device)
    batch = {
        'inputs': puzzles,
        'puzzle_identifiers': torch.zeros(batch_size, dtype=torch.long, device=device)
    }
    
    carry = model.initial_carry(batch)
    carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
    carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
    carry.halted = carry.halted.to(device)
    carry.steps = carry.steps.to(device)
    
    MAX_ACT_STEPS = 16
    act_steps = 0
    
    with torch.no_grad():
        carry, outputs = model(carry, batch)
        act_steps = 1
        
        # ORIGINAL TRM DESIGN: halt if q_halt > 0
        q_halt = outputs['q_halt_logits']
        halted = (q_halt > 0)
        
        print(f"  Step 1: q_halt={q_halt.item():.2f}, halted={halted.item()}")
        
        while not halted.all() and act_steps < MAX_ACT_STEPS:
            carry, outputs = model(carry, batch)
            q_halt = outputs['q_halt_logits']
            halted = (q_halt > 0)
            act_steps += 1
            print(f"  Step {act_steps}: q_halt={q_halt.item():.2f}, halted={halted.item()}")
    
    print(f"\n  Total ACT steps: {act_steps}")
    
    # With bias set to halt immediately, should halt at step 1
    assert act_steps == 1, f"Expected 1 step (immediate halt), got {act_steps}"
    
    # Now test the opposite: set to never halt (q_halt < 0)
    with torch.no_grad():
        model.inner.q_head.bias[0] = -10.0  # Low q_halt → never halt
    
    print(f"\n  Q head bias set to: halt={model.inner.q_head.bias[0].item():.1f}")
    
    carry = model.initial_carry(batch)
    carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
    carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
    carry.halted = carry.halted.to(device)
    carry.steps = carry.steps.to(device)
    
    with torch.no_grad():
        carry, outputs = model(carry, batch)
        act_steps = 1
        
        q_halt = outputs['q_halt_logits']
        halted = (q_halt > 0)  # ORIGINAL: halt if q_halt > 0
        
        while not halted.all() and act_steps < MAX_ACT_STEPS:
            carry, outputs = model(carry, batch)
            q_halt = outputs['q_halt_logits']
            halted = (q_halt > 0)
            act_steps += 1
    
    print(f"  Total ACT steps (never halt): {act_steps}")
    
    # With bias set to never halt, should run max steps
    assert act_steps == MAX_ACT_STEPS, f"Expected {MAX_ACT_STEPS} steps (never halt), got {act_steps}"
    
    print("  ✓ ACT halting works correctly during evaluation!")
    return True


def run_all_tests():
    """Run all Q head tests."""
    print("\n" + "=" * 60)
    print("Q HEAD UNIT TESTS")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("Output Shape", test_q_head_output_shape()))
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results.append(("Output Shape", False))
    
    try:
        results.append(("Gradient Flow", test_q_head_gradient_flow()))
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results.append(("Gradient Flow", False))
    
    try:
        results.append(("Validity Loss", test_validity_based_q_loss()))
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results.append(("Validity Loss", False))
    
    try:
        results.append(("Training Loop", test_q_head_training_loop()))
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results.append(("Training Loop", False))
    
    try:
        results.append(("ACT Eval Halting", test_act_halting_during_eval()))
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        results.append(("ACT Eval Halting", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

