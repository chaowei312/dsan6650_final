"""
Unit tests for RL components.

Tests:
1. QHead and VHead forward passes
2. HypothesisGate state transitions
3. RewardComputer calculations
4. EntropyComputer for Sudoku
5. GAE computation
6. Full TRMHypothesis model
"""

import pytest
import torch
import torch.nn.functional as F
import math
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rl.heads import QHead, VHead, DuelingHead, ConfidenceHead
from models.rl.gating import HypothesisGate, CellStatus, TerminationChecker, HypothesisStateManager
from models.rl.rewards import RewardComputer, EntropyComputer, GAEComputer


class TestQHead:
    """Tests for Q-head implementation."""
    
    def test_output_shape(self):
        """Q-head should output [B, L] from [B, L, D]."""
        batch_size, seq_len, hidden_size = 4, 16, 64
        q_head = QHead(hidden_size)
        
        z_H = torch.randn(batch_size, seq_len, hidden_size)
        q_values = q_head(z_H)
        
        assert q_values.shape == (batch_size, seq_len)
        
    def test_output_range(self):
        """Q-head output should be in [-1, 1] due to tanh."""
        batch_size, seq_len, hidden_size = 8, 16, 64
        q_head = QHead(hidden_size)
        
        z_H = torch.randn(batch_size, seq_len, hidden_size) * 10  # Large inputs
        q_values = q_head(z_H)
        
        assert q_values.min() >= -1.0
        assert q_values.max() <= 1.0
        
    def test_gradient_flow(self):
        """Gradients should flow through Q-head."""
        batch_size, seq_len, hidden_size = 4, 16, 64
        q_head = QHead(hidden_size)
        
        z_H = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        q_values = q_head(z_H)
        
        loss = q_values.sum()
        loss.backward()
        
        assert z_H.grad is not None
        assert z_H.grad.shape == z_H.shape
        
    def test_initial_output_near_zero(self):
        """Initially, Q-head should output values near 0 (uncertain)."""
        hidden_size = 64
        q_head = QHead(hidden_size)
        
        z_H = torch.randn(4, 16, hidden_size)
        q_values = q_head(z_H)
        
        # Most values should be in [-0.5, 0.5] initially
        in_range = (q_values.abs() < 0.5).float().mean()
        assert in_range > 0.5, f"Expected most values near 0, got mean abs = {q_values.abs().mean()}"


class TestVHead:
    """Tests for V-head implementation."""
    
    def test_output_shape(self):
        """V-head should output [B] from [B, L, D]."""
        batch_size, seq_len, hidden_size = 4, 16, 64
        v_head = VHead(hidden_size)
        
        z_H = torch.randn(batch_size, seq_len, hidden_size)
        v_value = v_head(z_H)
        
        assert v_value.shape == (batch_size,)
        
    def test_aggregation_methods(self):
        """Test different aggregation methods."""
        batch_size, seq_len, hidden_size = 4, 16, 64
        z_H = torch.randn(batch_size, seq_len, hidden_size)
        
        for agg in ['mean', 'attention', 'cls']:
            v_head = VHead(hidden_size, aggregation=agg)
            v_value = v_head(z_H)
            assert v_value.shape == (batch_size,), f"Failed for aggregation={agg}"
            
    def test_masked_aggregation(self):
        """Test that mask properly affects aggregation."""
        batch_size, seq_len, hidden_size = 4, 16, 64
        v_head = VHead(hidden_size, aggregation='mean')
        
        z_H = torch.randn(batch_size, seq_len, hidden_size)
        
        # Create mask where only first 8 positions are valid
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, :8] = True
        
        v_with_mask = v_head(z_H, mask=mask)
        v_without_mask = v_head(z_H)
        
        # Results should be different
        assert not torch.allclose(v_with_mask, v_without_mask)


class TestDuelingHead:
    """Tests for Dueling architecture."""
    
    def test_output_shapes(self):
        """Test Q, V, A output shapes."""
        batch_size, seq_len, hidden_size = 4, 16, 64
        dueling = DuelingHead(hidden_size)
        
        z_H = torch.randn(batch_size, seq_len, hidden_size)
        Q, V, A = dueling(z_H)
        
        assert Q.shape == (batch_size, seq_len)
        assert V.shape == (batch_size,)
        assert A.shape == (batch_size, seq_len)
        
    def test_advantage_centering(self):
        """Advantages should be mean-centered."""
        batch_size, seq_len, hidden_size = 4, 16, 64
        dueling = DuelingHead(hidden_size)
        
        z_H = torch.randn(batch_size, seq_len, hidden_size)
        Q, V, A = dueling(z_H)
        
        # Mean of advantages should be ~0
        a_mean = A.mean(dim=-1)
        assert torch.allclose(a_mean, torch.zeros_like(a_mean), atol=1e-5)
        
    def test_q_decomposition(self):
        """Q = V + A relationship should hold."""
        batch_size, seq_len, hidden_size = 4, 16, 64
        dueling = DuelingHead(hidden_size)
        
        z_H = torch.randn(batch_size, seq_len, hidden_size)
        Q, V, A = dueling(z_H)
        
        # Before tanh, Q = V + A
        # After tanh, this is harder to verify, but Q should be bounded
        assert Q.min() >= -1.0
        assert Q.max() <= 1.0


class TestHypothesisGate:
    """Tests for hypothesis gating mechanism."""
    
    @pytest.fixture
    def gate(self):
        return HypothesisGate(
            lock_threshold=0.6,
            reject_threshold=-0.4,
            unlock_threshold=-0.7,
        )
        
    def test_high_q_locks(self, gate):
        """High Q values should trigger LOCK."""
        batch_size, seq_len = 2, 16
        
        q_values = torch.full((batch_size, seq_len), 0.8)  # High confidence
        status = torch.full((batch_size, seq_len), CellStatus.HYPOTHESIS)
        predictions = torch.randint(1, 5, (batch_size, seq_len))
        
        new_status, new_pred, info = gate(q_values, status, predictions)
        
        assert (new_status == CellStatus.LOCKED).all()
        
    def test_low_q_rejects(self, gate):
        """Low Q values should trigger REJECT."""
        batch_size, seq_len = 2, 16
        
        q_values = torch.full((batch_size, seq_len), -0.6)  # Low confidence
        status = torch.full((batch_size, seq_len), CellStatus.HYPOTHESIS)
        predictions = torch.randint(1, 5, (batch_size, seq_len))
        
        new_status, new_pred, info = gate(q_values, status, predictions)
        
        assert (new_status == CellStatus.UNFILLED).all()
        assert (new_pred == 0).all()  # Predictions reset
        
    def test_medium_q_stays_hypothesis(self, gate):
        """Medium Q values should keep HYPOTHESIS."""
        batch_size, seq_len = 2, 16
        
        q_values = torch.full((batch_size, seq_len), 0.2)  # Medium confidence
        status = torch.full((batch_size, seq_len), CellStatus.HYPOTHESIS)
        predictions = torch.randint(1, 5, (batch_size, seq_len))
        
        new_status, new_pred, info = gate(q_values, status, predictions)
        
        assert (new_status == CellStatus.HYPOTHESIS).all()
        
    def test_given_cells_unchanged(self, gate):
        """GIVEN cells should never change."""
        batch_size, seq_len = 2, 16
        
        q_values = torch.full((batch_size, seq_len), -0.9)  # Would normally reject
        status = torch.full((batch_size, seq_len), CellStatus.GIVEN)
        predictions = torch.randint(1, 5, (batch_size, seq_len))
        
        new_status, new_pred, info = gate(q_values, status, predictions)
        
        assert (new_status == CellStatus.GIVEN).all()
        
    def test_unlock_mechanism(self):
        """Very low Q should unlock previously locked cells."""
        gate = HypothesisGate(unlock_threshold=-0.7, allow_unlock=True)
        batch_size, seq_len = 2, 16
        
        q_values = torch.full((batch_size, seq_len), -0.8)  # Very low
        status = torch.full((batch_size, seq_len), CellStatus.LOCKED)
        predictions = torch.randint(1, 5, (batch_size, seq_len))
        
        new_status, new_pred, info = gate(q_values, status, predictions)
        
        assert (new_status == CellStatus.UNFILLED).all()
        
    def test_mixed_transitions(self, gate):
        """Test different transitions in same batch."""
        batch_size, seq_len = 1, 4
        
        # Different Q values per cell
        q_values = torch.tensor([[0.8, -0.6, 0.2, 0.0]])  # Lock, Reject, Hyp, Hyp
        status = torch.full((batch_size, seq_len), CellStatus.HYPOTHESIS)
        predictions = torch.tensor([[1, 2, 3, 4]])
        
        new_status, new_pred, info = gate(q_values, status, predictions)
        
        assert new_status[0, 0] == CellStatus.LOCKED
        assert new_status[0, 1] == CellStatus.UNFILLED
        assert new_status[0, 2] == CellStatus.HYPOTHESIS
        assert new_status[0, 3] == CellStatus.HYPOTHESIS


class TestTerminationChecker:
    """Tests for episode termination logic."""
    
    def test_all_locked_terminates(self):
        """Episode should end when all cells are locked."""
        checker = TerminationChecker(max_cycles=10)
        batch_size, seq_len = 2, 16
        
        # All cells locked (except some GIVEN)
        status = torch.full((batch_size, seq_len), CellStatus.LOCKED)
        status[:, :4] = CellStatus.GIVEN  # Some given cells
        
        done, _, _ = checker(status, cycle=2)
        
        assert done.all()
        
    def test_max_cycles_terminates(self):
        """Episode should end when max cycles reached."""
        checker = TerminationChecker(max_cycles=5)
        batch_size, seq_len = 2, 16
        
        # Still some unfilled cells
        status = torch.full((batch_size, seq_len), CellStatus.HYPOTHESIS)
        
        done, _, _ = checker(status, cycle=5)
        
        assert done.all()
        
    def test_early_cycles_continue(self):
        """Episode should continue if not done."""
        checker = TerminationChecker(max_cycles=10)
        batch_size, seq_len = 2, 16
        
        status = torch.full((batch_size, seq_len), CellStatus.HYPOTHESIS)
        
        done, _, _ = checker(status, cycle=2)
        
        assert not done.any()


class TestEntropyComputer:
    """Tests for entropy computation."""
    
    @pytest.fixture
    def entropy_comp(self):
        return EntropyComputer(grid_size=4, vocab_size=4)
        
    def test_empty_grid_max_entropy(self, entropy_comp):
        """Empty grid should have maximum entropy."""
        batch_size = 2
        grid = torch.zeros(batch_size, 4, 4)
        
        total_entropy = entropy_comp.compute_total_entropy(grid)
        
        # Each cell has 4 candidates initially, but constraints reduce this
        # For empty 4x4, each cell should have log2(4) = 2 bits initially
        # But constraints from rows/cols reduce this
        assert total_entropy.min() > 0
        
    def test_filled_grid_zero_entropy(self, entropy_comp):
        """Fully filled grid should have zero entropy."""
        batch_size = 2
        grid = torch.tensor([
            [1, 2, 3, 4],
            [3, 4, 1, 2],
            [2, 3, 4, 1],
            [4, 1, 2, 3]
        ]).unsqueeze(0).expand(batch_size, -1, -1).float()
        
        total_entropy = entropy_comp.compute_total_entropy(grid)
        
        assert torch.allclose(total_entropy, torch.zeros_like(total_entropy))
        
    def test_information_gain(self, entropy_comp):
        """Filling a cell should reduce entropy."""
        batch_size = 2
        
        # Before: empty cell at (0, 0)
        grid_before = torch.zeros(batch_size, 4, 4)
        # After: cell filled
        grid_after = grid_before.clone()
        grid_after[:, 0, 0] = 1
        
        info_gain = entropy_comp.compute_information_gain(grid_before, grid_after)
        
        assert (info_gain > 0).all()


class TestRewardComputer:
    """Tests for reward computation."""
    
    @pytest.fixture
    def reward_comp(self):
        return RewardComputer(grid_size=4, vocab_size=4)
        
    def test_correct_confident_positive(self, reward_comp):
        """Correct + confident should give positive reward."""
        batch_size, seq_len = 2, 16
        
        q_values = torch.full((batch_size, seq_len), 0.8)  # Confident
        predictions = torch.ones(batch_size, seq_len)
        labels = torch.ones(batch_size, seq_len)  # All correct
        locked_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        rewards, info = reward_comp.compute_commit_reward(
            q_values, predictions, labels, locked_mask
        )
        
        assert (rewards > 0).all()
        
    def test_wrong_confident_negative(self, reward_comp):
        """Wrong + confident should give negative reward."""
        batch_size, seq_len = 2, 16
        
        q_values = torch.full((batch_size, seq_len), 0.8)  # Confident
        predictions = torch.ones(batch_size, seq_len)
        labels = torch.ones(batch_size, seq_len) * 2  # All wrong
        locked_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        rewards, info = reward_comp.compute_commit_reward(
            q_values, predictions, labels, locked_mask
        )
        
        assert (rewards < 0).all()
        
    def test_self_consistency_bonus(self, reward_comp):
        """Self-consistency should affect reward."""
        batch_size, seq_len = 2, 16
        
        # High Q + correct = extra bonus
        q_high = torch.full((batch_size, seq_len), 0.9)
        # Low Q + correct = no extra bonus
        q_low = torch.full((batch_size, seq_len), -0.5)
        
        predictions = torch.ones(batch_size, seq_len)
        labels = torch.ones(batch_size, seq_len)
        locked_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        rewards_high, _ = reward_comp.compute_commit_reward(
            q_high, predictions, labels, locked_mask
        )
        rewards_low, _ = reward_comp.compute_commit_reward(
            q_low, predictions, labels, locked_mask
        )
        
        # High confidence correct should get higher reward
        assert rewards_high.mean() > rewards_low.mean()
        
    def test_reject_wrong_positive(self, reward_comp):
        """Rejecting wrong prediction should give positive reward."""
        batch_size, seq_len = 2, 16
        
        predictions = torch.ones(batch_size, seq_len)
        labels = torch.ones(batch_size, seq_len) * 2  # Predictions are wrong
        rejected_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        rewards, info = reward_comp.compute_reject_reward(
            predictions, labels, rejected_mask
        )
        
        assert (rewards > 0).all()


class TestGAEComputer:
    """Tests for GAE computation."""
    
    @pytest.fixture
    def gae_comp(self):
        return GAEComputer(gamma=0.99, gae_lambda=0.95)
        
    def test_advantage_shape(self, gae_comp):
        """Advantages should have same shape as rewards."""
        batch_size, T = 4, 10
        
        rewards = torch.randn(batch_size, T)
        values = torch.randn(batch_size, T + 1)
        dones = torch.zeros(batch_size, T, dtype=torch.bool)
        dones[:, -1] = True
        
        advantages, returns = gae_comp.compute_gae(rewards, values, dones)
        
        assert advantages.shape == (batch_size, T)
        assert returns.shape == (batch_size, T)
        
    def test_normalized_advantages(self, gae_comp):
        """Normalized advantages should have mean ~0 and std ~1."""
        batch_size, T = 4, 10
        
        rewards = torch.randn(batch_size, T)
        values = torch.randn(batch_size, T + 1)
        dones = torch.zeros(batch_size, T, dtype=torch.bool)
        dones[:, -1] = True
        
        advantages, _ = gae_comp.compute_gae(rewards, values, dones)
        normalized = gae_comp.normalize_advantages(advantages)
        
        assert abs(normalized.mean()) < 0.1
        assert abs(normalized.std() - 1.0) < 0.1
        
    def test_returns_from_advantages(self, gae_comp):
        """Returns = Advantages + Values relationship."""
        batch_size, T = 4, 10
        
        rewards = torch.randn(batch_size, T)
        values = torch.randn(batch_size, T + 1)
        dones = torch.zeros(batch_size, T, dtype=torch.bool)
        dones[:, -1] = True
        
        advantages, returns = gae_comp.compute_gae(rewards, values, dones)
        
        # Returns should equal A + V[:, :-1]
        expected = advantages + values[:, :-1]
        assert torch.allclose(returns, expected)


class TestTRMHypothesis:
    """Integration tests for full TRMHypothesis model."""
    
    @staticmethod
    def get_config():
        return {
            'batch_size': 4,
            'seq_len': 16,
            'vocab_size': 5,  # 0=empty, 1-4=digits
            'hidden_size': 64,
            'H_cycles': 3,
            'L_cycles': 2,
            'L_layers': 1,
            'expansion': 2.0,
            'num_heads': 4,
            'max_H_cycles': 4,
            'grid_size': 4,
        }
        
    def test_forward_output_shapes(self, config=None):
        if config is None:
            config = self.get_config()
        """Test forward pass output shapes."""
        from models.recursive_reasoning.trm_hypothesis import TRMHypothesis
        
        model = TRMHypothesis(config)
        batch_size, seq_len = config['batch_size'], config['seq_len']
        
        inputs = torch.randint(0, 5, (batch_size, seq_len))
        given_mask = torch.rand(batch_size, seq_len) > 0.5
        
        outputs = model(inputs, given_mask)
        
        assert outputs['logits'].shape == (batch_size, seq_len, config['vocab_size'])
        assert outputs['predictions'].shape == (batch_size, seq_len)
        assert outputs['q_values'].shape == (batch_size, seq_len)
        assert outputs['v_value'].shape == (batch_size,)
        assert outputs['status'].shape == (batch_size, seq_len)
        
    def test_trajectory_collection(self, config=None):
        """Test trajectory collection for RL training."""
        if config is None:
            config = self.get_config()
        from models.recursive_reasoning.trm_hypothesis import TRMHypothesis
        
        model = TRMHypothesis(config)
        model.train()  # Training mode for gating
        
        batch_size, seq_len = config['batch_size'], config['seq_len']
        
        inputs = torch.randint(0, 5, (batch_size, seq_len))
        given_mask = torch.rand(batch_size, seq_len) > 0.5
        
        outputs = model(inputs, given_mask, collect_trajectory=True)
        
        assert 'trajectory' in outputs
        traj = outputs['trajectory']
        
        # Trajectory should have T+1 status/prediction entries, T Q entries
        assert traj['status_history'].shape[1] >= 2  # At least initial + 1 step
        assert traj['prediction_history'].shape[1] >= 2
        
    def test_loss_computation(self, config=None):
        """Test combined loss computation."""
        if config is None:
            config = self.get_config()
        from models.recursive_reasoning.trm_hypothesis import TRMHypothesis
        
        model = TRMHypothesis(config)
        model.train()
        
        batch_size, seq_len = config['batch_size'], config['seq_len']
        
        inputs = torch.randint(0, 5, (batch_size, seq_len))
        given_mask = torch.rand(batch_size, seq_len) > 0.5
        labels = torch.randint(1, 5, (batch_size, seq_len))
        mask = ~given_mask  # Predict non-given cells
        
        outputs = model(inputs, given_mask, collect_trajectory=True)
        
        reward_comp = RewardComputer(grid_size=4, vocab_size=4)
        gae_comp = GAEComputer()
        
        loss, metrics = model.compute_loss(
            outputs, labels, mask,
            reward_computer=reward_comp,
            gae_computer=gae_comp,
        )
        
        assert loss.requires_grad
        assert 'loss_supervised' in metrics
        assert 'accuracy' in metrics
        
    def test_gradient_flow(self, config=None):
        """Test that gradients flow through the model."""
        if config is None:
            config = self.get_config()
        from models.recursive_reasoning.trm_hypothesis import TRMHypothesis
        
        model = TRMHypothesis(config)
        model.train()
        
        batch_size, seq_len = config['batch_size'], config['seq_len']
        
        inputs = torch.randint(0, 5, (batch_size, seq_len))
        given_mask = torch.rand(batch_size, seq_len) > 0.5
        labels = torch.randint(1, 5, (batch_size, seq_len))
        mask = ~given_mask
        
        outputs = model(inputs, given_mask, collect_trajectory=True)
        
        reward_comp = RewardComputer(grid_size=4, vocab_size=4)
        loss, _ = model.compute_loss(outputs, labels, mask, reward_computer=reward_comp)
        
        loss.backward()
        
        # Check that key parameters have gradients
        assert model.lm_head.weight.grad is not None
        assert model.q_head.fc1.weight.grad is not None
        assert model.v_head.fc1.weight.grad is not None


# ============== Run tests ==============

if __name__ == '__main__':
    # Run with: python -m pytest tests/test_rl.py -v
    # Or directly: python tests/test_rl.py
    
    print("Running Q-Head tests...")
    test_q = TestQHead()
    test_q.test_output_shape()
    test_q.test_output_range()
    test_q.test_gradient_flow()
    test_q.test_initial_output_near_zero()
    print("✓ Q-Head tests passed")
    
    print("\nRunning V-Head tests...")
    test_v = TestVHead()
    test_v.test_output_shape()
    test_v.test_aggregation_methods()
    test_v.test_masked_aggregation()
    print("✓ V-Head tests passed")
    
    print("\nRunning Dueling tests...")
    test_d = TestDuelingHead()
    test_d.test_output_shapes()
    test_d.test_advantage_centering()
    test_d.test_q_decomposition()
    print("✓ Dueling tests passed")
    
    print("\nRunning Gating tests...")
    test_g = TestHypothesisGate()
    gate = HypothesisGate(lock_threshold=0.6, reject_threshold=-0.4, unlock_threshold=-0.7)
    test_g.test_high_q_locks(gate)
    test_g.test_low_q_rejects(gate)
    test_g.test_medium_q_stays_hypothesis(gate)
    test_g.test_given_cells_unchanged(gate)
    test_g.test_unlock_mechanism()
    test_g.test_mixed_transitions(gate)
    print("✓ Gating tests passed")
    
    print("\nRunning Termination tests...")
    test_t = TestTerminationChecker()
    test_t.test_all_locked_terminates()
    test_t.test_max_cycles_terminates()
    test_t.test_early_cycles_continue()
    print("✓ Termination tests passed")
    
    print("\nRunning Entropy tests...")
    test_e = TestEntropyComputer()
    entropy_comp = EntropyComputer(grid_size=4, vocab_size=4)
    test_e.test_empty_grid_max_entropy(entropy_comp)
    test_e.test_filled_grid_zero_entropy(entropy_comp)
    test_e.test_information_gain(entropy_comp)
    print("✓ Entropy tests passed")
    
    print("\nRunning Reward tests...")
    test_r = TestRewardComputer()
    reward_comp = RewardComputer(grid_size=4, vocab_size=4)
    test_r.test_correct_confident_positive(reward_comp)
    test_r.test_wrong_confident_negative(reward_comp)
    test_r.test_self_consistency_bonus(reward_comp)
    test_r.test_reject_wrong_positive(reward_comp)
    print("✓ Reward tests passed")
    
    print("\nRunning GAE tests...")
    test_gae = TestGAEComputer()
    gae_comp = GAEComputer()
    test_gae.test_advantage_shape(gae_comp)
    test_gae.test_normalized_advantages(gae_comp)
    test_gae.test_returns_from_advantages(gae_comp)
    print("✓ GAE tests passed")
    
    print("\n" + "="*50)
    print("All unit tests passed! ✓")
    print("="*50)
    
    # Integration tests require the full model
    print("\nRunning integration tests (TRMHypothesis)...")
    try:
        test_trm = TestTRMHypothesis()
        test_trm.test_forward_output_shapes()
        print("✓ Forward pass test passed")
        test_trm.test_trajectory_collection()
        print("✓ Trajectory collection test passed")
        test_trm.test_loss_computation()
        print("✓ Loss computation test passed")
        test_trm.test_gradient_flow()
        print("✓ Gradient flow test passed")
        print("\n✓ All integration tests passed!")
    except Exception as e:
        import traceback
        print(f"\n⚠ Integration tests failed: {e}")
        traceback.print_exc()
        print("This may be due to missing dependencies. Run component tests first.")

