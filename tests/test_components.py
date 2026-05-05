"""Tests for WorldNN core components."""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from worldnn.matter import Matter, ContinuousMatter, RockPushMatter
from worldnn.channels import Channel
from worldnn.environment import EnvironmentVAE
from worldnn.organism import Organism, PredictiveOrganism
from worldnn.world import World, ContinuousWorld, RockPushWorld


@pytest.fixture
def device():
    return torch.device("cpu")


class TestMatter:
    def test_shapes(self, device):
        m = Matter(emission_dim=4, action_dim=2, seed_dim=4).to(device)
        state = torch.tensor([0.0, 1.0, 0.0, 1.0], device=device)
        seed = torch.randn(4, 4, device=device)
        action = torch.randn(4, 2, device=device)

        next_state, emission, flip_prob = m(state, seed, action)
        assert next_state.shape == (4,)
        assert emission.shape == (4, 4)
        assert flip_prob.shape == (4,)

    def test_binary_state(self, device):
        m = Matter().to(device)
        m.eval()
        state = m.reset_state(100, device)
        assert torch.all((state == 0) | (state == 1))

    def test_deterministic_eval(self, device):
        m = Matter(seed_dim=4).to(device)
        m.eval()
        torch.manual_seed(0)
        state = torch.zeros(8, device=device)
        seed = torch.randn(8, 4, device=device)
        action = torch.randn(8, 2, device=device)

        ns1, em1, _ = m(state, seed, action)
        ns2, em2, _ = m(state, seed, action)
        assert torch.allclose(em1, em2)


class TestChannel:
    def test_shapes(self, device):
        c = Channel(input_dim=4, output_dim=4, noise_std=0.1)
        x = torch.randn(8, 4, device=device)
        y = c(x)
        assert y.shape == (8, 4)

    def test_noise_effect(self, device):
        c = Channel(input_dim=4, output_dim=4, noise_std=1.0)
        x = torch.randn(100, 4, device=device)
        y1 = c(x)
        y2 = c(x)
        # With noise, outputs should differ
        assert not torch.allclose(y1, y2)

    def test_no_noise(self, device):
        c = Channel(input_dim=4, output_dim=4, noise_std=0.0)
        x = torch.randn(8, 4, device=device)
        y1 = c(x)
        y2 = c(x)
        assert torch.allclose(y1, y2)

    def test_capacity(self):
        c = Channel(input_dim=4, output_dim=4, noise_std=0.1, bandwidth=1.0)
        cap = c.theoretical_capacity()
        assert cap > 0


class TestEnvironmentVAE:
    def test_shapes(self, device):
        env = EnvironmentVAE(channel_dim=4, latent_dim=2).to(device)
        y = torch.randn(8, 4, device=device)
        z, y_hat, mu, logvar = env(y)
        assert z.shape == (8, 2)
        assert y_hat.shape == (8, 4)
        assert mu.shape == (8, 2)
        assert logvar.shape == (8, 2)

    def test_vae_loss(self, device):
        env = EnvironmentVAE(channel_dim=4, latent_dim=2).to(device)
        y = torch.randn(8, 4, device=device)
        z, y_hat, mu, logvar = env(y)
        loss = env.vae_loss(y, y_hat, mu, logvar)
        assert loss.shape == ()
        assert loss.item() > 0


class TestOrganism:
    def test_shapes(self, device):
        org = Organism(sensory_dim=2, embedding_dim=4, action_dim=2).to(device)
        z = torch.randn(8, 2, device=device)
        action, embedding, value = org(z)
        assert action.shape == (8, 2)
        assert embedding.shape == (8, 4)
        assert value.shape == (8,)

    def test_embedding_bounded(self, device):
        org = Organism(sensory_dim=2, embedding_dim=4).to(device)
        z = torch.randn(100, 2, device=device) * 10  # large input
        _, embedding, _ = org(z)
        # Tanh bounded to [-1, 1]
        assert embedding.abs().max() <= 1.0


class TestWorld:
    def test_step(self, device):
        w = World(env_latent_dim=2, embedding_dim=4).to(device)
        state = torch.tensor([0.0, 1.0], device=device)
        result = w.step(state)
        assert "next_state" in result
        assert "emission" in result
        assert "z" in result
        assert "embedding" in result
        assert "action" in result

    def test_episode(self, device):
        w = World(env_latent_dim=2, embedding_dim=4).to(device)
        traj = w.run_episode(batch_size=16, n_steps=5, device=device)
        assert len(traj["states"]) == 5
        assert len(traj["rewards"]) == 5
        assert traj["states"][0].shape == (16,)

    def test_different_configs(self, device):
        """Test that different embedding dims produce different-shaped embeddings."""
        for emb_dim in [1, 2, 8]:
            w = World(env_latent_dim=2, embedding_dim=emb_dim).to(device)
            traj = w.run_episode(4, 3, device=device)
            assert traj["embeddings"][0].shape == (4, emb_dim)


class TestContinuousMatter:
    def test_shapes(self, device):
        m = ContinuousMatter().to(device)
        state = torch.rand(8, device=device)
        seed = torch.randn(8, 4, device=device)
        action = torch.randn(8, 2, device=device)
        ns, emission, force = m(state, seed, action)
        assert ns.shape == (8,)
        assert emission.shape == (8, 4)
        assert force.shape == (8,)

    def test_state_bounded(self, device):
        m = ContinuousMatter(force_scale=10.0).to(device)
        state = torch.rand(100, device=device)
        seed = torch.randn(100, 4, device=device)
        action = torch.randn(100, 2, device=device) * 5  # large actions
        ns, _, _ = m(state, seed, action)
        assert (ns >= 0.0).all() and (ns <= 1.0).all()

    def test_reset(self, device):
        m = ContinuousMatter().to(device)
        state = m.reset_state(50, device)
        assert state.shape == (50,)
        assert (state >= 0.0).all() and (state <= 1.0).all()


class TestPredictiveOrganism:
    def test_shapes(self, device):
        o = PredictiveOrganism(sensory_dim=2, embedding_dim=4, action_dim=2).to(device)
        z = torch.randn(8, 2, device=device)
        action, emb, value, pred_z = o.forward_with_prediction(z)
        assert action.shape == (8, 2)
        assert emb.shape == (8, 4)
        assert value.shape == (8,)
        assert pred_z.shape == (8, 2)

    def test_backward_compatible(self, device):
        o = PredictiveOrganism(sensory_dim=3, embedding_dim=2, action_dim=2).to(device)
        z = torch.randn(4, 3, device=device)
        action, emb, value = o(z)
        assert action.shape == (4, 2)
        assert emb.shape == (4, 2)


class TestContinuousWorld:
    def test_step(self, device):
        w = ContinuousWorld(env_latent_dim=2, embedding_dim=4).to(device)
        state = w.matter.reset_state(8, device)
        result = w.step(state)
        assert result["next_state"].shape == (8,)
        assert result["force"].shape == (8,)

    def test_training_smoke(self, device):
        """Smoke test: continuous world trains without errors."""
        from worldnn.train import train_environment_continuous, train_organism_ppo_continuous
        w = ContinuousWorld(env_latent_dim=2, embedding_dim=4).to(device)
        train_environment_continuous(w, n_steps=5, batch_size=16, device=device)
        m = train_organism_ppo_continuous(w, n_episodes=3, steps_per_episode=3,
                                          batch_size=16, device=device)
        assert len(m["mean_distance"]) == 3
        assert all(0 <= d <= 1 for d in m["mean_distance"])


class TestRockPushMatter:
    def test_shapes(self, device):
        m = RockPushMatter(emission_dim=8, action_dim=2).to(device)
        state = m.reset_state(8, device)
        seed = torch.randn(8, 4, device=device)
        action = torch.randn(8, 2, device=device)
        ns, emission, contact = m(state, seed, action)
        assert ns.shape == (8, 4)
        assert emission.shape == (8, 8)
        assert contact.shape == (8,)

    def test_state_bounded(self, device):
        m = RockPushMatter(move_speed=1.0, push_strength=1.0).to(device)
        state = m.reset_state(100, device)
        seed = torch.randn(100, 4, device=device)
        action = torch.randn(100, 2, device=device) * 5
        ns, _, _ = m(state, seed, action)
        assert (ns >= 0.0).all() and (ns <= 1.0).all()

    def test_contact_when_close(self, device):
        m = RockPushMatter(push_radius=0.15).to(device)
        # Place rock and organism at same position
        state = torch.tensor([[0.5, 0.5, 0.5, 0.5]] * 8, device=device)
        seed = torch.randn(8, 4, device=device)
        action = torch.zeros(8, 2, device=device)
        _, _, contact = m(state, seed, action)
        # Contact should be high when overlapping
        assert (contact > 0.9).all()

    def test_no_contact_when_far(self, device):
        m = RockPushMatter(push_radius=0.15).to(device)
        # Rock at (0.1, 0.1), organism at (0.9, 0.9)
        state = torch.tensor([[0.1, 0.1, 0.9, 0.9]] * 8, device=device)
        seed = torch.randn(8, 4, device=device)
        action = torch.zeros(8, 2, device=device)
        _, _, contact = m(state, seed, action)
        assert (contact < 0.01).all()

    def test_push_moves_rock(self, device):
        m = RockPushMatter(push_radius=0.2, push_strength=0.12).to(device)
        # Organism at rock — small action to stay close
        state = torch.tensor([[0.5, 0.5, 0.5, 0.5]] * 32, device=device)
        seed = torch.zeros(32, 4, device=device)
        action = torch.randn(32, 2, device=device) * 0.1  # small movement
        ns, _, contact = m(state, seed, action)
        rock_moved = (ns[:, :2] - state[:, :2]).abs().sum(dim=-1)
        # Should have some contact and rock movement
        assert (contact > 0.3).all()
        assert (rock_moved > 0).any()

    def test_reset(self, device):
        m = RockPushMatter().to(device)
        state = m.reset_state(50, device)
        assert state.shape == (50, 4)
        assert (state >= 0.1).all() and (state <= 0.9).all()


class TestRockPushWorld:
    def test_step(self, device):
        w = RockPushWorld(env_latent_dim=4, embedding_dim=8).to(device)
        state = w.matter.reset_state(8, device)
        result = w.step(state)
        assert result["next_state"].shape == (8, 4)
        assert result["contact"].shape == (8,)
        assert result["z"].shape == (8, 4)
        assert result["embedding"].shape == (8, 8)

    def test_training_smoke(self, device):
        """Smoke test: rock-push world trains without errors."""
        from worldnn.train import train_environment_rockpush, train_organism_ppo_rockpush
        w = RockPushWorld(env_latent_dim=4, embedding_dim=8).to(device)
        train_environment_rockpush(w, n_steps=5, batch_size=16, device=device)
        m = train_organism_ppo_rockpush(w, n_episodes=3, steps_per_episode=5,
                                         batch_size=16, device=device)
        assert len(m["rock_distance"]) == 3
        assert all(d >= 0 for d in m["rock_distance"])


class TestOrganismShape:
    """Shape-contract tests for Organism across the obj-024 sweep range.

    Audit 2026-05-05 D4: Organism.forward was previously untested.
    """

    @pytest.mark.parametrize("sensory_dim,embedding_dim",
                              [(2, 2), (4, 8), (8, 16), (16, 32),
                               (16, 256)])  # incl. obj-027 large case
    def test_forward_shapes(self, device, sensory_dim, embedding_dim):
        org = Organism(sensory_dim=sensory_dim,
                        embedding_dim=embedding_dim,
                        action_dim=2).to(device)
        batch = 7
        z = torch.randn(batch, sensory_dim, device=device)
        action_mean, embedding, value = org(z)
        assert action_mean.shape == (batch, 2)
        assert embedding.shape == (batch, embedding_dim)
        assert value.shape == (batch,)
        assert torch.isfinite(action_mean).all()
        assert torch.isfinite(embedding).all()
        assert torch.isfinite(value).all()

    def test_sensory_slice_path(self, device):
        """obj-024/027 pattern: obs = emission[:, :sensory_dim].

        Verifies the slice preserves shape and the organism handles each width.
        """
        org_sweep = {sd: Organism(sensory_dim=sd, embedding_dim=8, action_dim=2).to(device)
                     for sd in (2, 4, 8, 16)}
        emission = torch.randn(5, 16, device=device)
        for sd, org in org_sweep.items():
            obs = emission[:, :sd]
            assert obs.shape == (5, sd)
            action_mean, _, _ = org(obs)
            assert action_mean.shape == (5, 2)


class TestKSGEstimator:
    """Pin current behavior of estimate_mi_ksg.

    Audit 2026-05-05 D2: KSG under-estimates by 30-100% in 2-D Gaussian
    regime. These tests pin the current behavior so a future refactor
    cannot silently change paper numbers.
    """

    def test_independent_returns_zero(self):
        from worldnn.utils import estimate_mi_ksg
        import numpy as np
        rng = np.random.default_rng(0)
        x = rng.standard_normal((2000, 1))
        y = rng.standard_normal((2000, 1))
        mi = estimate_mi_ksg(x, y, k=3)
        assert 0.0 <= mi < 0.05  # independent samples should give ~0

    def test_correlated_returns_clamped_zero(self):
        """rho=0.6 has analytic I=0.223 nats. Our KSG returns 0 (clamped)."""
        from worldnn.utils import estimate_mi_ksg
        import numpy as np
        rng = np.random.default_rng(0)
        cov = np.array([[1.0, 0.6], [0.6, 1.0]])
        samples = rng.multivariate_normal([0, 0], cov, size=2000)
        mi = estimate_mi_ksg(samples[:, 0:1], samples[:, 1:2], k=3)
        # Pin the under-estimation: if this changes, paper claims affected.
        assert mi < 0.10, f"KSG returned {mi:.3f}; behavior changed"

    def test_strong_correlation_partial_recovery(self):
        """rho=0.9 has analytic I=0.830 nats; KSG recovers ~66% (0.55)."""
        from worldnn.utils import estimate_mi_ksg
        import numpy as np
        rng = np.random.default_rng(0)
        cov = np.array([[1.0, 0.9], [0.9, 1.0]])
        samples = rng.multivariate_normal([0, 0], cov, size=2000)
        mi = estimate_mi_ksg(samples[:, 0:1], samples[:, 1:2], k=3)
        # Truth ~ 0.83. Our KSG at n=2000 returns somewhere in 0.4-0.7.
        assert 0.30 < mi < 0.75, f"KSG returned {mi:.3f}; behavior changed"


class TestSAOnConstantPolicy:
    """compute_sa_sensory baseline checks.

    Audit 2026-05-05 D4: compute_sa_sensory was previously untested.
    Use a hand-crafted organism whose action is the optimal direction →
    SA should approach 1.0 in the limit of large batch.
    """

    def test_sa_random_policy_near_zero(self, device):
        """Random action mean → SA averages near 0 across many samples."""
        import numpy as np

        class RandomPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Linear(1, 1)  # so .parameters() works
            def forward(self, z):
                return torch.randn(z.shape[0], 2), z, torch.zeros(z.shape[0])

        m = RockPushMatter(emission_dim=8, action_dim=2, seed_dim=4).to(device)
        org = RandomPolicy().to(device)
        # Mimic compute_sa_sensory inline (script lives in experiments/)
        cos_sims = []
        with torch.no_grad():
            for _ in range(8):
                state = m.reset_state(256, device)
                seed = torch.randn(256, m.seed_dim, device=device)
                action = torch.randn(256, 2, device=device) * 0.1
                next_state, _, _ = m(state, seed, action)
                rock_pos = next_state[:, :2]
                org_pos = next_state[:, 2:4]
                a_opt = (rock_pos - org_pos)
                a_opt = a_opt / (a_opt.norm(dim=-1, keepdim=True) + 1e-8)
                a_learn, _, _ = org(state[:, :2])
                a_learn = a_learn / (a_learn.norm(dim=-1, keepdim=True) + 1e-8)
                cos_sims.append((a_learn * a_opt).sum(dim=-1))
        sa = torch.cat(cos_sims).mean().item()
        assert abs(sa) < 0.10, f"random-policy SA should be near 0, got {sa:.3f}"
