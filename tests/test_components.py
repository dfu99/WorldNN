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
