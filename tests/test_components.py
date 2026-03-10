"""Tests for WorldNN core components."""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from worldnn.matter import Matter
from worldnn.channels import Channel
from worldnn.environment import EnvironmentVAE
from worldnn.organism import Organism
from worldnn.world import World


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
