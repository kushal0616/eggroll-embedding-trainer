import torch
import pytest
from src.eggroll.noise import Rank1NoiseGenerator, NoiseConfig
from src.eggroll.shaping import FitnessShaper, AntitheticShaper
from src.eggroll.update import EGGROLLUpdater, UpdateConfig


class TestNoiseGenerator:
    
    def test_sample_shape(self):
        config = NoiseConfig(population_size=64, sigma=0.02, seed=42)
        generator = Rank1NoiseGenerator(config, output_size=32, input_size=64, device="cpu")
        
        A, B = generator.sample(step=0)
        
        assert A.shape == (32, 32)
        assert B.shape == (32, 64)
    
    def test_deterministic_sampling(self):
        config = NoiseConfig(population_size=64, sigma=0.02, seed=42)
        gen1 = Rank1NoiseGenerator(config, output_size=32, input_size=64, device="cpu")
        gen2 = Rank1NoiseGenerator(config, output_size=32, input_size=64, device="cpu")
        
        A1, B1 = gen1.sample(step=0)
        A2, B2 = gen2.sample(step=0)
        
        assert torch.allclose(A1, A2)
        assert torch.allclose(B1, B2)
    
    def test_antithetic_signs(self):
        config = NoiseConfig(population_size=64, sigma=0.02, seed=42)
        generator = Rank1NoiseGenerator(config, output_size=32, device="cpu")
        
        signs = generator.get_antithetic_signs()
        
        assert signs.shape == (64,)
        assert (signs[:32] == 1).all()
        assert (signs[32:] == -1).all()


class TestFitnessShaper:
    
    def test_rank_transform_bounds(self):
        shaper = FitnessShaper(method="rank")
        fitness = torch.randn(100)
        
        shaped = shaper(fitness)
        
        assert shaped.min() == pytest.approx(-0.5, rel=1e-5)
        assert shaped.max() == pytest.approx(0.5, rel=1e-5)
    
    def test_zscore_normalization(self):
        shaper = FitnessShaper(method="zscore")
        fitness = torch.randn(100) * 10 + 5
        
        shaped = shaper(fitness)
        
        assert shaped.mean().item() == pytest.approx(0.0, abs=1e-5)
        assert shaped.std().item() == pytest.approx(1.0, rel=1e-2)


class TestAntitheticShaper:
    
    def test_antithetic_difference(self):
        shaper = AntitheticShaper()
        
        fitness = torch.tensor([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        delta = shaper(fitness)
        
        assert delta.shape == (4,)
        expected = (fitness[:4] - fitness[4:]) / 2
        assert torch.allclose(delta, expected)


class TestEGGROLLUpdater:
    
    def test_update_shape(self):
        config = UpdateConfig(learning_rate=0.1, clip_norm=1.0)
        updater = EGGROLLUpdater(config, output_size=32, input_size=64, device="cpu")
        
        W = torch.randn(32, 64)
        A = torch.randn(16, 32)
        B = torch.randn(16, 64)
        fitness = torch.randn(16)
        
        W_new = updater.apply_update(W, A, B, fitness, sigma=0.02)
        
        assert W_new.shape == (32, 64)
        assert not torch.allclose(W_new, W)
    
    def test_gradient_clipping(self):
        config = UpdateConfig(learning_rate=1.0, clip_norm=0.1)
        updater = EGGROLLUpdater(config, output_size=32, input_size=64, device="cpu")
        
        W = torch.zeros(32, 64)
        A = torch.ones(16, 32) * 10
        B = torch.ones(16, 64) * 10
        fitness = torch.ones(16)
        
        W_new = updater.apply_update(W, A, B, fitness, sigma=0.02)
        
        update_norm = torch.norm(W_new, p='fro').item()
        assert update_norm <= 0.1 + 1e-5
