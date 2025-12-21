import torch
import pytest
from src.eggroll.ndcg import NDCGComputer, CachedNDCGComputer


class TestNDCGComputer:
    
    @pytest.fixture
    def ndcg_computer(self):
        return NDCGComputer(k=3, device="cpu")
    
    def test_perfect_ranking(self, ndcg_computer):
        scores = torch.tensor([[3.0, 2.0, 1.0, 0.0]]).unsqueeze(-1)
        relevance = torch.tensor([[2, 1, 1, 0]])
        
        ndcg_mean, _ = ndcg_computer.compute_ndcg(scores, relevance)
        assert ndcg_mean.item() == pytest.approx(1.0, rel=1e-5)
    
    def test_worst_ranking(self, ndcg_computer):
        scores = torch.tensor([[0.0, 1.0, 2.0, 3.0]]).unsqueeze(-1)
        relevance = torch.tensor([[2, 1, 1, 0]])
        
        ndcg_mean, _ = ndcg_computer.compute_ndcg(scores, relevance)
        assert ndcg_mean.item() < 1.0
    
    def test_batch_computation(self, ndcg_computer):
        B, P, N = 4, 10, 8
        scores = torch.randn(B, P, N)
        relevance = torch.randint(0, 3, (B, P))
        
        ndcg_mean, ndcg_per_query = ndcg_computer.compute_ndcg(scores, relevance)
        
        assert ndcg_mean.shape == (N,)
        assert ndcg_per_query.shape == (B, N)
        assert (ndcg_mean >= 0).all() and (ndcg_mean <= 1).all()
    
    def test_idcg_computation(self, ndcg_computer):
        relevance = torch.tensor([[2, 1, 0, 0]])
        idcg = ndcg_computer.compute_idcg(relevance)
        
        expected = (2**2 - 1) / 1.0 + (2**1 - 1) / 1.585 + (2**0 - 1) / 2.0
        assert idcg.item() == pytest.approx(expected, rel=1e-2)


class TestCachedNDCGComputer:
    
    def test_caching(self):
        computer = CachedNDCGComputer(k=3, device="cpu")
        
        query_ids = ["q1", "q2"]
        relevance = torch.tensor([[2, 1, 0], [1, 1, 0]])
        
        computer.cache_idcg(query_ids, relevance)
        
        cached = computer.get_cached_idcg(query_ids)
        expected = computer.compute_idcg(relevance)
        
        assert torch.allclose(cached, expected)
