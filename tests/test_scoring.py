import torch
import pytest
from src.eggroll.scoring import VectorizedScorer, ChunkedScorer


class TestVectorizedScorer:
    
    @pytest.fixture
    def scorer(self):
        return VectorizedScorer(sigma=0.1)
    
    def test_base_scores_shape(self, scorer):
        B, P, D_in, D_out = 4, 16, 64, 32
        H_q = torch.randn(B, D_in)
        H_d = torch.randn(B, P, D_in)
        W = torch.randn(D_out, D_in)
        
        q_base, d_base, scores_base = scorer.compute_base_scores(H_q, H_d, W)
        
        assert q_base.shape == (B, D_out)
        assert d_base.shape == (B, P, D_out)
        assert scores_base.shape == (B, P)
    
    def test_perturbed_scores_shape(self, scorer):
        B, P, D_in, D_out, M = 4, 16, 64, 32, 16
        N = 2 * M
        
        H_q = torch.randn(B, D_in)
        H_d = torch.randn(B, P, D_in)
        W = torch.randn(D_out, D_in)
        A = torch.randn(M, D_out)
        B_noise = torch.randn(M, D_in)
        
        scores = scorer.compute_all_scores(H_q, H_d, W, A, B_noise)
        
        assert scores.shape == (B, P, N)
    
    def test_identity_projection(self, scorer):
        B, P, D = 2, 4, 8
        H_q = torch.randn(B, D)
        H_d = torch.randn(B, P, D)
        W = torch.eye(D)
        
        q_base, d_base, scores_base = scorer.compute_base_scores(H_q, H_d, W)
        
        expected_scores = torch.einsum('bd,bpd->bp', H_q, H_d)
        assert torch.allclose(scores_base, expected_scores, atol=1e-5)


class TestChunkedScorer:
    
    def test_chunked_matches_full(self):
        B, P, D_in, D_out, M = 4, 16, 64, 32, 32
        
        H_q = torch.randn(B, D_in)
        H_d = torch.randn(B, P, D_in)
        W = torch.randn(D_out, D_in)
        A = torch.randn(M, D_out)
        B_noise = torch.randn(M, D_in)
        
        full_scorer = VectorizedScorer(sigma=0.1)
        chunked_scorer = ChunkedScorer(sigma=0.1, chunk_size=8)
        
        full_scores = full_scorer.compute_all_scores(H_q, H_d, W, A, B_noise)
        chunked_scores = chunked_scorer.compute_all_scores_chunked(H_q, H_d, W, A, B_noise)
        
        assert torch.allclose(full_scores, chunked_scores, atol=1e-5)
