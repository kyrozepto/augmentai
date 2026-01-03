"""
Tests for AutoSearch module.
"""

import pytest
from pathlib import Path

from augmentai.core.policy import Policy, Transform
from augmentai.search import PolicySampler, PolicyEvaluator, PolicyOptimizer, SearchResult
from augmentai.search.optimizer import OptimizerConfig, quick_search
from augmentai.search.result import GenerationStats


class TestPolicySampler:
    """Test policy candidate generation."""
    
    def test_sample_generates_valid_policies(self):
        """Sampler generates valid policies for domain."""
        sampler = PolicySampler(seed=42)
        
        policies = sampler.sample("natural", n=5)
        
        assert len(policies) == 5
        for p in policies:
            assert isinstance(p, Policy)
            assert p.domain == "natural"
            assert len(p.transforms) >= 2
    
    def test_sample_respects_domain_rules(self):
        """Generated policies respect domain constraints."""
        sampler = PolicySampler(seed=42)
        
        policies = sampler.sample("medical", n=10)
        
        # Medical domain should not have ElasticTransform
        for p in policies:
            transform_names = [t.name for t in p.transforms]
            assert "ElasticTransform" not in transform_names
            assert "ColorJitter" not in transform_names
    
    def test_mutate_creates_variation(self):
        """Mutation creates policy variation."""
        sampler = PolicySampler(seed=42)
        
        original = Policy(
            name="test",
            domain="natural",
            transforms=[
                Transform("HorizontalFlip", 0.5),
                Transform("Rotate", 0.5, parameters={"limit": 15}),
            ]
        )
        
        mutated = sampler.mutate(original, strength=0.5)
        
        assert mutated.name != original.name
        # Should be different in some way
        orig_probs = [t.probability for t in original.transforms]
        mut_probs = [t.probability for t in mutated.transforms]
        assert orig_probs != mut_probs or len(original.transforms) != len(mutated.transforms)
    
    def test_crossover_combines_parents(self):
        """Crossover combines transforms from parents."""
        sampler = PolicySampler(seed=42)
        
        parent1 = Policy(
            name="p1",
            domain="natural",
            transforms=[
                Transform("HorizontalFlip", 0.5),
                Transform("Rotate", 0.5),
            ]
        )
        parent2 = Policy(
            name="p2",
            domain="natural",
            transforms=[
                Transform("GaussNoise", 0.3),
                Transform("GaussianBlur", 0.4),
            ]
        )
        
        child = sampler.crossover(parent1, parent2)
        
        assert child.domain == "natural"
        assert len(child.transforms) >= 2


class TestPolicyEvaluator:
    """Test policy evaluation."""
    
    def test_evaluate_returns_score(self):
        """Evaluator returns score between 0 and 1."""
        evaluator = PolicyEvaluator(domain="natural")
        
        policy = Policy(
            name="test",
            domain="natural",
            transforms=[
                Transform("HorizontalFlip", 0.5),
                Transform("Rotate", 0.5, parameters={"limit": 15}),
                Transform("RandomBrightnessContrast", 0.3),
            ]
        )
        
        result = evaluator.evaluate(policy)
        
        assert 0.0 <= result.score <= 1.0
        assert "diversity" in result.metrics
        assert "coverage" in result.metrics
    
    def test_diverse_policy_scores_higher(self):
        """Policy with diverse transforms scores higher on diversity."""
        evaluator = PolicyEvaluator(domain="natural")
        
        # Diverse policy (multiple categories)
        diverse = Policy(
            name="diverse",
            domain="natural",
            transforms=[
                Transform("HorizontalFlip", 0.5),  # geometric
                Transform("RandomBrightnessContrast", 0.3),  # color
                Transform("GaussNoise", 0.2),  # noise
                Transform("GaussianBlur", 0.2),  # blur
            ]
        )
        
        # Homogeneous policy (same category)
        homogeneous = Policy(
            name="homogeneous",
            domain="natural",
            transforms=[
                Transform("HorizontalFlip", 0.5),
                Transform("VerticalFlip", 0.5),
                Transform("Rotate", 0.5),
                Transform("RandomCrop", 0.5),
            ]
        )
        
        diverse_result = evaluator.evaluate(diverse)
        homogeneous_result = evaluator.evaluate(homogeneous)
        
        assert diverse_result.metrics["diversity"] > homogeneous_result.metrics["diversity"]
    
    def test_batch_evaluate(self):
        """Batch evaluation works correctly."""
        evaluator = PolicyEvaluator(domain="natural")
        
        policies = [
            Policy(name=f"test_{i}", domain="natural", transforms=[
                Transform("HorizontalFlip", 0.5),
            ])
            for i in range(5)
        ]
        
        results = evaluator.evaluate_batch(policies)
        
        assert len(results) == 5
        for r in results:
            assert 0.0 <= r.score <= 1.0


class TestPolicyOptimizer:
    """Test evolutionary optimization."""
    
    def test_search_returns_result(self):
        """Search returns valid SearchResult."""
        config = OptimizerConfig(
            population_size=5,
            generations=3,
            seed=42,
        )
        optimizer = PolicyOptimizer(config)
        
        result = optimizer.search("natural", budget=20)
        
        assert isinstance(result, SearchResult)
        assert result.best_policy is not None
        assert result.best_score >= 0.0
        assert result.budget_used > 0
        assert len(result.history) > 0
    
    def test_search_respects_budget(self):
        """Search does not exceed budget."""
        config = OptimizerConfig(
            population_size=10,
            generations=10,
            seed=42,
        )
        optimizer = PolicyOptimizer(config)
        
        budget = 30
        result = optimizer.search("natural", budget=budget)
        
        assert result.budget_used <= budget
    
    def test_search_improves_over_generations(self):
        """Score tends to improve over generations."""
        config = OptimizerConfig(
            population_size=10,
            generations=5,
            seed=42,
        )
        optimizer = PolicyOptimizer(config)
        
        result = optimizer.search("natural", budget=60)
        
        # Check history shows some progression
        if len(result.history) >= 2:
            first_gen = result.history[0]
            last_gen = result.history[-1]
            # Average should improve or stay same
            assert last_gen["best_score"] >= first_gen["best_score"] * 0.9


class TestSearchResult:
    """Test SearchResult container."""
    
    def test_summary(self):
        """Summary returns string."""
        result = SearchResult(
            best_policy=Policy("test", "natural", []),
            best_score=0.85,
            domain="natural",
            budget_used=50,
            search_time=1.5,
        )
        
        summary = result.summary()
        
        assert "0.85" in summary
        assert "50" in summary
    
    def test_to_json(self):
        """Result can be serialized to JSON."""
        result = SearchResult(
            best_policy=Policy("test", "natural", [
                Transform("HorizontalFlip", 0.5),
            ]),
            best_score=0.85,
            domain="natural",
            budget_used=50,
            search_time=1.5,
        )
        
        json_str = result.to_json()
        
        assert "best_score" in json_str
        assert "0.85" in json_str
    
    def test_save(self, tmp_path):
        """Result can be saved to directory."""
        result = SearchResult(
            best_policy=Policy("test_policy", "natural", [
                Transform("HorizontalFlip", 0.5),
            ]),
            best_score=0.85,
            domain="natural",
            budget_used=50,
            search_time=1.5,
        )
        
        output_dir = tmp_path / "search_output"
        result_path = result.save(output_dir)
        
        assert result_path.exists()
        assert (output_dir / "best_policy.yaml").exists()


class TestQuickSearch:
    """Test convenience function."""
    
    def test_quick_search_works(self):
        """Quick search convenience function works."""
        result = quick_search("natural", budget=15, seed=42)
        
        assert isinstance(result, SearchResult)
        assert result.best_policy is not None
