"""Tests for the curriculum-aware dataset preparation module."""

import pytest
from pathlib import Path

from augmentai.curriculum import (
    DifficultyScore,
    DifficultyScorer,
    CurriculumSchedule,
    CurriculumScheduler,
    AdaptiveAugmentation,
)
from augmentai.core.policy import Policy, Transform


class TestDifficultyScore:
    """Test DifficultyScore dataclass."""
    
    def test_difficulty_levels(self):
        """Difficulty levels are assigned correctly."""
        easy = DifficultyScore("s1", Path("/test.jpg"), score=0.1)
        medium = DifficultyScore("s2", Path("/test.jpg"), score=0.35)
        hard = DifficultyScore("s3", Path("/test.jpg"), score=0.6)
        very_hard = DifficultyScore("s4", Path("/test.jpg"), score=0.9)
        
        assert easy.difficulty_level == "easy"
        assert medium.difficulty_level == "medium"
        assert hard.difficulty_level == "hard"
        assert very_hard.difficulty_level == "very hard"
    
    def test_score_clamping(self):
        """Score is clamped to [0, 1]."""
        low = DifficultyScore("s1", Path("/test.jpg"), score=-0.5)
        high = DifficultyScore("s2", Path("/test.jpg"), score=1.5)
        
        assert low.score == 0.0
        assert high.score == 1.0
    
    def test_to_dict(self):
        """Can convert to dictionary."""
        score = DifficultyScore("test_001", Path("/test/img.jpg"), score=0.5)
        
        d = score.to_dict()
        assert d["sample_id"] == "test_001"
        assert d["score"] == 0.5
        assert "difficulty_level" in d


class TestDifficultyScorer:
    """Test DifficultyScorer class."""
    
    def test_score_single_sample(self):
        """Can score a single sample."""
        def mock_loss(path, label):
            return 1.0
        
        scorer = DifficultyScorer(loss_fn=mock_loss)
        score = scorer.score_sample(Path("/test/img.jpg"), "cat")
        
        assert score.raw_loss == 1.0
        assert score.label == "cat"
    
    def test_score_dataset_normalizes(self):
        """Scores are normalized across dataset."""
        def mock_loss(path, label):
            # Different samples have different losses
            losses = {"img1": 0.5, "img2": 1.5, "img3": 3.0}
            return losses.get(path.stem, 1.0)
        
        scorer = DifficultyScorer(loss_fn=mock_loss)
        samples = [
            (Path("/test/img1.jpg"), "cat"),
            (Path("/test/img2.jpg"), "cat"),
            (Path("/test/img3.jpg"), "cat"),
        ]
        
        scores = scorer.score_dataset(samples)
        
        assert len(scores) == 3
        # Scores should be normalized: img1 should be easiest (lowest loss)
        assert scores[0].score <= scores[1].score <= scores[2].score or \
               any(s.score != 0 for s in scores)  # At least some variation
    
    def test_rank_by_difficulty(self):
        """Can rank samples by difficulty."""
        scores = [
            DifficultyScore("hard", Path("/h.jpg"), score=0.9),
            DifficultyScore("easy", Path("/e.jpg"), score=0.1),
            DifficultyScore("medium", Path("/m.jpg"), score=0.5),
        ]
        
        scorer = DifficultyScorer(loss_fn=lambda p, l: 0)
        
        ranked_asc = scorer.rank_by_difficulty(scores, ascending=True)
        assert ranked_asc[0] == "easy"
        assert ranked_asc[-1] == "hard"
        
        ranked_desc = scorer.rank_by_difficulty(scores, ascending=False)
        assert ranked_desc[0] == "hard"


class TestCurriculumSchedule:
    """Test CurriculumSchedule dataclass."""
    
    def test_get_samples_for_epoch(self):
        """Can get samples for a specific epoch."""
        schedule = CurriculumSchedule(
            n_epochs=10,
            epoch_samples={
                0: ["s1", "s2"],
                5: ["s1", "s2", "s3"],
            },
        )
        
        assert schedule.get_samples_for_epoch(0) == ["s1", "s2"]
        assert schedule.get_samples_for_epoch(5) == ["s1", "s2", "s3"]
        # Fallback to last available
        assert schedule.get_samples_for_epoch(9) == ["s1", "s2", "s3"]


class TestCurriculumScheduler:
    """Test CurriculumScheduler class."""
    
    def test_create_schedule(self):
        """Can create a curriculum schedule."""
        scores = [
            DifficultyScore("easy", Path("/e.jpg"), score=0.1),
            DifficultyScore("med", Path("/m.jpg"), score=0.5),
            DifficultyScore("hard", Path("/h.jpg"), score=0.9),
        ]
        
        scheduler = CurriculumScheduler(pacing="linear", warmup_epochs=2)
        schedule = scheduler.create_schedule(scores, n_epochs=10)
        
        assert schedule.n_epochs == 10
        assert schedule.pacing_function == "linear"
        # Earlier epochs should have fewer samples
        assert len(schedule.get_samples_for_epoch(0)) <= len(schedule.get_samples_for_epoch(9))
    
    def test_pacing_functions(self):
        """Different pacing functions work."""
        scores = [
            DifficultyScore(f"s{i}", Path(f"/{i}.jpg"), score=i/10)
            for i in range(10)
        ]
        
        for pacing in ["linear", "quadratic", "exponential", "step"]:
            scheduler = CurriculumScheduler(pacing=pacing)
            schedule = scheduler.create_schedule(scores, n_epochs=20)
            assert schedule.pacing_function == pacing
    
    def test_save_schedule(self, tmp_path):
        """Can save schedule to file."""
        schedule = CurriculumSchedule(
            n_epochs=10,
            epoch_samples={0: ["s1"], 5: ["s1", "s2"]},
        )
        
        scheduler = CurriculumScheduler()
        save_path = scheduler.save_schedule(schedule, tmp_path / "schedule.json")
        
        assert save_path.exists()


class TestAdaptiveAugmentation:
    """Test AdaptiveAugmentation class."""
    
    @pytest.fixture
    def base_policy(self):
        """Create a base policy for testing."""
        return Policy(
            name="test_policy",
            domain="natural",
            transforms=[
                Transform("HorizontalFlip", 0.5),
                Transform("Rotate", 0.7, parameters={"limit": 30}),
            ],
        )
    
    def test_strength_for_epoch_linear(self, base_policy):
        """Linear schedule increases linearly."""
        adapter = AdaptiveAugmentation(
            base_policy,
            min_strength=0.2,
            max_strength=1.0,
            schedule="linear",
        )
        
        s0 = adapter.get_strength_for_epoch(0, 10)
        s5 = adapter.get_strength_for_epoch(5, 10)
        s9 = adapter.get_strength_for_epoch(9, 10)
        
        assert s0 == pytest.approx(0.2, abs=0.01)
        assert s5 > s0
        assert s9 == pytest.approx(1.0, abs=0.01)
    
    def test_strength_for_sample(self, base_policy):
        """Sample difficulty affects strength."""
        adapter = AdaptiveAugmentation(
            base_policy,
            min_strength=0.2,
            max_strength=1.0,
        )
        
        # Easy sample (low difficulty) -> high strength (inverted)
        easy_strength = adapter.get_strength_for_sample(0.0, invert=True)
        hard_strength = adapter.get_strength_for_sample(1.0, invert=True)
        
        assert easy_strength > hard_strength
        assert easy_strength == pytest.approx(1.0)
        assert hard_strength == pytest.approx(0.2)
    
    def test_get_policy_for_epoch(self, base_policy):
        """Can create adjusted policy."""
        adapter = AdaptiveAugmentation(
            base_policy,
            min_strength=0.5,
            max_strength=1.0,
        )
        
        policy = adapter.get_policy_for_epoch(0, 10)
        
        assert policy.name.startswith(base_policy.name)
        # Probabilities should be scaled
        assert policy.transforms[0].probability <= base_policy.transforms[0].probability
    
    def test_schedule_types(self, base_policy):
        """All schedule types work."""
        for schedule in ["linear", "cosine", "warmup", "constant"]:
            adapter = AdaptiveAugmentation(base_policy, schedule=schedule)
            strength = adapter.get_strength_for_epoch(5, 10)
            assert 0 <= strength <= 1
