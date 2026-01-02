"""Tests for the policy versioning module."""

import pytest

from augmentai.core.policy import Policy, Transform
from augmentai.versioning import PolicyVersionControl, PolicyVersion, PolicyDiff


class TestPolicyDiff:
    """Test PolicyDiff dataclass."""
    
    def test_has_changes_false(self):
        """Empty diff has no changes."""
        diff = PolicyDiff()
        assert diff.has_changes is False
    
    def test_has_changes_added(self):
        """Diff with added transforms has changes."""
        diff = PolicyDiff(added_transforms=[Transform("New", 0.5)])
        assert diff.has_changes is True
    
    def test_summary(self):
        """Summary shows change counts."""
        diff = PolicyDiff(
            added_transforms=[Transform("A", 0.5)],
            removed_transforms=[Transform("B", 0.5), Transform("C", 0.5)],
        )
        
        summary = diff.summary
        assert "+1 added" in summary
        assert "-2 removed" in summary
    
    def test_format_diff(self):
        """Can format diff as text."""
        diff = PolicyDiff(
            added_transforms=[Transform("New", 0.5)],
            removed_transforms=[Transform("Old", 0.3)],
        )
        
        formatted = diff.format_diff()
        assert "+ New" in formatted
        assert "- Old" in formatted
    
    def test_to_dict(self):
        """Can convert to dictionary."""
        diff = PolicyDiff(
            added_transforms=[Transform("New", 0.5)],
        )
        
        d = diff.to_dict()
        assert len(d["added_transforms"]) == 1
        assert d["added_transforms"][0]["name"] == "New"


class TestPolicyVersion:
    """Test PolicyVersion dataclass."""
    
    def test_to_dict(self):
        """Can convert to dictionary."""
        policy = Policy(name="test", domain="natural", transforms=[])
        version = PolicyVersion(
            policy=policy,
            version="v1",
            hash="abc123",
            message="Initial commit",
        )
        
        d = version.to_dict()
        assert d["version"] == "v1"
        assert d["hash"] == "abc123"
        assert d["message"] == "Initial commit"
    
    def test_from_dict(self):
        """Can create from dictionary."""
        data = {
            "version": "v2",
            "hash": "def456",
            "timestamp": "2024-01-01T00:00:00",
            "parent_version": "v1",
            "message": "Update",
            "policy": {
                "name": "test",
                "domain": "natural",
                "transforms": [],
            },
            "metadata": {},
        }
        
        version = PolicyVersion.from_dict(data)
        assert version.version == "v2"
        assert version.parent_version == "v1"
        assert version.policy.name == "test"


class TestPolicyVersionControl:
    """Test PolicyVersionControl class."""
    
    def test_commit_creates_version(self, tmp_path):
        """Commit creates a new version."""
        vc = PolicyVersionControl(tmp_path)
        
        policy = Policy(
            name="test_policy",
            domain="natural",
            transforms=[Transform("HorizontalFlip", 0.5)],
        )
        
        version = vc.commit(policy, "Initial commit")
        
        assert version.version == "v1"
        assert version.message == "Initial commit"
        assert version.parent_version is None
    
    def test_commit_increments_version(self, tmp_path):
        """Each commit increments version number."""
        vc = PolicyVersionControl(tmp_path)
        
        policy = Policy(name="test", domain="natural", transforms=[])
        
        v1 = vc.commit(policy, "v1")
        v2 = vc.commit(policy, "v2")
        v3 = vc.commit(policy, "v3")
        
        assert v1.version == "v1"
        assert v2.version == "v2"
        assert v3.version == "v3"
        assert v2.parent_version == "v1"
        assert v3.parent_version == "v2"
    
    def test_get_version(self, tmp_path):
        """Can retrieve a specific version."""
        vc = PolicyVersionControl(tmp_path)
        
        policy = Policy(
            name="test",
            domain="natural",
            transforms=[Transform("Rotate", 0.5)],
        )
        
        vc.commit(policy, "v1")
        
        retrieved = vc.get_version("test", "v1")
        assert retrieved is not None
        assert retrieved.policy.name == "test"
        assert len(retrieved.policy.transforms) == 1
    
    def test_get_latest(self, tmp_path):
        """Get latest returns most recent version."""
        vc = PolicyVersionControl(tmp_path)
        
        policy = Policy(name="test", domain="natural", transforms=[])
        
        vc.commit(policy, "First")
        policy.add_transform(Transform("Flip", 0.5))
        vc.commit(policy, "Second")
        
        latest = vc.get_latest("test")
        assert latest is not None
        assert latest.version == "v2"
        assert len(latest.policy.transforms) == 1
    
    def test_history(self, tmp_path):
        """Can get version history."""
        vc = PolicyVersionControl(tmp_path)
        
        policy = Policy(name="test", domain="natural", transforms=[])
        
        vc.commit(policy, "v1")
        vc.commit(policy, "v2")
        vc.commit(policy, "v3")
        
        history = vc.history("test")
        assert len(history) == 3
        assert history[0].version == "v1"
        assert history[2].version == "v3"
    
    def test_diff_policies(self, tmp_path):
        """Can compute diff between policies."""
        vc = PolicyVersionControl(tmp_path)
        
        old_policy = Policy(
            name="test",
            domain="natural",
            transforms=[
                Transform("HorizontalFlip", 0.5),
                Transform("Rotate", 0.5, parameters={"limit": 15}),
            ],
        )
        
        new_policy = Policy(
            name="test",
            domain="natural",
            transforms=[
                Transform("HorizontalFlip", 0.7),  # Changed probability
                Transform("GaussNoise", 0.3),      # New
            ],
        )
        
        diff = vc.diff(old_policy, new_policy)
        
        assert len(diff.added_transforms) == 1
        assert diff.added_transforms[0].name == "GaussNoise"
        
        assert len(diff.removed_transforms) == 1
        assert diff.removed_transforms[0].name == "Rotate"
        
        assert len(diff.modified_transforms) == 1
        assert diff.modified_transforms[0][0].name == "HorizontalFlip"
        assert "HorizontalFlip" in diff.probability_changes
    
    def test_diff_versions(self, tmp_path):
        """Can diff using version strings."""
        vc = PolicyVersionControl(tmp_path)
        
        policy = Policy(
            name="test",
            domain="natural",
            transforms=[Transform("Flip", 0.5)],
        )
        vc.commit(policy, "v1")
        
        policy.transforms[0] = Transform("Flip", 0.8)  # Change probability
        vc.commit(policy, "v2")
        
        diff = vc.diff("v1", "v2", policy_name="test")
        assert "Flip" in diff.probability_changes
    
    def test_export_to_dvc(self, tmp_path):
        """Can export to DVC format."""
        vc = PolicyVersionControl(tmp_path)
        
        policy = Policy(
            name="my_policy",
            domain="medical",
            transforms=[Transform("Rotate", 0.5)],
        )
        version = vc.commit(policy, "Export test")
        
        output_dir = tmp_path / "dvc_export"
        policy_file = vc.export_to_dvc(version, output_dir)
        
        assert policy_file.exists()
        assert (output_dir / "params.yaml").exists()
    
    def test_list_policies(self, tmp_path):
        """Can list tracked policies."""
        vc = PolicyVersionControl(tmp_path)
        
        p1 = Policy(name="policy_a", domain="natural", transforms=[])
        p2 = Policy(name="policy_b", domain="medical", transforms=[])
        
        vc.commit(p1, "a")
        vc.commit(p2, "b")
        
        policies = vc.list_policies()
        assert "policy_a" in policies
        assert "policy_b" in policies
