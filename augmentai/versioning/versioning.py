"""
Policy versioning and diff tracking.

Track policy changes across experiments and compute diffs between versions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from augmentai.core.policy import Policy, Transform


@dataclass
class PolicyDiff:
    """Difference between two policy versions."""
    
    added_transforms: list[Transform] = field(default_factory=list)
    removed_transforms: list[Transform] = field(default_factory=list)
    modified_transforms: list[tuple[Transform, Transform]] = field(default_factory=list)
    parameter_changes: dict[str, dict[str, Any]] = field(default_factory=dict)
    probability_changes: dict[str, tuple[float, float]] = field(default_factory=dict)
    
    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return bool(
            self.added_transforms or
            self.removed_transforms or
            self.modified_transforms or
            self.parameter_changes or
            self.probability_changes
        )
    
    @property
    def summary(self) -> str:
        """Get one-line summary."""
        parts = []
        if self.added_transforms:
            parts.append(f"+{len(self.added_transforms)} added")
        if self.removed_transforms:
            parts.append(f"-{len(self.removed_transforms)} removed")
        if self.modified_transforms:
            parts.append(f"~{len(self.modified_transforms)} modified")
        return ", ".join(parts) if parts else "No changes"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "added_transforms": [t.to_dict() for t in self.added_transforms],
            "removed_transforms": [t.to_dict() for t in self.removed_transforms],
            "modified_transforms": [
                {"old": old.to_dict(), "new": new.to_dict()}
                for old, new in self.modified_transforms
            ],
            "parameter_changes": self.parameter_changes,
            "probability_changes": {
                k: {"old": v[0], "new": v[1]} 
                for k, v in self.probability_changes.items()
            },
        }
    
    def format_diff(self) -> str:
        """Format diff as human-readable text."""
        lines = []
        
        for t in self.removed_transforms:
            lines.append(f"- {t.name} (p={t.probability})")
        
        for t in self.added_transforms:
            lines.append(f"+ {t.name} (p={t.probability})")
        
        for old, new in self.modified_transforms:
            lines.append(f"~ {old.name}:")
            if old.probability != new.probability:
                lines.append(f"    probability: {old.probability} → {new.probability}")
            for key in set(list(old.parameters.keys()) + list(new.parameters.keys())):
                old_val = old.parameters.get(key)
                new_val = new.parameters.get(key)
                if old_val != new_val:
                    lines.append(f"    {key}: {old_val} → {new_val}")
        
        return "\n".join(lines) if lines else "No changes"


@dataclass
class PolicyVersion:
    """A versioned snapshot of a policy."""
    
    policy: Policy
    version: str
    hash: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    parent_version: str | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "hash": self.hash,
            "timestamp": self.timestamp,
            "parent_version": self.parent_version,
            "message": self.message,
            "policy": self.policy.to_dict(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyVersion:
        return cls(
            policy=Policy.from_dict(data["policy"]),
            version=data["version"],
            hash=data["hash"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            parent_version=data.get("parent_version"),
            message=data.get("message", ""),
            metadata=data.get("metadata", {}),
        )


class PolicyVersionControl:
    """
    Version control system for augmentation policies.
    
    Tracks policy versions, computes diffs, and provides history.
    Can export to DVC-compatible format for integration with ML pipelines.
    """
    
    def __init__(self, storage_dir: Path) -> None:
        """
        Initialize version control.
        
        Args:
            storage_dir: Directory to store policy versions
        """
        self.storage_dir = Path(storage_dir)
        self.versions_dir = self.storage_dir / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        
        self._index_file = self.storage_dir / "index.json"
        self._index: dict[str, list[str]] = self._load_index()
    
    def _load_index(self) -> dict[str, list[str]]:
        """Load or create version index."""
        if self._index_file.exists():
            return json.loads(self._index_file.read_text())
        return {}
    
    def _save_index(self) -> None:
        """Save version index."""
        self._index_file.write_text(json.dumps(self._index, indent=2))
    
    def _compute_hash(self, policy: Policy) -> str:
        """Compute unique hash for a policy."""
        hasher = hashlib.sha256()
        hasher.update(json.dumps(policy.to_dict(), sort_keys=True).encode())
        return hasher.hexdigest()[:12]
    
    def _generate_version(self, policy_name: str) -> str:
        """Generate next version number for a policy."""
        versions = self._index.get(policy_name, [])
        return f"v{len(versions) + 1}"
    
    def commit(self, policy: Policy, message: str = "") -> PolicyVersion:
        """
        Commit a new version of a policy.
        
        Args:
            policy: The policy to commit
            message: Optional commit message
            
        Returns:
            The created PolicyVersion
        """
        policy_hash = self._compute_hash(policy)
        version_str = self._generate_version(policy.name)
        
        # Get parent version
        versions = self._index.get(policy.name, [])
        parent = versions[-1] if versions else None
        
        version = PolicyVersion(
            policy=policy,
            version=version_str,
            hash=policy_hash,
            parent_version=parent,
            message=message,
        )
        
        # Save version file
        version_file = self.versions_dir / f"{policy.name}_{version_str}.json"
        version_file.write_text(json.dumps(version.to_dict(), indent=2))
        
        # Update index
        if policy.name not in self._index:
            self._index[policy.name] = []
        self._index[policy.name].append(version_str)
        self._save_index()
        
        return version
    
    def get_version(self, policy_name: str, version: str) -> PolicyVersion | None:
        """Get a specific version of a policy."""
        version_file = self.versions_dir / f"{policy_name}_{version}.json"
        if not version_file.exists():
            return None
        
        data = json.loads(version_file.read_text())
        return PolicyVersion.from_dict(data)
    
    def get_latest(self, policy_name: str) -> PolicyVersion | None:
        """Get the latest version of a policy."""
        versions = self._index.get(policy_name, [])
        if not versions:
            return None
        return self.get_version(policy_name, versions[-1])
    
    def history(self, policy_name: str) -> list[PolicyVersion]:
        """Get version history for a policy."""
        versions = self._index.get(policy_name, [])
        return [
            self.get_version(policy_name, v)
            for v in versions
            if self.get_version(policy_name, v) is not None
        ]
    
    def diff(
        self,
        policy1: Policy | str,
        policy2: Policy | str,
        policy_name: str | None = None,
    ) -> PolicyDiff:
        """
        Compute diff between two policies or versions.
        
        Args:
            policy1: First policy or version string (e.g., "v1")
            policy2: Second policy or version string
            policy_name: Required if using version strings
            
        Returns:
            PolicyDiff with changes
        """
        # Resolve version strings to policies
        if isinstance(policy1, str):
            if policy_name is None:
                raise ValueError("policy_name required when using version strings")
            version = self.get_version(policy_name, policy1)
            if version is None:
                raise ValueError(f"Version {policy1} not found")
            policy1 = version.policy
        
        if isinstance(policy2, str):
            if policy_name is None:
                raise ValueError("policy_name required when using version strings")
            version = self.get_version(policy_name, policy2)
            if version is None:
                raise ValueError(f"Version {policy2} not found")
            policy2 = version.policy
        
        return self._compute_diff(policy1, policy2)
    
    def _compute_diff(self, old: Policy, new: Policy) -> PolicyDiff:
        """Compute diff between two policies."""
        old_transforms = {t.name: t for t in old.transforms}
        new_transforms = {t.name: t for t in new.transforms}
        
        old_names = set(old_transforms.keys())
        new_names = set(new_transforms.keys())
        
        added = [new_transforms[n] for n in new_names - old_names]
        removed = [old_transforms[n] for n in old_names - new_names]
        
        modified = []
        param_changes = {}
        prob_changes = {}
        
        for name in old_names & new_names:
            old_t = old_transforms[name]
            new_t = new_transforms[name]
            
            is_modified = False
            
            if old_t.probability != new_t.probability:
                prob_changes[name] = (old_t.probability, new_t.probability)
                is_modified = True
            
            if old_t.parameters != new_t.parameters:
                param_changes[name] = {
                    "old": old_t.parameters,
                    "new": new_t.parameters,
                }
                is_modified = True
            
            if is_modified:
                modified.append((old_t, new_t))
        
        return PolicyDiff(
            added_transforms=added,
            removed_transforms=removed,
            modified_transforms=modified,
            parameter_changes=param_changes,
            probability_changes=prob_changes,
        )
    
    def export_to_dvc(self, version: PolicyVersion, output_dir: Path) -> Path:
        """
        Export policy version in DVC-compatible format.
        
        Args:
            version: The version to export
            output_dir: Output directory
            
        Returns:
            Path to the exported .yaml file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export policy as YAML
        policy_file = output_dir / f"{version.policy.name}.yaml"
        policy_file.write_text(version.policy.to_yaml())
        
        # Create DVC params file
        params_file = output_dir / "params.yaml"
        params = {
            "policy": {
                "name": version.policy.name,
                "version": version.version,
                "hash": version.hash,
                "domain": version.policy.domain,
                "transforms": [t.name for t in version.policy.transforms],
            }
        }
        import yaml
        params_file.write_text(yaml.dump(params, default_flow_style=False))
        
        return policy_file
    
    def list_policies(self) -> list[str]:
        """List all tracked policies."""
        return list(self._index.keys())
