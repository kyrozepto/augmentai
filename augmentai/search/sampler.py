"""
LLM-based policy candidate generation for AutoSearch.

Uses the LLM to intelligently generate augmentation policy candidates
based on domain knowledge and search context.
"""

from __future__ import annotations

import random
from typing import Any

from augmentai.core.policy import Policy, Transform
from augmentai.core.schema import DEFAULT_SCHEMA
from augmentai.domains import get_domain
from augmentai.rules.enforcement import RuleEnforcer


class PolicySampler:
    """
    Generate candidate augmentation policies using LLM + domain rules.
    
    Provides two generation modes:
    1. sample(): Generate fresh policies via LLM
    2. mutate(): Create variations of existing policies
    """
    
    # All available transforms for sampling
    TRANSFORM_POOL = [
        "HorizontalFlip",
        "VerticalFlip",
        "Rotate",
        "RandomBrightnessContrast",
        "GaussNoise",
        "GaussianBlur",
        "CLAHE",
        "ShiftScaleRotate",
        "CoarseDropout",
        "RandomCrop",
        "RandomScale",
        "Perspective",
        "OpticalDistortion",
        "GridDistortion",
        "ElasticTransform",
        "ColorJitter",
        "HueSaturationValue",
        "ChannelShuffle",
        "MotionBlur",
        "MedianBlur",
        "Sharpen",
        "Emboss",
        "RandomGamma",
        "RandomToneCurve",
        "Solarize",
        "Posterize",
        "Equalize",
        "Normalize",
    ]
    
    def __init__(
        self,
        seed: int = 42,
        use_llm: bool = False,
        llm_client: Any = None,
    ) -> None:
        """
        Initialize the policy sampler.
        
        Args:
            seed: Random seed for reproducibility
            use_llm: Whether to use LLM for intelligent sampling
            llm_client: Optional LLM client for intelligent sampling
        """
        self.seed = seed
        self.use_llm = use_llm
        self.llm_client = llm_client
        self.rng = random.Random(seed)
        self.schema = DEFAULT_SCHEMA
    
    def sample(
        self,
        domain: str,
        n: int = 10,
        context: str | None = None,
    ) -> list[Policy]:
        """
        Generate n candidate policies for the domain.
        
        Args:
            domain: Domain name (medical, ocr, satellite, natural)
            n: Number of candidates to generate
            context: Optional context about the dataset/task
            
        Returns:
            List of valid candidate policies
        """
        domain_obj = get_domain(domain)
        enforcer = RuleEnforcer(domain_obj)
        
        candidates = []
        for i in range(n):
            # Generate random policy
            policy = self._generate_random_policy(domain, i)
            
            # Enforce domain rules
            result = enforcer.enforce_policy(policy)
            if result.success and result.policy:
                candidates.append(result.policy)
            else:
                # Policy was completely invalid, try again with safer defaults
                safe_policy = self._generate_safe_policy(domain, i)
                result = enforcer.enforce_policy(safe_policy)
                if result.success and result.policy:
                    candidates.append(result.policy)
        
        return candidates
    
    def _generate_random_policy(self, domain: str, idx: int) -> Policy:
        """Generate a random policy with varied transforms."""
        domain_obj = get_domain(domain)
        
        # Get valid transforms (not forbidden)
        valid_transforms = [
            t for t in self.TRANSFORM_POOL 
            if t not in domain_obj.forbidden_transforms
        ]
        
        # Random number of transforms (3-8)
        n_transforms = self.rng.randint(3, 8)
        selected = self.rng.sample(
            valid_transforms, 
            min(n_transforms, len(valid_transforms))
        )
        
        transforms = []
        for name in selected:
            # Random probability
            prob = round(self.rng.uniform(0.2, 0.8), 2)
            
            # Get default parameters from schema
            params = self._get_random_params(name)
            
            transforms.append(Transform(
                name=name,
                probability=prob,
                parameters=params,
            ))
        
        return Policy(
            name=f"{domain}_search_candidate_{idx}",
            domain=domain,
            transforms=transforms,
        )
    
    def _generate_safe_policy(self, domain: str, idx: int) -> Policy:
        """Generate a safe policy using only recommended transforms."""
        domain_obj = get_domain(domain)
        
        # Use recommended transforms only
        recommended = list(domain_obj.recommended_transforms)
        if not recommended:
            recommended = ["HorizontalFlip", "Rotate", "RandomBrightnessContrast"]
        
        n_transforms = min(self.rng.randint(2, 5), len(recommended))
        selected = self.rng.sample(recommended, n_transforms)
        
        transforms = []
        for name in selected:
            prob = round(self.rng.uniform(0.3, 0.6), 2)
            params = self._get_random_params(name)
            transforms.append(Transform(name=name, probability=prob, parameters=params))
        
        return Policy(
            name=f"{domain}_safe_candidate_{idx}",
            domain=domain,
            transforms=transforms,
        )
    
    def _get_random_params(self, transform_name: str) -> dict[str, Any]:
        """Get randomized parameters for a transform within valid ranges."""
        params: dict[str, Any] = {}
        
        # Common parameter randomization
        if transform_name == "Rotate":
            params["limit"] = self.rng.choice([10, 15, 20, 30, 45])
        elif transform_name == "ShiftScaleRotate":
            params["shift_limit"] = round(self.rng.uniform(0.05, 0.2), 2)
            params["scale_limit"] = round(self.rng.uniform(0.05, 0.2), 2)
            params["rotate_limit"] = self.rng.choice([15, 30, 45])
        elif transform_name == "RandomBrightnessContrast":
            params["brightness_limit"] = round(self.rng.uniform(0.1, 0.3), 2)
            params["contrast_limit"] = round(self.rng.uniform(0.1, 0.3), 2)
        elif transform_name == "GaussNoise":
            var_min = self.rng.randint(5, 20)
            var_max = var_min + self.rng.randint(10, 40)
            params["var_limit"] = (var_min, var_max)
        elif transform_name == "GaussianBlur":
            params["blur_limit"] = self.rng.choice([3, 5, 7])
        elif transform_name == "CLAHE":
            params["clip_limit"] = round(self.rng.uniform(1.0, 4.0), 1)
        elif transform_name == "CoarseDropout":
            params["max_holes"] = self.rng.randint(4, 12)
            params["max_height"] = self.rng.randint(8, 32)
            params["max_width"] = self.rng.randint(8, 32)
        
        return params
    
    def mutate(
        self,
        policy: Policy,
        strength: float = 0.3,
    ) -> Policy:
        """
        Create a mutation of an existing policy.
        
        Mutations include:
        - Add/remove transforms
        - Adjust probabilities
        - Modify parameters
        
        Args:
            policy: Policy to mutate
            strength: Mutation strength (0.0 to 1.0)
            
        Returns:
            New mutated policy
        """
        domain_obj = get_domain(policy.domain)
        
        # Clone transforms
        new_transforms = [
            Transform(
                name=t.name,
                probability=t.probability,
                parameters=dict(t.parameters) if t.parameters else {},
            )
            for t in policy.transforms
        ]
        
        # Decide what mutations to apply
        mutations_applied = []
        
        # 1. Probability mutations
        for t in new_transforms:
            if self.rng.random() < strength:
                delta = self.rng.uniform(-0.2, 0.2)
                t.probability = max(0.1, min(1.0, t.probability + delta))
                mutations_applied.append(f"prob({t.name})")
        
        # 2. Add transform
        if self.rng.random() < strength and len(new_transforms) < 10:
            valid_transforms = [
                t for t in self.TRANSFORM_POOL 
                if t not in domain_obj.forbidden_transforms
                and t not in [tr.name for tr in new_transforms]
            ]
            if valid_transforms:
                new_name = self.rng.choice(valid_transforms)
                new_t = Transform(
                    name=new_name,
                    probability=round(self.rng.uniform(0.3, 0.6), 2),
                    parameters=self._get_random_params(new_name),
                )
                new_transforms.append(new_t)
                mutations_applied.append(f"add({new_name})")
        
        # 3. Remove transform
        if self.rng.random() < strength * 0.5 and len(new_transforms) > 2:
            idx = self.rng.randrange(len(new_transforms))
            removed = new_transforms.pop(idx)
            mutations_applied.append(f"remove({removed.name})")
        
        return Policy(
            name=f"{policy.name}_mutated",
            domain=policy.domain,
            transforms=new_transforms,
        )
    
    def crossover(
        self,
        parent1: Policy,
        parent2: Policy,
    ) -> Policy:
        """
        Create offspring by combining two parent policies.
        
        Takes transforms from both parents based on random selection.
        
        Args:
            parent1: First parent policy
            parent2: Second parent policy
            
        Returns:
            New child policy
        """
        domain = parent1.domain
        domain_obj = get_domain(domain)
        
        # Collect all transforms from both parents
        all_transforms = {}
        for t in parent1.transforms:
            all_transforms[t.name] = t
        for t in parent2.transforms:
            if t.name not in all_transforms or self.rng.random() > 0.5:
                all_transforms[t.name] = t
        
        # Select subset
        transform_list = list(all_transforms.values())
        n_select = self.rng.randint(
            min(3, len(transform_list)),
            min(8, len(transform_list))
        )
        selected = self.rng.sample(transform_list, n_select)
        
        # Create child
        child_transforms = [
            Transform(
                name=t.name,
                probability=t.probability,
                parameters=dict(t.parameters) if t.parameters else {},
            )
            for t in selected
        ]
        
        return Policy(
            name=f"{domain}_crossover_child",
            domain=domain,
            transforms=child_transforms,
        )
