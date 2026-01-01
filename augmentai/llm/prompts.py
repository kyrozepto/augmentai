"""
Prompt templates for LLM-based augmentation policy design.

These prompts embed domain knowledge and safety constraints to guide
the LLM in generating valid augmentation recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from augmentai.domains.base import Domain
    from augmentai.core.schema import PolicySchema


SYSTEM_PROMPT = """You are AugmentAI, an expert data augmentation advisor for computer vision tasks.

Your role is to help users design effective, domain-appropriate augmentation policies. You:

1. **Understand the domain**: Medical imaging, OCR, satellite imagery, etc. each have specific constraints.
2. **Recommend safe transforms**: Only suggest augmentations that are valid for the user's domain.
3. **Explain your reasoning**: Tell users WHY certain augmentations are good or bad for their task.
4. **Respect hard constraints**: NEVER suggest transforms that are FORBIDDEN in the domain.
5. **Be conservative**: When in doubt, prefer safer, milder augmentations.

CRITICAL RULES:
- You are an ADVISOR, not the final authority. A safety validator will check your recommendations.
- Always output your transform recommendations in a structured format.
- Include probabilities and parameter ranges for each transform.
- Explain any risks or trade-offs.

{domain_context}
{schema_context}
"""

POLICY_GENERATION_PROMPT = """Based on the user's description, suggest an augmentation policy.

Output your recommendation as a JSON object with this exact structure:
{{
    "reasoning": "Brief explanation of your augmentation strategy",
    "policy_name": "Suggested name for this policy",
    "transforms": [
        {{
            "name": "TransformName",
            "probability": 0.5,
            "parameters": {{"param1": value1}},
            "reasoning": "Why this transform is appropriate"
        }}
    ],
    "warnings": ["Any potential issues to be aware of"],
    "alternatives": ["Other transforms that could be considered"]
}}

Only include transforms that are SAFE for the specified domain. If the user requests something unsafe, explain why you cannot include it.
"""

REFINEMENT_PROMPT = """The user wants to modify the current policy. Current policy:
{current_policy}

Update the policy based on the user's feedback. Output the same JSON structure as before, with your modifications.
"""


@dataclass
class PromptBuilder:
    """
    Builds prompts for the LLM with domain and schema context.
    """
    
    domain: Domain
    schema: PolicySchema
    
    def build_system_prompt(self) -> str:
        """Build the system prompt with domain constraints."""
        domain_context = self.domain.get_context_for_llm()
        
        # Build schema context (available transforms)
        schema_lines = ["## Available Transforms:"]
        for name, spec in self.schema.transforms.items():
            # Skip forbidden transforms
            if name in self.domain.forbidden_transforms:
                continue
            
            param_str = ""
            if spec.parameters:
                params = [f"{p.name}: [{p.min_value}, {p.max_value}]" 
                          for p in spec.parameters.values()]
                param_str = f" (params: {', '.join(params)})"
            
            schema_lines.append(f"- {name}: {spec.description}{param_str}")
        
        schema_context = "\n".join(schema_lines)
        
        return SYSTEM_PROMPT.format(
            domain_context=domain_context,
            schema_context=schema_context,
        )
    
    def build_generation_prompt(self, user_description: str) -> str:
        """Build a prompt for initial policy generation."""
        return f"""User's task description:
{user_description}

{POLICY_GENERATION_PROMPT}"""
    
    def build_refinement_prompt(
        self, 
        user_feedback: str, 
        current_policy_json: str
    ) -> str:
        """Build a prompt for refining an existing policy."""
        return f"""User's feedback:
{user_feedback}

{REFINEMENT_PROMPT.format(current_policy=current_policy_json)}"""
    
    def build_explanation_prompt(self, transform_name: str) -> str:
        """Build a prompt for explaining a specific transform."""
        return f"""Explain the '{transform_name}' augmentation in the context of the {self.domain.name} domain.

Cover:
1. What does this transform do?
2. How does it help training?
3. What are the risks for this domain?
4. What parameter ranges are safe?
5. When should it be used?"""
    
    def build_comparison_prompt(self, policy_a_json: str, policy_b_json: str) -> str:
        """Build a prompt for comparing two policies."""
        return f"""Compare these two augmentation policies for the {self.domain.name} domain:

Policy A:
{policy_a_json}

Policy B:
{policy_b_json}

Analyze:
1. Which policy provides more diversity?
2. Which is safer for this domain?
3. Which would you recommend and why?
4. What improvements could be made to each?"""


# Quick prompts for common interactions
QUICK_PROMPTS = {
    "more_aggressive": "Make the augmentations more aggressive with higher probabilities and stronger parameters.",
    "more_conservative": "Make the augmentations more conservative with lower probabilities and milder parameters.",
    "add_color": "Add color-related augmentations if they are safe for this domain.",
    "add_geometric": "Add geometric augmentations if they are safe for this domain.",
    "add_noise": "Add noise and blur augmentations if they are safe for this domain.",
    "remove_risky": "Remove any augmentations that might be risky or have potential issues.",
    "optimize_speed": "Optimize the policy for training speed (prefer simple transforms).",
    "optimize_diversity": "Maximize augmentation diversity while staying within domain constraints.",
}
