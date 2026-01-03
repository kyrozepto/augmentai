"""
Evolutionary optimizer for AutoSearch.

Uses genetic algorithm-style optimization to find optimal augmentation policies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from augmentai.core.policy import Policy
from augmentai.search.sampler import PolicySampler
from augmentai.search.evaluator import PolicyEvaluator, EvaluationResult
from augmentai.search.result import SearchResult, GenerationStats
from augmentai.utils.progress import (
    ProgressTracker,
    print_info,
    print_success,
    print_debug,
    is_verbose,
)


@dataclass
class OptimizerConfig:
    """Configuration for the policy optimizer."""
    
    # Population size per generation
    population_size: int = 20
    
    # Number of generations to run
    generations: int = 10
    
    # Fraction of population to keep (elitism)
    elite_fraction: float = 0.2
    
    # Mutation probability
    mutation_rate: float = 0.5
    
    # Mutation strength (how much to perturb)
    mutation_strength: float = 0.3
    
    # Crossover probability
    crossover_rate: float = 0.3
    
    # Early stopping if no improvement for N generations
    patience: int = 3
    
    # Random seed
    seed: int = 42


class PolicyOptimizer:
    """
    Optimize augmentation policies using evolutionary search.
    
    Algorithm:
    1. Initialize population via sampler
    2. Evaluate fitness of each policy
    3. Select top performers (elite)
    4. Generate offspring via mutation and crossover
    5. Repeat until budget exhausted or converged
    """
    
    def __init__(
        self,
        config: OptimizerConfig | None = None,
        progress_callback: Callable[[int, float], None] | None = None,
    ) -> None:
        """
        Initialize the optimizer.
        
        Args:
            config: Optimizer configuration
            progress_callback: Optional callback(generation, best_score)
        """
        self.config = config or OptimizerConfig()
        self.progress_callback = progress_callback
        
        # Initialize sampler and evaluator
        self.sampler = PolicySampler(seed=self.config.seed)
        self.evaluator: PolicyEvaluator | None = None
    
    def search(
        self,
        domain: str,
        budget: int = 100,
        context: str | None = None,
    ) -> SearchResult:
        """
        Run evolutionary search to find optimal policy.
        
        Args:
            domain: Domain for policies (medical, ocr, etc.)
            budget: Maximum number of policy evaluations
            context: Optional context about dataset/task
            
        Returns:
            SearchResult with best policy and history
        """
        start_time = time.time()
        
        # Initialize evaluator for this domain
        self.evaluator = PolicyEvaluator(domain=domain)
        
        # Calculate generations based on budget
        pop_size = min(self.config.population_size, budget // 2)
        max_gens = budget // pop_size
        
        print_debug(f"Search config: pop={pop_size}, gens={max_gens}, budget={budget}")
        
        # Initialize population
        population = self.sampler.sample(domain, n=pop_size, context=context)
        
        # Track all evaluated policies
        all_candidates: list[tuple[Policy, float]] = []
        history: list[dict[str, Any]] = []
        
        # Track for early stopping
        best_score_ever = -float("inf")
        generations_without_improvement = 0
        
        # Evolutionary loop
        evaluations_used = 0
        
        with ProgressTracker("Searching", total_steps=max_gens) as tracker:
            for gen in range(max_gens):
                if evaluations_used >= budget:
                    break
                
                tracker.update(f"Generation {gen + 1}/{max_gens}")
                
                # Evaluate population
                results = self.evaluator.evaluate_batch(population)
                evaluations_used += len(population)
                
                # Record candidates
                for policy, result in zip(population, results):
                    all_candidates.append((policy, result.score))
                
                # Sort by score (descending)
                scored_pop = sorted(
                    zip(population, results),
                    key=lambda x: x[1].score,
                    reverse=True
                )
                
                # Get statistics
                scores = [r.score for _, r in scored_pop]
                gen_best = scores[0]
                gen_avg = sum(scores) / len(scores)
                gen_worst = scores[-1]
                
                stats = GenerationStats(
                    generation=gen,
                    best_score=gen_best,
                    avg_score=gen_avg,
                    worst_score=gen_worst,
                    population_size=len(population),
                )
                history.append(stats.to_dict())
                
                print_debug(
                    f"Gen {gen}: best={gen_best:.4f}, avg={gen_avg:.4f}"
                )
                
                # Check for improvement
                if gen_best > best_score_ever:
                    best_score_ever = gen_best
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
                
                # Early stopping
                if generations_without_improvement >= self.config.patience:
                    print_debug(f"Early stopping at gen {gen} (no improvement)")
                    break
                
                # Progress callback
                if self.progress_callback:
                    self.progress_callback(gen, gen_best)
                
                # Select elite
                n_elite = max(2, int(len(scored_pop) * self.config.elite_fraction))
                elite = [p for p, _ in scored_pop[:n_elite]]
                
                # Generate next population
                next_population = list(elite)  # Keep elite
                
                # Fill rest with mutations and crossovers
                while len(next_population) < pop_size:
                    if evaluations_used >= budget:
                        break
                    
                    # Decide: mutate or crossover
                    if len(elite) >= 2 and self.sampler.rng.random() < self.config.crossover_rate:
                        # Crossover
                        p1, p2 = self.sampler.rng.sample(elite, 2)
                        child = self.sampler.crossover(p1, p2)
                        stats.crossovers += 1
                    else:
                        # Mutate
                        parent = self.sampler.rng.choice(elite)
                        child = self.sampler.mutate(
                            parent, 
                            strength=self.config.mutation_strength
                        )
                        stats.mutations += 1
                    
                    next_population.append(child)
                
                population = next_population
                tracker.advance()
        
        # Find overall best
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        best_policy, best_score = all_candidates[0]
        
        search_time = time.time() - start_time
        
        print_success(
            f"Search complete: score={best_score:.4f}, "
            f"evals={evaluations_used}, time={search_time:.1f}s"
        )
        
        return SearchResult(
            best_policy=best_policy,
            best_score=best_score,
            domain=domain,
            budget_used=evaluations_used,
            search_time=search_time,
            history=history,
            all_candidates=all_candidates[:20],  # Keep top 20
            seed=self.config.seed,
        )


def quick_search(
    domain: str,
    budget: int = 50,
    seed: int = 42,
) -> SearchResult:
    """
    Quick search with default settings.
    
    Convenience function for simple search invocation.
    
    Args:
        domain: Domain name
        budget: Evaluation budget
        seed: Random seed
        
    Returns:
        SearchResult with best policy
    """
    config = OptimizerConfig(
        population_size=min(10, budget // 3),
        generations=max(3, budget // 10),
        seed=seed,
    )
    optimizer = PolicyOptimizer(config)
    return optimizer.search(domain, budget)
