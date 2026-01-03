# AugmentAI Documentation

Welcome to the AugmentAI documentation! AugmentAI is an LLM-powered data augmentation policy designer for computer vision.

## Quick Navigation

- **[Quick Start](quickstart.md)** - Get started in 5 minutes
- **[Domain Guides](domains/)** - Domain-specific augmentation strategies
- **[API Reference](api/)** - Python API documentation
- **[Architecture](architecture/)** - Design decisions and internals

## What is AugmentAI?

AugmentAI automates data augmentation for computer vision with three key principles:

1. **LLM-Powered Design**: Describe your dataset in natural language
2. **Domain-Safe Policies**: Hard constraints prevent invalid augmentations
3. **Full Reproducibility**: Seed locking, manifests, and version tracking

## Installation

```bash
pip install augmentai
# or from source
git clone https://github.com/kyrozepto/aai.git
cd aai && pip install -e .
```

## Core Commands

```bash
# One-command dataset preparation
augmentai prepare ./dataset --domain medical

# Interactive policy design
augmentai chat --domain medical

# List available domains
augmentai domains

# Validate a policy
augmentai validate policy.yaml --domain medical
```

## Design Philosophy

> **The LLM suggests. Rules decide. Code executes.**

AugmentAI separates concerns:
- The LLM provides intelligent suggestions based on your description
- Domain rules enforce safety (you cannot override forbidden transforms)
- Compiled code runs deterministically with locked seeds
