# AGENTS.md - AugmentAI Repository Guide

## What This Project Does

AugmentAI is an LLM-powered CLI tool for domain-safe data preparation. Users describe their dataset and task, and the system generates reproducible augmentation pipelines with hard domain constraints.

**Design Philosophy**: The LLM suggests. Rules decide. Code executes.

## Project Structure

```
augmentai/
├── cli/           # CLI commands (typer + rich)
│   ├── app.py     # Main entry point
│   ├── chat.py    # Interactive chat session
│   └── prepare.py # One-command data preparation
├── core/          # Policy, Transform, Config, Manifest
├── domains/       # Domain rules (medical, ocr, satellite, natural)
├── inspection/    # Dataset auto-detection & analysis
├── splitting/     # Train/val/test splitting strategies
├── export/        # Script & folder generation
├── llm/           # LLM client (OpenAI, Ollama, LM Studio)
├── rules/         # Safety validator & constraint enforcer
└── compilers/     # Export to Albumentations
```

## Key Commands

```bash
# One-command dataset preparation
augmentai prepare ./dataset --domain medical

# Interactive policy design
augmentai chat --domain medical

# List domains and constraints
augmentai domains

# Validate existing policy
augmentai validate policy.yaml --domain medical
```

## Key Concepts

| Component | Purpose |
|-----------|---------|
| `Policy` | Collection of transforms with probabilities |
| `Transform` | Single augmentation with parameters |
| `Domain` | Rules defining allowed/forbidden transforms |
| `SafetyValidator` | Enforces hard constraints (removes forbidden) |
| `ReproducibilityManifest` | Captures seed, versions, hashes for reproducibility |

## Layer Separation

| Layer | Responsibility |
|-------|---------------|
| **LLM** | Suggests transforms, explains reasoning |
| **Rules** | Enforces constraints, clamps parameters |
| **Execution** | Runs pipelines deterministically |

The LLM **cannot** bypass rules or execute code directly.

## Entry Points

- **Prepare**: `augmentai/cli/prepare.py` → `augmentai prepare`
- **Chat**: `augmentai/cli/chat.py` → `augmentai chat`
- **Domains**: `augmentai/domains/{medical,ocr,satellite,natural}.py`

## Common Tasks

### Add a New Domain
1. Create `augmentai/domains/your_domain.py`
2. Subclass `Domain`, implement `_setup()` with constraints
3. Register in `augmentai/domains/__init__.py` `_DOMAINS` dict

### Add a New Transform
1. Add `TransformSpec` to `augmentai/core/schema.py`
2. Add mapping in `augmentai/compilers/albumentations.py`

### Fix LLM Response Parsing
- Parser: `augmentai/llm/parser.py`
- Prompts: `augmentai/llm/prompts.py`

## Testing

```bash
pytest tests/ -v                    # Run all tests
pytest tests/test_domains.py -v     # Domain constraints
pytest tests/test_prepare.py -v     # Prepare command modules
```

## Environment

- Python 3.9+
- API key via `.env` file (`OPENAI_API_KEY`) or use Ollama locally

