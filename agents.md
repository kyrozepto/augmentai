# AGENTS.md - AugmentAI Repository Guide

## What This Project Does

AugmentAI is an LLM-powered CLI tool that helps users design image augmentation policies through natural conversation. Users describe their dataset (medical, OCR, satellite, etc.), and the system generates safe, reproducible augmentation pipelines.

## Project Structure

```
augmentai/
├── cli/           # Chat interface & commands (typer + rich)
├── core/          # Policy, Transform, Config dataclasses
├── domains/       # Domain rules (medical, ocr, satellite, natural)
├── llm/           # LLM client (OpenAI, Ollama, LM Studio)
├── rules/         # Safety validator & constraint enforcer
└── compilers/     # Export to Albumentations (Kornia/torchvision planned)
```

## Key Concepts

| Component | Purpose |
|-----------|---------|
| `Policy` | Collection of transforms with probabilities |
| `Transform` | Single augmentation with parameters |
| `Domain` | Rules defining allowed/forbidden transforms |
| `SafetyValidator` | Enforces hard constraints (removes forbidden transforms) |

## Entry Points

- **CLI**: `augmentai/cli/app.py` → `augmentai chat`
- **Chat Session**: `augmentai/cli/chat.py`
- **Domains**: `augmentai/domains/{medical,ocr,satellite,natural}.py`

## Common Tasks

### Add a New Domain
1. Create `augmentai/domains/your_domain.py`
2. Subclass `Domain`, implement `_setup()` with constraints
3. Register in `augmentai/domains/__init__.py`

### Add a New Transform
1. Add `TransformSpec` to `augmentai/core/schema.py`
2. Add mapping in `augmentai/compilers/albumentations.py`

### Fix LLM Response Parsing
- Parser: `augmentai/llm/parser.py`
- Prompts: `augmentai/llm/prompts.py`

## Testing

```bash
pytest tests/ -v                    # Run all tests
pytest tests/test_domains.py -v     # Domain constraints only
```

## Environment

- Python 3.9+
- API key via `.env` file (`OPENAI_API_KEY`) or use Ollama locally
