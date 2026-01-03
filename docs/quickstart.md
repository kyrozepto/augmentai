# Quick Start: Your First 5 Minutes with AugmentAI

Get from raw dataset to training-ready augmentations in under 5 minutes.

## Prerequisites

- Python 3.9+
- An LLM provider (OpenAI API key OR local Ollama)

## Step 1: Install AugmentAI

```bash
# Install from PyPI
pip install augmentai

# Or install from source
git clone https://github.com/kyrozepto/aai.git
cd aai
pip install -e .
```

## Step 2: Set Up LLM Provider

**Option A: OpenAI (Recommended)**
```bash
export OPENAI_API_KEY="your-api-key"
```

**Option B: Ollama (Free, Local)**
```bash
# Install from https://ollama.ai
ollama pull llama3.2
ollama serve
```

## Step 3: Prepare Your Dataset

```bash
# Basic usage (auto-detects domain)
augmentai prepare ./my_dataset

# Specify medical domain
augmentai prepare ./xray_images --domain medical

# Custom split ratio
augmentai prepare ./photos --split 70/15/15

# Preview without writing files
augmentai prepare ./data --dry-run
```

## Step 4: Use the Output

After running `prepare`, you'll have:

```
prepared/
├── data/
│   ├── train/          # Training split
│   ├── val/            # Validation split
│   └── test/           # Test split
├── augment.py          # Run this to augment!
├── config.yaml         # Editable pipeline config
├── manifest.json       # Reproducibility info
└── requirements.txt    # Dependencies
```

**Run augmentations:**
```bash
cd prepared
pip install -r requirements.txt
python augment.py
```

## Step 5: Customize (Optional)

**Edit config.yaml to tweak transforms:**
```yaml
transforms:
  - name: Rotate
    probability: 0.5
    parameters:
      limit: 30  # ← Change rotation range
```

**Or use interactive chat:**
```bash
augmentai chat --domain medical
# Then describe what you want in natural language
```

## Common Options

| Flag | Description |
|------|-------------|
| `--domain` | Force domain (medical, ocr, satellite, natural) |
| `--split` | Train/val/test ratio (default: 80/10/10) |
| `--seed` | Random seed for reproducibility |
| `--dry-run` | Preview without writing files |
| `--preview` | Generate before/after image samples |
| `-v, --verbose` | Detailed output |
| `-q, --quiet` | Minimal output |

## Next Steps

- [Domain Guides](domains/) - Learn domain-specific best practices
- [API Reference](api/) - Use AugmentAI programmatically
- [Architecture](architecture/) - Understand the design
