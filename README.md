# AugmentAI 🎨

**LLM-Powered Data Augmentation Policy Designer**

Design domain-safe, task-aware augmentation policies through natural language conversation. No more manual hyperparameter tuning—just describe your dataset and constraints, and get scientifically sound augmentation pipelines.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- **Natural Language Interface**: Describe your dataset in plain English
- **Domain-Aware Constraints**: Built-in rules for medical, OCR, satellite, and natural images
- **Safety-First Design**: Hard constraints that prevent scientifically invalid augmentations
- **LLM as Advisor**: GPT-4o-mini, Ollama, or LM Studio—LLM suggests, deterministic backend validates
- **Multiple Backends**: Export to Albumentations (Kornia/torchvision coming soon)
- **Reproducible Policies**: Export as YAML, JSON, or executable Python code

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kyrozepto/aai.git
cd aai

# Install in development mode
pip install -e .

# Or install with all backends
pip install -e ".[all]"
```

### Set up your LLM provider

```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key"

# Or use Ollama (free, local)
ollama pull llama3.2
```

### Start designing policies

```bash
# Start interactive chat
augmentai chat --domain medical

# List available domains
augmentai domains

# Validate an existing policy
augmentai validate my_policy.yaml --domain medical
```

## 💬 Example Session

```
$ augmentai chat --domain medical

🎨 AugmentAI - Data Augmentation Policy Designer

You: I'm working on a CT scan segmentation task for lung nodule detection.
     I need augmentations that preserve anatomical structure.

🤖 AugmentAI: For medical CT segmentation, I recommend a conservative policy
that maintains anatomical integrity. Here's my suggestion:

✅ Safe Transforms:
  • HorizontalFlip (p=0.5) - Anatomically valid
  • Rotate (p=0.5, limit=±15°) - Conservative rotation
  • RandomBrightnessContrast (p=0.3, limit=0.1) - Scanner variation
  • GaussNoise (p=0.3, var=0.01) - Noise simulation

⚠️ Automatically excluded (medical domain rules):
  • ElasticTransform - Breaks anatomical structures
  • ColorJitter - Invalid for grayscale CT

/export lung_nodule_policy.py
✓ Exported to lung_nodule_policy.py
```

## 🏥 Domain Safety

AugmentAI enforces **hard constraints** that cannot be overridden:

| Domain | Forbidden Transforms | Reason |
|--------|---------------------|--------|
| Medical | ElasticTransform, GridDistortion, ColorJitter | Breaks anatomy, invalid for grayscale |
| OCR | MotionBlur, ElasticTransform | Destroys text legibility |
| Satellite | ColorJitter, HSV, ChannelShuffle | Breaks spectral band relationships |

## 📁 Project Structure

```
augmentai/
├── cli/              # Command-line interface
├── core/             # Policy data structures
├── domains/          # Domain rule definitions
├── llm/              # LLM client and prompts
├── rules/            # Safety validation
├── compilers/        # Backend code generation
└── examples/         # Example policies
```

## 🔧 Configuration

Create `augmentai.yaml` in your project:

```yaml
llm:
  provider: openai  # or ollama, lmstudio
  model: gpt-4o-mini
  temperature: 0.7

backend: albumentations
output_dir: ./policies
```

## 🎯 Custom Domains

Define your own domain constraints in YAML:

```yaml
# my_domain.yaml
name: my_custom_domain
description: Custom constraints for my task

constraints:
  - transform_name: ElasticTransform
    level: forbidden
    reason: Not suitable for my task

  - transform_name: Rotate
    level: recommended
    reason: Good for rotation invariance
    parameter_limits:
      limit: [-30, 30]

recommended_transforms:
  - HorizontalFlip
  - RandomBrightnessContrast
```

Load with:
```bash
augmentai chat --domain-file my_domain.yaml
```

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Albumentations](https://albumentations.ai/) for the augmentation backend
- [Rich](https://rich.readthedocs.io/) for beautiful terminal UI
- OpenAI, Ollama, LM Studio for LLM support
