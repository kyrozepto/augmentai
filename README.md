# AugmentAI 🎨

**LLM-Powered Data Augmentation Policy Designer**

Design domain-safe, task-aware augmentation policies through natural language conversation. No more manual hyperparameter tuning—just describe your dataset and constraints, and get scientifically sound augmentation pipelines.

> **Design Philosophy**: The LLM suggests. Rules decide. Code executes.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- **One-Command Preparation**: `augmentai prepare ./dataset` - inspect, split, augment, export
- **Natural Language Interface**: Describe your dataset in plain English
- **Domain-Aware Constraints**: Built-in rules for medical, OCR, satellite, and natural images
- **Safety-First Design**: Hard constraints that prevent scientifically invalid augmentations
- **Full Reproducibility**: Seed locking, manifest tracking, deterministic pipelines
- **LLM as Advisor**: GPT-4o-mini, Ollama, or LM Studio—LLM suggests, rules validate
- **Executable Output**: Generate standalone Python scripts ready to run

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

### One-Command Dataset Preparation

```bash
# Prepare dataset with auto-detected domain
augmentai prepare ./dataset

# Medical domain with custom split
augmentai prepare ./lung_ct --domain medical --split 70/15/15

# Preview what would happen (dry run)
augmentai prepare ./images --dry-run
```

### Interactive Policy Design

```bash
# Start interactive chat
augmentai chat --domain medical

# List available domains
augmentai domains

# Validate an existing policy
augmentai validate my_policy.yaml --domain medical
```

## � Output Structure

Running `augmentai prepare` generates:

```
prepared/
├── data/
│   ├── train/           # Training split
│   ├── val/             # Validation split
│   └── test/            # Test split
├── augmented/           # Output for augmented data
├── augment.py           # Standalone augmentation script
├── config.yaml          # Pipeline configuration
├── manifest.json        # Reproducibility manifest
├── requirements.txt     # Dependencies
└── README.md            # Usage instructions
```

## 🎯 Real-World Examples

### 🏥 Medical: Dental Panoramic X-Ray

```bash
augmentai prepare "C:\Users\you\datasets\panoramic-xray" --domain medical --split 70/15/15

# Output:
# ✓ Detected: 1,247 images across 4 classes (cavity, healthy, fracture, caries)
# ✓ Split: 873 train, 187 val, 187 test
# ✓ Policy: Conservative augmentations preserving dental anatomy
# ✓ Forbidden: ElasticTransform, ColorJitter (would distort teeth/bone structure)
```

### 🛰️ Satellite: Land Use Classification

```bash
augmentai prepare ./sentinel2_tiles --domain satellite --seed 42

# Multi-spectral imagery gets special treatment:
# ✓ Allowed: Rotation (any angle), flips, scale
# ✓ Forbidden: ColorJitter, HSV, ChannelShuffle (breaks spectral bands!)
```

### 📝 OCR: Document Scanning

```bash
augmentai prepare ./scanned_receipts --domain ocr

# Text legibility is preserved:
# ✓ Allowed: Slight rotation (±5°), brightness adjustment
# ✓ Forbidden: MotionBlur, ElasticTransform (destroys text)
```

### 🖼️ Natural: Instagram-style Photos

```bash
augmentai prepare ./pet_photos --domain natural

# Maximum flexibility for general images:
# ✓ Allowed: Everything! Color jitter, elastic, cutout, mixup
# ✓ Strong augmentations for robust models
```

### 🔬 Research: Reproducible Experiments

```bash
# Exact same augmentations every time
augmentai prepare ./experiment_data --seed 12345 --output ./exp_v1

# Later, reproduce with the manifest
cat ./exp_v1/manifest.json
# {
#   "seed": 12345,
#   "dataset_hash": "a1b2c3d4...",
#   "policy_hash": "e5f6g7h8...",
#   "augmentai_version": "0.1.0"
# }
```

### 🏭 Production: Batch Processing

```bash
# Prepare multiple datasets with consistent settings
for dataset in chest_xray brain_mri skin_lesion; do
    augmentai prepare ./raw/$dataset --domain medical --output ./prepared/$dataset
done
```

### 💻 Python API: Full Control

```python
from augmentai.core import Policy, Transform
from augmentai.domains import get_domain
from augmentai.rules.enforcement import RuleEnforcer
from augmentai.export import ScriptGenerator

# Create custom policy
policy = Policy(
    name="dental_xray_v2",
    domain="medical",
    transforms=[
        Transform("HorizontalFlip", 0.5),
        Transform("Rotate", 0.7, parameters={"limit": 20}),
        Transform("CLAHE", 0.4, parameters={"clip_limit": 2.0}),
        Transform("GaussNoise", 0.3, parameters={"var_limit": (10, 30)}),
        Transform("RandomBrightnessContrast", 0.5),
    ]
)

# Validate against medical rules
enforcer = RuleEnforcer(get_domain("medical"))
result = enforcer.enforce_policy(policy)
print(f"Safe transforms: {len(result.policy.transforms)}")

# Generate standalone script
generator = ScriptGenerator()
script = generator.generate_augment_script(result.policy, seed=42)
with open("my_augment.py", "w") as f:
    f.write(script)
```

### 🔄 Modifying Generated Config

After running `augmentai prepare`, edit `config.yaml`:

```yaml
# config.yaml - tweak as needed!
policy:
  name: dental_xray_policy
  domain: medical
  seed: 42
transforms:
  - name: HorizontalFlip
    probability: 0.5          # ← Increase to 0.7?
  - name: Rotate
    probability: 0.5
    parameters:
      limit: 15               # ← Bump to 25 for more variety
  - name: CLAHE               # ← Add new transforms!
    probability: 0.4
    parameters:
      clip_limit: 2.0
```

Then re-run the generated script:
```bash
python augment.py --input data/train --output augmented/train --seed 42
```

### 🎓 Interactive Learning Session

```
$ augmentai chat --domain medical

You: I have dental panoramic x-rays with cavity annotations. What augmentations 
     are safe? I'm worried about distorting the teeth.

🤖 AugmentAI: Great concern! For dental x-rays, I recommend:

✅ Safe for dental imaging:
  • HorizontalFlip (p=0.5) - Teeth are roughly symmetric
  • Rotate (p=0.5, limit=±10°) - Small rotations are realistic
  • CLAHE (p=0.3) - Enhances contrast in x-rays
  • GaussNoise (p=0.2) - Simulates sensor noise

⚠️ Automatically blocked:
  • ElasticTransform - Would warp tooth boundaries unrealistically
  • GridDistortion - Same issue, breaks tooth geometry  
  • ColorJitter - X-rays are grayscale, color transforms invalid

You: Can I increase rotation to 30 degrees?

🤖 AugmentAI: I'd advise caution. Dental panoramic x-rays are typically taken 
with the patient positioned consistently. Rotations beyond ±15° might create 
unrealistic training samples. However, if your dataset has varied positioning, 
±20° could work. I'll set it to 20° as a compromise.

/preview
┌────────────────────────────┬─────────────┬──────────────────────┐
│ Transform                  │ Probability │ Parameters           │
├────────────────────────────┼─────────────┼──────────────────────┤
│ HorizontalFlip             │ 50%         │ -                    │
│ Rotate                     │ 50%         │ limit=20             │
│ CLAHE                      │ 30%         │ clip_limit=2.0       │
│ GaussNoise                 │ 20%         │ var_limit=(5, 25)    │
│ RandomBrightnessContrast   │ 30%         │ brightness_limit=0.1 │
└────────────────────────────┴─────────────┴──────────────────────┘

/export dental_cavity_policy.py
✓ Exported to dental_cavity_policy.py
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
├── cli/              # CLI commands (prepare, chat, validate)
├── core/             # Policy, Transform, Manifest, Pipeline
├── domains/          # Domain rules (medical, ocr, satellite, natural)
├── inspection/       # Dataset auto-detection & analysis
├── splitting/        # Train/val/test splitting strategies
├── export/           # Script & folder generation
├── llm/              # LLM client and prompts
├── rules/            # Safety validation & enforcement
└── compilers/        # Backend code generation
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

## 🗺️ Extended Roadmap

### Near Term (v0.2 – v0.3): Make AugmentAI Trustworthy at Scale

*Focus: Data correctness & confidence*

- [x] **Dataset Linter (Pre-Prepare)** ✅
  - Detect duplicates, corrupt images, mismatched masks
  - Warn about class imbalance and label leakage
  - Runs automatically before prepare (`--skip-lint`, `--lint-only`)

- [x] **Augmentation Safety Validator** ✅
  - Test augmentation–label consistency
  - Flag transforms that break segmentation masks or OCR legibility
  - Critical for medical & OCR domains

- [x] **Augmentation Preview & Diff** ✅
  - Visual before/after samples
  - Show what changed per transform
  - HTML/JSON dry-run reports (`--preview`, `--preview-count`)


### Mid Term (v0.4 – v0.5): Make AugmentAI Evidence-Driven

*Focus: Prove augmentations help*

- [ ] **Automatic Augmentation Ablation**
  - Measure contribution of each transform
  - Rank augmentations by validation impact
  - Export ablation reports

- [ ] **Augmentation-Aware Robustness Metrics**
  - Evaluate model sensitivity per augmentation
  - Identify fragile invariances early

- [ ] **Policy Comparison & Versioning**
  - Diff augmentation policies
  - Track changes across experiments
  - Integrate with DVC / dataset manifests

### Long Term (v0.6+): Close the Data–Model Loop

*Focus: Data-centric learning*

- [ ] **Model-Guided Data Repair**
  - Use uncertainty to suggest relabel / reweight / remove samples
  - Feedback loop from trained model to data prep

- [ ] **Curriculum-Aware Dataset Preparation**
  - Order data from easy → hard
  - Adaptive augmentation strength over epochs

- [ ] **Domain Shift Simulation**
  - Generate controlled distribution shifts
  - Stress-test generalization before deployment

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Albumentations](https://albumentations.ai/) for the augmentation backend
- [Rich](https://rich.readthedocs.io/) for beautiful terminal UI
- OpenAI, Ollama, LM Studio for LLM support

