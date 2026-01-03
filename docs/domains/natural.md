# Natural Images Domain Guide

The natural domain is for general photography including pet photos, product images, social media content, and any natural scenes.

## Safety Philosophy

Maximum flexibility:
- **No scientific constraints**
- **All transforms generally valid**
- **Strong augmentation for robust models**

## Forbidden Transforms

**None** - The natural domain has no forbidden transforms.

All augmentations are allowed, including:
- Color jitter, hue/saturation adjustments
- Elastic and grid distortions
- Heavy blur and noise
- Cutout, mixup, and other aggressive techniques

## Recommended Transforms

Effective augmentation strategy:

```yaml
transforms:
  - name: HorizontalFlip
    probability: 0.5
    
  - name: VerticalFlip
    probability: 0.2
    
  - name: Rotate
    probability: 0.5
    parameters:
      limit: 30
    
  - name: RandomBrightnessContrast
    probability: 0.5
    parameters:
      brightness_limit: 0.2
      contrast_limit: 0.2
    
  - name: ColorJitter
    probability: 0.3
    
  - name: GaussNoise
    probability: 0.3
    parameters:
      var_limit: [10, 50]
    
  - name: Blur
    probability: 0.2
    parameters:
      blur_limit: 7
    
  - name: CoarseDropout  # Cutout variant
    probability: 0.2
```

## Advanced Techniques

### AutoAugment-style Policies

```yaml
augment_style: autoaugment
# Uses learned augmentation policies
```

### Progressive Augmentation

```yaml
curriculum:
  start_strength: 0.3
  end_strength: 1.0
  epochs: 100
```

## Example: Pet Classification

```bash
augmentai prepare ./pet_photos --domain natural --task "breed classification"
```

This will:
1. Apply strong geometric transforms
2. Add color variations
3. Include random erasing/cutout
4. Maximize data diversity for robust models
