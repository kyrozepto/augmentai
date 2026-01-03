# Medical Domain Guide

The medical domain is designed for diagnostic imaging including X-rays, CT scans, MRIs, and histopathology slides.

## Safety Philosophy

Medical images have strict requirements:
- **Anatomical correctness must be preserved**
- **Diagnostic features cannot be distorted**
- **Grayscale integrity must be maintained**

## Forbidden Transforms

These transforms are **automatically blocked** in medical domain:

| Transform | Reason |
|-----------|--------|
| `ElasticTransform` | Distorts anatomical structures unrealistically |
| `GridDistortion` | Creates non-physical deformations |
| `ColorJitter` | Medical images are typically grayscale |
| `OpticalDistortion` | Warps diagnostic features |
| `RandomSolarize` | Inverts pixel values incorrectly |
| `ChannelShuffle` | Invalid for single-channel images |

## Recommended Transforms

Safe and effective for medical imaging:

```yaml
transforms:
  - name: HorizontalFlip
    probability: 0.5
    # Note: Use only if anatomy is symmetric
    
  - name: Rotate
    probability: 0.5
    parameters:
      limit: 15  # Conservative rotation limit
    
  - name: RandomBrightnessContrast
    probability: 0.3
    parameters:
      brightness_limit: 0.1
      contrast_limit: 0.1
    
  - name: GaussNoise
    probability: 0.2
    parameters:
      var_limit: [5, 20]
    
  - name: CLAHE
    probability: 0.3
    parameters:
      clip_limit: 2.0
```

## Special Considerations

### Patient-Level Splitting

For medical datasets, use patient-level splitting to prevent data leakage:

```bash
augmentai prepare ./medical_data --strategy group --group-by patient_id
```

### Mask Consistency

When augmenting segmentation masks, AugmentAI automatically applies the same transforms to both image and mask.

## Example: Dental Panoramic X-Ray

```bash
augmentai prepare ./dental_xrays --domain medical --task "cavity detection"
```

This will:
1. Block transforms that could distort teeth
2. Apply conservative rotations (±15°)
3. Add mild noise to simulate different X-ray machines
4. Preserve grayscale values
