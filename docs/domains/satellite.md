# Satellite Domain Guide

The satellite domain is optimized for remote sensing and multi-spectral imagery including satellite photos, drone imagery, and aerial surveys.

## Safety Philosophy

Spectral integrity matters:
- **Band relationships encode scientific data**
- **Color transforms break spectral signatures**
- **Geometry is more flexible (any rotation valid)**

## Forbidden Transforms

These transforms are **automatically blocked** in satellite domain:

| Transform | Reason |
|-----------|--------|
| `ColorJitter` | Destroys spectral signatures |
| `HueSaturationValue` | Invalid for multi-band data |
| `ChannelShuffle` | Breaks band ordering |
| `RGBShift` | Corrupts spectral relationships |
| `ToSepia` / `ToGray` | Loses band information |

## Recommended Transforms

Safe and effective for satellite imagery:

```yaml
transforms:
  - name: HorizontalFlip
    probability: 0.5
    
  - name: VerticalFlip
    probability: 0.5
    
  - name: Rotate
    probability: 0.5
    parameters:
      limit: 180  # Full rotation is valid
    
  - name: RandomScale
    probability: 0.3
    parameters:
      scale_limit: 0.2
    
  - name: RandomBrightnessContrast
    probability: 0.3
    parameters:
      brightness_limit: 0.1  # Conservative
      contrast_limit: 0.1
```

## Special Considerations

### Multi-Spectral Images

For more than 3 channels, use custom handling:
```yaml
# Preserve all spectral bands
multi_spectral: true
channels: [R, G, B, NIR, SWIR]
```

### Large Tile Handling

Satellite images are often large tiles:
```bash
augmentai prepare ./tiles --domain satellite --patch-size 512
```

### Temporal Consistency

For time-series data, maintain consistency:
```yaml
temporal_augmentation:
  apply_same_to_sequence: true
```

## Example: Land Use Classification

```bash
augmentai prepare ./sentinel2_tiles --domain satellite --task "land use classification"
```

This will:
1. Allow full 360° rotations and all flips
2. Block all color/hue transforms
3. Preserve spectral band relationships
4. Apply scale augmentations for zoom invariance
