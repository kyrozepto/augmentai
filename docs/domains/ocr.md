# OCR Domain Guide

The OCR domain is optimized for text recognition tasks including scanned documents, receipts, handwriting, and license plates.

## Safety Philosophy

Text legibility is paramount:
- **Characters must remain readable**
- **Blur and distortion destroy text**
- **Spatial relationships matter for words**

## Forbidden Transforms

These transforms are **automatically blocked** in OCR domain:

| Transform | Reason |
|-----------|--------|
| `MotionBlur` | Makes text unreadable |
| `ElasticTransform` | Warps character shapes |
| `GridDistortion` | Breaks character alignment |
| `GaussianBlur` (heavy) | Destroys text edges |
| `Downscale` (extreme) | Loses fine details |

## Recommended Transforms

Safe for text preservation:

```yaml
transforms:
  - name: Rotate
    probability: 0.5
    parameters:
      limit: 5  # Very conservative rotation
    
  - name: RandomBrightnessContrast
    probability: 0.3
    parameters:
      brightness_limit: 0.1
      contrast_limit: 0.1
    
  - name: GaussNoise
    probability: 0.2
    parameters:
      var_limit: [5, 15]  # Light noise only
    
  - name: Sharpen
    probability: 0.2
    # Enhances text edges
    
  - name: CLAHE
    probability: 0.3
    # Improves contrast for faded documents
```

## Special Considerations

### Document Scanning Artifacts

Add realistic scanning artifacts:
```yaml
transforms:
  - name: ImageCompression
    probability: 0.2
    parameters:
      quality_lower: 85
      quality_upper: 100
```

### Perspective for Natural OCR

For license plates or signs:
```yaml
transforms:
  - name: Perspective
    probability: 0.3
    parameters:
      scale: 0.05  # Subtle perspective change
```

## Example: Receipt Scanning

```bash
augmentai prepare ./receipts --domain ocr --task "receipt text extraction"
```

This will:
1. Apply very conservative rotations (±5°)
2. Avoid any blur transforms
3. Add subtle brightness variations
4. Preserve text sharpness
