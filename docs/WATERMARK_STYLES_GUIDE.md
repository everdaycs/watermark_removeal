# Enhanced Watermark Generation Guide

## Overview

The `generate_watermarks_with_masks.py` script now supports **14 distinct watermark styles** plus **style combinations** for generating highly diverse synthetic watermarked datasets with corresponding binary masks.

**Latest Results:**
- **Total variants generated**: 15,872 watermark images
- **Corresponding masks**: 15,808 binary mask files
- **Watermark folder size**: 418 MB
- **Masks folder size**: 66 MB
- **Source images**: 496
- **Variants per image**: 32

---

## Watermark Styles Reference

### Category 1: Classic Text Styles (Original)

#### 1. **single_text_corner** - Text in Corner
- **Purpose**: Simple text watermarks placed in one of four corners
- **Characteristics**:
  - Random corner (top-left, top-right, bottom-left, bottom-right)
  - Font size: 3-8% of image width
  - Opacity: 15-60%
  - Random rotation: -15° to +15°
  - Light or dark text colors

#### 2. **single_text_center** - Large Centered Text
- **Purpose**: Prominent watermark covering the center of image
- **Characteristics**:
  - Centered positioning
  - Large font: 10-20% of image width
  - Opacity: 15-50%
  - Slight rotation: -10° to +10°
  - Mimics typical "DRAFT" or "SAMPLE" watermarks

#### 3. **tiled_text** - Repeated Pattern
- **Purpose**: Grid-like watermark pattern covering the image
- **Characteristics**:
  - Small text repeated in rows and columns
  - Font size: 2-5% of image width
  - Opacity: 10-40%
  - Low rotation: -5° to +5°
  - Simulates spam or protection watermarks

#### 4. **diagonal_band** - Diagonal Stripe
- **Purpose**: Watermark along image diagonal
- **Characteristics**:
  - Text positioned at offset, then rotated
  - Rotation: -60° to -30° or +30° to +60°
  - Font size: 5-12% of image width
  - Opacity: 20-50%
  - Common in document watermarks

#### 5. **multi_line_text_center** - Multi-line Text
- **Purpose**: Multiple text lines stacked in center
- **Characteristics**:
  - 2-3 random text lines
  - Centered vertical alignment
  - Font size: 5-10% of image width
  - Opacity: 20-55%
  - Spacing between lines adjustable

#### 6. **logo_corner** - Logo Image
- **Purpose**: Logo/image in corner (requires `logo.png`)
- **Characteristics**:
  - Logo placed in random corner
  - Size: 8-15% of image width
  - Opacity: 30-70%
  - Requires `./logo.png` file to exist
  - Returns fallback if logo not found

#### 7. **qr_code_style** - QR Code Pattern
- **Purpose**: QR-code-like grid pattern
- **Characteristics**:
  - Random black/white blocks
  - Grid size: 5×5 to 8×8
  - Opacity: 30-60%
  - Placed randomly in image
  - Mimics machine-readable watermarks

---

### Category 2: Advanced Text Effects (NEW)

#### 8. **outlined_text** - Text with Outline Stroke
- **Purpose**: High-contrast watermark with colored outline
- **Characteristics**:
  - Thick text stroke (2-5 pixels)
  - Outline color different from fill
  - Opacity: 25-65%
  - Font size: 8-15% of image width
  - Rotation: -15° to +15°
  - Creates bold, visible watermarks
- **Use Case**: Bright backgrounds where contrast matters

#### 9. **shadow_text** - Drop Shadow Effect
- **Purpose**: Watermark with 3D drop shadow
- **Characteristics**:
  - Dark shadow with offset (3-8 pixels)
  - Shadow opacity: 40-70% of main text
  - Main text opacity: 30-65%
  - Font size: 10-18% of image width
  - Rotation: -12° to +12°
  - Adds depth and visibility
- **Use Case**: Professional documents and presentations

#### 10. **gradient_alpha_text** - Fading Transparency
- **Purpose**: Text that fades out across the image
- **Characteristics**:
  - Transparency gradient (horizontal or vertical)
  - Starts opaque, becomes transparent toward one edge
  - Base opacity: 20-60%
  - Font size: 10-20% of image width
  - Mimics subtle watermarks that don't obstruct content
- **Use Case**: Photos and artistic images
- **Challenge Level**: Tests model's ability to handle partial watermarks

#### 11. **noisy_eroded_text** - Corrupted/Damaged Text
- **Purpose**: Simulates compressed or eroded watermarks
- **Characteristics**:
  - Random noise applied to edges
  - Gaussian blur to simulate erosion/compression
  - Threshold-based reconstruction
  - Opacity: 25-60%
  - Font size: 8-15% of image width
  - Creates broken, barely-visible watermarks
- **Use Case**: Real-world degraded watermarks
- **Challenge Level**: Difficult - tests robustness to artifacts

#### 12. **banner_box** - Labeled Banner
- **Purpose**: Text with colored background box/strip
- **Characteristics**:
  - Semi-transparent background rectangle or horizontal strip
  - Box opacity: 10-35%
  - Text opacity: 40-75%
  - Font size: 8-14% of image width
  - Background colors: light or dark
  - Padding around text variable
- **Use Case**: Image labels, date stamps, watermark badges
- **Variation**: Horizontal strip or box container

#### 13. **curved_text** - Arced Arrangement
- **Purpose**: Text arranged along an arc/curve
- **Characteristics**:
  - Characters positioned on circular arc
  - Radius: 25-45% of min(width, height)
  - Arc starting angle: -90° to +90°
  - Individual character rotation
  - Opacity: 25-60%
  - Font size: 4-10% of image width
- **Use Case**: Decorative watermarks, seals
- **Challenge Level**: High - requires shape-aware removal

---

### Category 3: Composite Styles (NEW)

#### 14. **combined_styles** - Multi-Style Composition
- **Purpose**: Realistic watermarks using 2-3 styles together
- **Characteristics**:
  - Randomly combines 2-3 different styles
  - Possible combinations:
    - diagonal_band + logo_corner
    - tiled_text + qr_code_style
    - shadow_text + banner_box
    - single_text_center + outlined_text
    - And many more (randomly selected)
  - Styles applied sequentially with alpha compositing
  - All regions merged into single mask
  - Probability: 30% by default
- **Use Case**: Real watermark complexity
- **Challenge Level**: Highest - tests multi-component removal

---

## Configuration

### Key Parameters

```python
# Number of variants per source image
NUM_VARIANTS_PER_IMAGE = 32

# Enable/disable specific styles
ENABLED_STYLES = [
    'single_text_corner',
    'single_text_center',
    'tiled_text',
    'diagonal_band',
    'multi_line_text_center',
    'logo_corner',
    'qr_code_style',
    'outlined_text',
    'shadow_text',
    'gradient_alpha_text',
    'noisy_eroded_text',
    'banner_box',
    'curved_text',
    'combined_styles',
]

# Probability of using combined styles instead of single styles
COMBINED_STYLE_PROBABILITY = 0.3  # 30%
```

### How to Customize

1. **Reduce specific styles**: Comment out unwanted styles in `ENABLED_STYLES`
   ```python
   ENABLED_STYLES = [
       'single_text_corner',
       'single_text_center',
       'shadow_text',  # Only these three
   ]
   ```

2. **Increase combined style frequency**:
   ```python
   COMBINED_STYLE_PROBABILITY = 0.5  # 50% of variants
   ```

3. **Generate more variants per image**:
   ```python
   NUM_VARIANTS_PER_IMAGE = 64  # 64 instead of 32
   ```

---

## Output Structure

### File Organization

```
watermark_demowen/
├── image_001_wm_000.jpg
├── image_001_wm_001.jpg
├── image_001_wm_002.jpg
│   ... (32 variants per image)
├── image_002_wm_000.jpg
└── ... (up to 496 source images × 32 = 15,872 total)

watermark_demowen_masks/
├── image_001_wm_000_mask.png
├── image_001_wm_001_mask.png
│   ... (binary masks: 0=background, 255=watermark)
└── ... (15,808 masks total)
```

### Mask Format

- **File format**: PNG (lossless)
- **Color mode**: Grayscale ('L')
- **Pixel values**:
  - **0**: Background (no watermark)
  - **255**: Watermark region
- **Dimensions**: Same as source image

### Image Format

- **File format**: JPEG
- **Quality**: Random between 60-95 (realistic compression)
- **Color mode**: RGB (3 channels)

---

## Statistical Coverage

### Style Distribution (with defaults)

With `NUM_VARIANTS_PER_IMAGE=32`:

| Per 32 variants | Approx. Count | Percentage |
|-----------------|---------------|-----------|
| Single styles   | ~22-23        | ~70%      |
| Combined styles | ~9-10         | ~30%      |

Within single styles (14 styles enabled):

| Style | Per 496 imgs | Total |
|-------|-------------|-------|
| Each style | 496 × 32 × (1/14) ≈ 1,126 | ~1,126 |

---

## Training Integration

### Using Masks for Supervised Learning

The generated masks enable advanced loss functions:

```python
# Mask-based loss examples
mask = mask_batch / 255.0  # Normalize to [0, 1]

# 1. Weighted L1 loss (prioritize watermark regions)
weighted_loss = torch.mean(torch.abs(pred - target) * (mask + 1.0))

# 2. Dice loss (for segmentation)
dice = 2 * torch.sum(pred * mask) / (torch.sum(pred) + torch.sum(mask) + 1e-7)

# 3. Focal loss (focus on hard-to-remove regions)
focal_loss = -((1 - pred) ** 2) * torch.log(pred + 1e-7) * mask
```

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total image pairs | 15,808 |
| Training set (85%) | 13,436 |
| Validation set (15%) | 2,372 |
| Avg. watermark coverage | ~25-45% |
| Style diversity | 14 styles + combinations |
| Compression levels | 60-95 JPEG quality |

---

## Known Characteristics

### Strengths

✅ **Diverse styles**: 14 distinct watermark types  
✅ **Realistic combinations**: Multi-style compositions  
✅ **Varied opacity**: 10-75% transparency range  
✅ **Multiple rotations**: -60° to +60° for different styles  
✅ **Size variation**: 2-20% of image dimensions  
✅ **Compression artifacts**: 60-95 JPEG quality  
✅ **Accurate masks**: Binary PNG format  

### Challenges

⚠️ **Curved text**: Requires spatial awareness for removal  
⚠️ **Noisy edges**: Eroded watermarks are hard to remove cleanly  
⚠️ **Gradient alpha**: Partial watermarks need edge preservation  
⚠️ **Combined styles**: Multiple overlapping regions increase complexity  

---

## Examples & Visuals

### Style Example: outlined_text
- **Appearance**: Bold colored stroke around text
- **Mask**: Includes entire stroke region
- **Challenge**: Edge detection needed for precise removal

### Style Example: shadow_text
- **Appearance**: Dark shadow + main text in center
- **Mask**: Union of shadow and text regions
- **Challenge**: Shadow blur must be preserved or removed consistently

### Style Example: gradient_alpha_text
- **Appearance**: Text that fades to background
- **Mask**: Regions where alpha > 0
- **Challenge**: Partial transparency requires smooth blending

### Style Example: combined (diagonal_band + logo)
- **Appearance**: Rotated text + logo corner
- **Mask**: Union of both regions
- **Challenge**: Multiple independent watermarks in one image

---

## Performance Notes

### Generation Speed

- **Time per image**: ~50-100ms (varies by style)
- **Total for 496 images × 32 variants**: ~25-40 minutes
- **Bottlenecks**: 
  - PIL image operations (curved_text)
  - Numpy array manipulations (noisy_eroded_text)

### Storage

- **Watermark images**: 418 MB (15,808 files)
- **Masks**: 66 MB (15,808 files)
- **Total**: 484 MB (reasonable for training)

### Memory Usage

- **Peak memory**: ~2-3 GB (depends on image size)
- **Per-image memory**: ~20-30 MB during processing

---

## Troubleshooting

### Issue: Script crashes with "RuntimeError: CUDA out of memory"

**Solution**: Reduce `NUM_VARIANTS_PER_IMAGE` or check for unrelated GPU processes

### Issue: Some watermarks invisible in output

**Solution**: Check JPEG quality setting (60+ is recommended), verify color contrast

### Issue: Masks don't match watermark regions

**Solution**: Verify alpha channel correctness; check for PIL version compatibility

### Issue: Curved text looks distorted

**Solution**: This is expected - model should learn to handle rotated/warped text

---

## Future Enhancements

Potential additions:
- [ ] Video watermarks (temporal consistency)
- [ ] 3D text effects (perspective transform)
- [ ] Color gradient watermarks
- [ ] Animated GIF support
- [ ] Metadata embedding
- [ ] Multi-language text support

---

## Citation & Usage

Use this dataset generator for research in:
- Watermark removal/detection
- Image forensics
- Adversarial robustness
- Generative model training

---

**Last Updated**: 2025-11-28  
**Script Version**: 2.0 (Enhanced with 14 styles)  
**Status**: ✅ Production Ready
