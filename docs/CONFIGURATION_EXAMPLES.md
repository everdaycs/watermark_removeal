# Quick Configuration Examples

## Example 1: Only Basic Styles (Fastest)

```python
# In generate_watermarks_with_masks.py, modify:
NUM_VARIANTS_PER_IMAGE = 16

ENABLED_STYLES = [
    'single_text_corner',
    'single_text_center',
    'tiled_text',
]

COMBINED_STYLE_PROBABILITY = 0.0  # Disable combinations
```

**Result**: Fast generation, simple watermarks
- Estimated variants: 496 × 16 = 7,936 images
- Generation time: ~10-15 minutes

---

## Example 2: All Styles with Low Probability for Combinations

```python
NUM_VARIANTS_PER_IMAGE = 24

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

COMBINED_STYLE_PROBABILITY = 0.2  # 20% combinations
```

**Result**: Balanced complexity and diversity
- Estimated variants: 496 × 24 = 11,904 images
- Generation time: ~20-25 minutes

---

## Example 3: Maximum Diversity (Slowest)

```python
NUM_VARIANTS_PER_IMAGE = 48

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

COMBINED_STYLE_PROBABILITY = 0.5  # 50% combinations
```

**Result**: Maximum watermark diversity, best for model robustness
- Estimated variants: 496 × 48 = 23,808 images
- Generation time: ~50-60 minutes
- Storage: ~750 MB watermarks + 130 MB masks

---

## Example 4: Focus on Realistic Watermarks

```python
# Disable artistic/experimental styles
NUM_VARIANTS_PER_IMAGE = 32

ENABLED_STYLES = [
    'single_text_corner',
    'single_text_center',
    'diagonal_band',
    'shadow_text',
    'banner_box',
    'combined_styles',
]

COMBINED_STYLE_PROBABILITY = 0.4  # 40% realistic combinations
```

**Result**: Focus on real-world watermarks
- Estimated variants: 496 × 32 = 15,872 images
- Style focus: Professional documents, social media

---

## Example 5: Challenge Dataset for Model Testing

```python
# Include only difficult styles
NUM_VARIANTS_PER_IMAGE = 32

ENABLED_STYLES = [
    'gradient_alpha_text',    # Partial transparency
    'noisy_eroded_text',      # Damaged watermarks
    'curved_text',            # Spatial warping
    'combined_styles',        # Multi-component
]

COMBINED_STYLE_PROBABILITY = 0.8  # 80% combinations for max difficulty
```

**Result**: Test model robustness on hard cases
- Estimated variants: 496 × 32 = 15,872 images
- Challenge level: Maximum
- Good for adversarial testing

---

## Style Selection Matrix

| Style | Difficulty | Speed | Realism | Coverage |
|-------|-----------|-------|---------|----------|
| single_text_corner | ⭐ | ⚡⚡⚡ | ⭐⭐⭐ | 5-15% |
| single_text_center | ⭐⭐ | ⚡⚡⚡ | ⭐⭐ | 15-30% |
| tiled_text | ⭐ | ⚡⚡ | ⭐⭐ | 30-60% |
| diagonal_band | ⭐⭐ | ⚡⚡⚡ | ⭐⭐⭐ | 20-50% |
| multi_line_text | ⭐ | ⚡⚡ | ⭐⭐ | 15-35% |
| logo_corner | ⭐⭐ | ⚡⚡⚡ | ⭐⭐⭐⭐ | 5-20% |
| qr_code_style | ⭐ | ⚡⚡ | ⭐ | 10-30% |
| outlined_text | ⭐⭐⭐ | ⚡⚡ | ⭐⭐ | 10-25% |
| shadow_text | ⭐⭐ | ⚡⚡ | ⭐⭐⭐ | 15-35% |
| gradient_alpha | ⭐⭐⭐ | ⚡ | ⭐⭐⭐ | 5-40% |
| noisy_eroded | ⭐⭐⭐⭐ | ⚡ | ⭐⭐⭐ | 5-35% |
| banner_box | ⭐⭐ | ⚡⚡ | ⭐⭐⭐⭐ | 15-50% |
| curved_text | ⭐⭐⭐ | ⚡ | ⭐⭐ | 10-30% |
| combined | ⭐⭐⭐⭐ | ⚡ | ⭐⭐⭐⭐ | 15-60% |

---

## How to Run with Custom Configuration

1. **Edit the script**:
   ```bash
   nano generate_watermarks_with_masks.py
   ```

2. **Modify parameters** (around line 35-50):
   ```python
   NUM_VARIANTS_PER_IMAGE = 32
   ENABLED_STYLES = [...]
   COMBINED_STYLE_PROBABILITY = 0.3
   ```

3. **Save and run**:
   ```bash
   cd /home/kaga/Desktop/watermaker\ remover/watermark_removal_project
   source .venv/bin/activate
   python generate_watermarks_with_masks.py
   ```

4. **Monitor progress**:
   ```bash
   watch -n 2 "ls watermark_demowen | wc -l"
   ```

---

## Expected Output Messages

```
============================================================
增强的水印生成脚本 - 支持14种风格和组合
============================================================
输入目录: /home/kaga/Desktop/watermaker remover/20251127_no_watermark_demo
输出目录: ./watermark_demowen
掩码目录: ./watermark_demowen_masks
每张图像变体数: 32
组合风格概率: 30%
启用的风格: single_text_corner, single_text_center, ...
============================================================

找到 496 张图像

处理: image_001
  ✓ 生成 32/32 个变体
处理: image_002
  ✓ 生成 32/32 个变体
...

============================================================
✓ 完成! 生成了 15872 个水印变体
  - 水印图像保存在: ./watermark_demowen
  - 掩码保存在: ./watermark_demowen_masks
============================================================
```

---

## Integration with Training

Once you have generated watermarks and masks, use them in your training script:

```python
# In train.py or dataset.py
dataset = WatermarkDataset(
    watermark_dir="./watermark_demowen",
    mask_dir="./watermark_demowen_masks",
    image_size=256,
    augmentation=True
)

# Use masks in loss function
def compute_loss(pred, target, mask):
    # Prioritize watermark regions
    weighted_l1 = torch.mean(torch.abs(pred - target) * (mask/255 + 1.0))
    
    # Structural similarity on watermark areas only
    ssim_loss = ssim(pred, target, mask=mask)
    
    # Total loss
    loss = weighted_l1 + ssim_loss
    return loss
```

---

## Performance Tuning

### Speed Optimization

```python
# Reduce quality to speed up generation
# Modify in generate_watermark_variant():
# quality = random.randint(60, 75)  # Instead of 60-95
```

### Memory Optimization

```python
# Process in batches instead of all at once
# (Already implemented, but can be further optimized)
```

### Quality Optimization

```python
# Increase JPEG quality for sharper watermarks
# quality = random.randint(85, 98)  # Instead of 60-95
```

---

**Tips**: Start with Example 2 for a good balance between diversity and speed!
