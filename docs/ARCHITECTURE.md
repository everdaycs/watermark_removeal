# Script Architecture & Implementation Details

## File Structure Overview

```
generate_watermarks_with_masks.py
├── Configuration Section (lines 1-50)
│   ├── Global parameters
│   ├── Style enable/disable list
│   └── Probability settings
│
├── Helper Functions (lines 52-100)
│   ├── ensure_dirs()
│   ├── get_image_files()
│   ├── load_image()
│   ├── get_font()
│   ├── get_text_bbox()
│   └── get_estimated_brightness()
│
├── Style Functions (lines 102-520)
│   ├── Group 1: Classic Styles (lines 102-260)
│   │   ├── single_text_corner()
│   │   ├── single_text_center()
│   │   ├── tiled_text()
│   │   ├── diagonal_band()
│   │   ├── multi_line_text_center()
│   │   ├── logo_corner()
│   │   └── qr_code_style()
│   │
│   ├── Group 2: Advanced Styles (lines 262-480)
│   │   ├── outlined_text()
│   │   ├── shadow_text()
│   │   ├── gradient_alpha_text()
│   │   ├── noisy_eroded_text()
│   │   ├── banner_box()
│   │   └── curved_text()
│   │
│   └── Group 3: Composite (lines 482-520)
│       └── combined_styles()
│
├── Main Processing (lines 522-620)
│   ├── generate_watermark_variant()
│   ├── process_image()
│   └── main()
│
└── Entry Point (lines 622-625)
    └── if __name__ == '__main__'
```

---

## Key Functions Reference

### 1. Style Function Signature

Every style function follows this pattern:

```python
def style_name(w, h):
    """
    Style description
    
    Args:
        w (int): Image width
        h (int): Image height
    
    Returns:
        PIL.Image (RGBA): Overlay image with watermark
    """
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # ... generate watermark ...
    
    return overlay
```

### 2. Parameters Flow

```
generate_watermark_variant()
    ├─ Load image
    ├─ Select style (random or combined)
    ├─ Call style_function(w, h)
    │   └─ Returns RGBA overlay
    ├─ Composite with image
    ├─ Generate mask from alpha
    └─ Save image & mask pair
```

### 3. Mask Generation Logic

```python
# Extract alpha channel
overlay_array = np.array(overlay)
alpha_mask = overlay_array[:, :, 3] > 0  # Boolean mask

# Create binary mask image
mask_array = np.zeros((h, w), dtype=np.uint8)
mask_array[alpha_mask] = 255  # 255 where watermark exists

# Save as PNG
mask = Image.fromarray(mask_array, mode='L')
mask.save(mask_path, 'PNG')
```

---

## Style Implementation Details

### Text-Based Styles (1-5, 8-10, 12-13)

**Common pattern**:
1. Get text dimensions
2. Calculate position (based on style)
3. Draw text with color + alpha
4. Optional rotation
5. Return overlay

**Key variables**:
- `font_size`: int, typically 3-20% of image width
- `alpha`: float, 0.0-1.0 (converted to 0-255 for PIL)
- `color`: tuple (R, G, B) in [0-255]
- `rotation`: float, degrees (-60 to +60)

### Effect-Based Styles (9, 10, 11)

**Shadow (Style 9)**:
1. Draw shadow (dark color, offset)
2. Draw main text on top
3. Both have alpha values
4. Combined alpha creates depth effect

**Gradient Alpha (Style 10)**:
1. Draw text at full opacity
2. Create gradient mask (1D to 2D expansion)
3. Multiply alpha channel by gradient
4. Result: Text that fades gradually

**Noisy Eroded (Style 11)**:
1. Draw text normally
2. Extract alpha channel
3. Add Gaussian noise
4. Apply Gaussian blur
5. Threshold to restore edges
6. Result: Damaged/compressed appearance

### Container-Based Styles (12)

**Banner Box**:
1. Calculate text dimensions
2. Create rectangle with padding
3. Draw box (low opacity)
4. Draw text on top (higher opacity)
5. Multiple color and direction variants

### Complex Styles (13, 14)

**Curved Text (Style 13)**:
```python
for i, char in enumerate(char_list):
    angle = arc_start_angle + i * angle_step
    # Calculate position on circle
    char_x = center_x + radius * cos(angle)
    char_y = center_y + radius * sin(angle)
    # Rotate character to match arc
    char_img = char_img.rotate(-angle)
    # Paste on main overlay
    overlay.paste(char_img, (char_x, char_y), char_img)
```

**Combined Styles (Style 14)**:
```python
overlay = Image.new('RGBA', (w, h), transparent)
for selected_style in randomly_select_2_3_styles():
    style_overlay = selected_style(w, h)
    overlay = Image.alpha_composite(overlay, style_overlay)
return overlay
```

---

## Random Parameter Ranges

| Parameter | Min | Max | Type | Notes |
|-----------|-----|-----|------|-------|
| Font size % | 0.02 | 0.20 | Relative | % of image width |
| Alpha (opacity) | 0.10 | 0.75 | Absolute | 10-75% transparency |
| Rotation | -60 | +60 | Degrees | Per-style limits |
| Outline width | 2 | 5 | Pixels | For outlined_text |
| Shadow offset | 3 | 8 | Pixels | dx, dy in pixels |
| Gradient radius | 0.25 | 0.45 | Relative | % of min(w, h) |
| JPEG quality | 60 | 95 | % | Compression factor |

---

## Color Palettes

### Light Colors (for dark backgrounds)
```python
[(255, 255, 255),      # Pure white
 (230, 230, 230),      # Off-white
 (200, 200, 200)]      # Light gray
```

### Dark Colors (for light backgrounds)
```python
[(0, 0, 0),            # Pure black
 (30, 30, 30),         # Off-black
 (50, 50, 50),         # Dark gray
 (100, 100, 100),      # Medium gray
 (150, 150, 150)]      # Light gray
```

### Accent Colors (for contrast)
```python
[(200, 0, 0),          # Red
 (100, 150, 200),      # Blue
 (100, 100, 100),      # Gray
 (200, 100, 100)]      # Muted red
```

---

## Composite Logic

When combining styles (Style 14), the script uses **alpha compositing**:

```python
overlay1 = style_func_1(w, h)  # Returns RGBA
overlay2 = style_func_2(w, h)  # Returns RGBA

# Alpha composite merges layers
combined = Image.alpha_composite(overlay1, overlay2)

# Result: Union of both alphas
# Mask covers regions from both styles
```

**Mathematical model**:
- $\alpha_{result}(x,y) = \alpha_1(x,y) + \alpha_2(x,y) \cdot (1 - \alpha_1(x,y))$
- Where $\alpha \in [0, 1]$ for each pixel

**Mask consequence**:
- Pixel is 255 if $\alpha_{result} > 0$
- Overlapping regions stay 255 (OR logic)

---

## Error Handling

### Try-Catch Strategy

```python
def generate_watermark_variant(image_path, output_index):
    try:
        # Load image
        image = load_image(image_path)
        if image is None:
            return False, None, None, None
        
        # Select style
        try:
            overlay = selected_style(w, h)
        except Exception as e:
            print(f"Warning: Style failed - {e}")
            return False, None, None, None
        
        # Composite and save
        ...
        
    except Exception as e:
        print(f"Error: {e}")
        return False, None, None, None
```

### Fallback Mechanisms

1. **Style fails** → Skip variant, continue to next
2. **Logo missing** → Return None from `logo_corner()`, use empty overlay
3. **Scipy missing** → Fall back to PIL's `ImageFilter.GaussianBlur()`
4. **Font missing** → Use `ImageFont.load_default()`

---

## Performance Characteristics

### Time Complexity (per image)

| Style | Complexity | Notes |
|-------|-----------|-------|
| single_text_* | O(1) | Simple operations |
| tiled_text | O(w×h / text_size²) | Grid loops |
| qr_code_style | O(grid_size²) | Grid generation |
| gradient_alpha | O(w×h) | Pixel-wise gradient |
| noisy_eroded | O(w×h) | Convolution (blur) |
| curved_text | O(len(text)) | Per-character ops |
| combined | O(styles × comp_time) | Sequential |

### Memory Profile

```
Per image (512×512):
├── Input image: ~0.8 MB (RGBA)
├── Overlay: ~1 MB (RGBA)
├── Mask: ~0.25 MB (grayscale)
├── Intermediate: ~2-3 MB (numpy arrays)
└── Total: ~4-5 MB peak
```

---

## Extensibility

### Adding a New Style

```python
def new_style_name(w, h):
    """Your new watermark style"""
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # ... implementation ...
    
    return overlay

# Then add to ENABLED_STYLES:
ENABLED_STYLES = [
    # ... existing styles ...
    'new_style_name',
]

# And to generate_watermark_variant():
if 'new_style_name' in ENABLED_STYLES:
    style_functions.append(('new_style_name', 
                           lambda: new_style_name(w, h)))
```

---

## Dependencies

```python
import os              # File operations
import random          # Random selection
import math            # Trigonometry (curved_text)
from pathlib import Path  # Path handling
from PIL import Image, ImageDraw, ImageFont, ImageFilter
                       # Image manipulation
import numpy as np     # Array operations (masks)

# Optional:
from scipy import ndimage  # Advanced filtering (noisy_eroded_text)
```

---

## Debugging Tips

### Verify a single style:

```python
from PIL import Image

# Test single_text_corner
overlay = single_text_corner(512, 512)
overlay.save('test_overlay.png')

# Convert to mask
overlay_array = np.array(overlay)
alpha_mask = overlay_array[:, :, 3] > 0
mask_array = np.zeros((512, 512), dtype=np.uint8)
mask_array[alpha_mask] = 255
Image.fromarray(mask_array, 'L').save('test_mask.png')
```

### Check generated file structure:

```bash
# Count files per image
for img in watermark_demowen/*_wm_000.jpg; do
    base=$(basename "$img" _wm_000.jpg)
    echo "Image: $base"
    ls watermark_demowen/${base}_wm_*.jpg | wc -l
done
```

### Verify mask-watermark consistency:

```python
from PIL import Image
import numpy as np

# Load watermark and mask
watermark = Image.open('watermark_demowen/image_001_wm_000.jpg')
mask = Image.open('watermark_demowen_masks/image_001_wm_000_mask.png')

# Both should have same dimensions
assert watermark.size == mask.size
print(f"✓ Size match: {watermark.size}")

# Check mask values are 0 or 255
mask_array = np.array(mask)
assert set(np.unique(mask_array)) == {0, 255}
print(f"✓ Mask values correct")
```

---

## Future Optimization Opportunities

1. **Parallel processing**: Use multiprocessing for image generation
2. **Batch rendering**: Pre-compute font metrics
3. **GPU acceleration**: Use CUDA for alpha compositing (large-scale)
4. **Caching**: Store rendered fonts between images
5. **Lazy loading**: Stream images instead of loading all

---

**Version**: 2.0 Enhanced  
**Last Updated**: 2025-11-28  
**Maintainability**: High (modular design)  
**Extensibility**: Easy (add new style functions)
