# SVG Watermark Styles - Code Examples & API Reference

## API Reference

### Core Helper Function

#### `load_svg_as_rgba(svg_path, target_width, target_height) → Image.Image`

Render SVG to PIL RGBA image with caching.

**Parameters**:
- `svg_path` (str): Path to SVG file
- `target_width` (int): Target width in pixels
- `target_height` (int): Target height in pixels

**Returns**: 
- PIL Image (RGBA) if successful
- None if failed

**Caching**:
- Key: `(svg_path, target_width, target_height)`
- Duration: Entire session
- Lookup: O(1) hash lookup

**Example**:
```python
from PIL import Image
from tools.generate_watermarks_with_masks import load_svg_as_rgba

# First call: renders SVG (50-200ms)
logo = load_svg_as_rgba("./logos/elecfans-logo.svg", 150, 100)

# Second call: cached (<1ms)
logo2 = load_svg_as_rgba("./logos/elecfans-logo.svg", 150, 100)

# Verify RGBA format
assert logo.mode == 'RGBA'
assert logo.size == (150, 100)
```

### Style Functions

#### 1. `elecfans_logo_svg_corner(w, h) → Image.Image`

Generate ElecFans logo watermark in corner.

**Parameters**:
- `w` (int): Image width
- `h` (int): Image height

**Returns**: RGBA overlay image (transparent background)

**Example**:
```python
overlay = elecfans_logo_svg_corner(1024, 768)
# Returns RGBA image with logo in corner
# Logo size: ~77-128px high (6-10% of 768)
# Logo position: bottom-right or bottom-left
# Alpha: 30-50%
```

#### 2. `elecfans_web_svg_center(w, h) → Image.Image`

Generate ElecFans web watermark centered.

**Parameters**:
- `w` (int): Image width
- `h` (int): Image height

**Returns**: RGBA overlay image

**Example**:
```python
overlay = elecfans_web_svg_center(1024, 768)
# Returns RGBA image with web banner centered
# Width: ~307-512px (30-50% of 1024)
# Height: ~77-138px (10-18% of 768)
# Position: Horizontally centered, vertically ±38px offset
# Alpha: 30-50%
```

#### 3. `wechat_svg_corner_id(w, h) → Image.Image`

Generate WeChat social watermark with account ID.

**Parameters**:
- `w` (int): Image width
- `h` (int): Image height

**Returns**: RGBA overlay image

**Example**:
```python
overlay = wechat_svg_corner_id(1024, 768)
# Returns RGBA image with WeChat icon + text
# Components:
#   - Icon: 23-38px square (3-6% of height)
#   - Text: White, ≥14px font
#   - Layout: [Icon] [Spacing] [Text]
# Position: 90% bottom-right, 10% bottom-left
# Total width: ~82-154px (8-15% of 1024)
# Alpha: 30-50%
```

## Configuration Examples

### Enable Only SVG Styles

```python
# In generate_watermarks_with_masks.py:

ENABLED_STYLES = [
    'elecfans_logo_svg_corner',   # Only SVG styles
    'elecfans_web_svg_center',
    'wechat_svg_corner_id',
]

# All 32 variants will use SVG watermarks
```

### Adjust SVG Style Probabilities

```python
# Make WeChat more common (50% of watermarks)
WECHAT_SVG_WEIGHT = 0.50

# Disable ElecFans Logo (remove from ENABLED_STYLES)
ENABLED_STYLES = [
    # ... other styles ...
    # 'elecfans_logo_svg_corner',  # Commented out
    'elecfans_web_svg_center',
    'wechat_svg_corner_id',
]
```

### Use Custom SVG Files

```python
# Add new SVG paths
CUSTOM_LOGO_SVG = "./logos/my-custom-logo.svg"

# Create wrapper function
def custom_logo_corner(w, h):
    """Custom logo watermark"""
    target_height = int(h * random.uniform(0.06, 0.10))
    target_width = int(w * random.uniform(0.10, 0.18))
    
    svg_img = load_svg_as_rgba(CUSTOM_LOGO_SVG, target_width, target_height)
    if svg_img is None:
        return None
    
    # Apply transparency
    alpha = random.uniform(MIN_ALPHA, MAX_ALPHA)
    svg_array = np.array(svg_img)
    svg_array[:, :, 3] = (svg_array[:, :, 3] * alpha).astype(np.uint8)
    svg_img = Image.fromarray(svg_array, 'RGBA')
    
    # Position in overlay
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    pos = (w - 24 - svg_img.width, h - 24 - svg_img.height)
    overlay.paste(svg_img, pos, svg_img)
    
    return overlay

# Register
ENABLED_STYLES = [
    # ... other styles ...
    'custom_logo_corner',
]

# ... and add to style_functions list in generate_watermark_variant()
```

## Integration Examples

### Training Integration

```python
import torch
from dataset import WatermarkDataset
from models import UNet

# Dataset includes all 20 watermark styles (16 text + 3 SVG + 1 combined)
dataset = WatermarkDataset(
    source_dir="20251127_no_watermark_demo",
    watermarked_dir="watermark_demowen",
    mask_dir="watermark_demowen_masks"
)

# Model trains on diverse watermarks
model = UNet()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    for i, batch in enumerate(dataset):
        clean = batch['clean'].to(device)
        watermarked = batch['watermarked'].to(device)
        mask = batch['mask'].to(device)
        
        # Forward pass
        output = model(watermarked)
        
        # Loss combines removal + mask prediction
        removal_loss = criterion(output, clean)
        mask_loss = mask_criterion(model.mask_pred, mask)
        
        total_loss = removal_loss + 0.5 * mask_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

### Custom SVG Rendering Pipeline

```python
from PIL import Image
import numpy as np
from tools.generate_watermarks_with_masks import load_svg_as_rgba

def apply_svg_watermark(image_path, svg_path, position='bottom-right'):
    """
    Apply SVG watermark to image file
    
    Args:
        image_path: Input image file
        svg_path: SVG watermark file
        position: 'bottom-right', 'bottom-left', 'center', 'top-right', etc.
    
    Returns:
        PIL Image with watermark applied
    """
    # Load source image
    image = Image.open(image_path).convert('RGBA')
    w, h = image.size
    
    # Render SVG
    svg_height = int(h * 0.10)  # 10% of image height
    svg_width = int(w * 0.15)   # 15% of image width
    
    svg_img = load_svg_as_rgba(svg_path, svg_width, svg_height)
    if svg_img is None:
        return image
    
    # Apply transparency
    alpha = 0.4  # 40% opacity
    svg_array = np.array(svg_img)
    svg_array[:, :, 3] = (svg_array[:, :, 3] * alpha).astype(np.uint8)
    svg_img = Image.fromarray(svg_array, 'RGBA')
    
    # Position watermark
    margin = 16
    positions = {
        'bottom-right': (w - margin - svg_width, h - margin - svg_height),
        'bottom-left': (margin, h - margin - svg_height),
        'top-right': (w - margin - svg_width, margin),
        'top-left': (margin, margin),
        'center': ((w - svg_width) // 2, (h - svg_height) // 2),
    }
    
    pos = positions.get(position, positions['bottom-right'])
    
    # Composite watermark
    image.paste(svg_img, pos, svg_img)
    
    return image

# Usage
result = apply_svg_watermark(
    "photo.jpg",
    "./logos/elecfans-logo.svg",
    position='bottom-right'
)
result.save("watermarked.jpg")
```

### Batch SVG Watermarking

```python
import os
from pathlib import Path
from tools.generate_watermarks_with_masks import load_svg_as_rgba
from PIL import Image
import numpy as np

def batch_apply_svg(input_dir, output_dir, svg_path, alpha=0.4):
    """
    Apply SVG watermark to all images in directory
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # Load image
            image = Image.open(input_path).convert('RGBA')
            w, h = image.size
            
            # Render SVG
            svg_height = int(h * 0.08)
            svg_width = int(w * 0.12)
            svg_img = load_svg_as_rgba(svg_path, svg_width, svg_height)
            
            if svg_img is None:
                image.save(output_path)
                continue
            
            # Apply transparency
            svg_array = np.array(svg_img)
            svg_array[:, :, 3] = (svg_array[:, :, 3] * alpha).astype(np.uint8)
            svg_img = Image.fromarray(svg_array, 'RGBA')
            
            # Position
            pos = (w - 16 - svg_width, h - 16 - svg_height)
            image.paste(svg_img, pos, svg_img)
            
            # Save
            image.convert('RGB').save(output_path, 'JPEG', quality=95)
            print(f"✓ {filename}")
            
        except Exception as e:
            print(f"✗ {filename}: {e}")

# Usage
batch_apply_svg(
    input_dir="./raw_images",
    output_dir="./watermarked_images",
    svg_path="./logos/elecfans-logo.svg",
    alpha=0.35
)
```

### Validation & Testing

```python
from tools.generate_watermarks_with_masks import (
    validate_watermark_visibility,
    load_svg_as_rgba
)
import numpy as np

def test_svg_visibility(svg_path, test_size=(100, 100)):
    """Verify SVG watermark passes visibility validation"""
    
    # Render SVG
    svg_img = load_svg_as_rgba(svg_path, *test_size)
    if svg_img is None:
        print(f"✗ Failed to render {svg_path}")
        return False
    
    # Create overlay
    overlay = np.zeros((*test_size, 4), dtype=np.uint8)
    svg_array = np.array(svg_img)
    
    # Paste SVG
    h, w = test_size
    y_pos = (h - svg_array.shape[0]) // 2
    x_pos = (w - svg_array.shape[1]) // 2
    
    overlay[y_pos:y_pos+svg_array.shape[0], 
            x_pos:x_pos+svg_array.shape[1]] = svg_array
    
    # Validate
    result = validate_watermark_visibility(overlay, w, h)
    
    print(f"SVG: {svg_path}")
    print(f"  Visible: {result['is_visible']}")
    print(f"  Coverage: {result['coverage']:.2%}")
    print(f"  Avg Alpha: {result['avg_alpha']:.2f}")
    if result['reasons']:
        for reason in result['reasons']:
            print(f"  Issue: {reason}")
    
    return result['is_visible']

# Test all SVG styles
test_svg_visibility("./logos/elecfans-logo.svg")
test_svg_visibility("./logos/elecfans-web.svg")
test_svg_visibility("./logos/WeChat.svg")
```

## Performance Profiling

```python
import time
from tools.generate_watermarks_with_masks import load_svg_as_rgba

def profile_svg_loading():
    """Profile SVG loading performance"""
    
    svg_path = "./logos/elecfans-logo.svg"
    sizes = [(100, 100), (200, 150), (300, 200)]
    
    print("SVG Loading Performance")
    print("-" * 50)
    
    for w, h in sizes:
        # First call (no cache)
        start = time.time()
        img1 = load_svg_as_rgba(svg_path, w, h)
        time1 = time.time() - start
        
        # Second call (cached)
        start = time.time()
        img2 = load_svg_as_rgba(svg_path, w, h)
        time2 = time.time() - start
        
        print(f"Size {w}×{h}:")
        print(f"  First load:  {time1*1000:.1f}ms (rendering)")
        print(f"  Second load: {time2*1000:.3f}ms (cached)")
        print(f"  Speedup:     {time1/time2:.0f}×")

profile_svg_loading()
```

## Advanced Customization

### Dynamic SVG Text

```python
def wechat_svg_with_custom_id(w, h, account_id):
    """WeChat watermark with custom account ID"""
    
    # ... SVG icon rendering code ...
    
    # Dynamic text rendering
    overlay = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    font = get_font(16)
    
    # Position text next to icon
    text_x = icon_x + icon_size + spacing
    text_y = base_y + (total_height - text_bbox[1]) // 2
    
    overlay = draw_text_with_alpha(overlay, (text_x, text_y), 
                                  account_id, font, (255, 255, 255), alpha)
    
    return overlay

# Usage with custom account IDs
for account_id in ["官方账号", "品牌合作", "媒体认证"]:
    overlay = wechat_svg_with_custom_id(1024, 768, account_id)
```

---

**Last Updated**: November 29, 2025
**Version**: 1.0 SVG Styles API
