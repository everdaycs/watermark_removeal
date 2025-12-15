"""\
微调数据生成脚本（SVG版）- 仅生成 ElecFans LOGO（elecfans-logo.svg）水印

从 `generate_svg_with_masks.py` 中抽取并最小化：
- 仅保留 elecfans_logo_svg_corner 样式
- 生成 watermarked JPG + 对应 mask PNG
- 命名使用 `_wm_` 分隔，兼容现有 `dataset.py` 的解析逻辑

输出目录结构：
  <data_dir>/images/*.jpg
  <data_dir>/masks/*_mask.png

示例：
  python generate_finetune_svg_elecfans_logo.py --num_images 50
"""

import argparse
import io
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# Optional dependency: CairoSVG
# ----------------------------------------------------------------------------
try:
    import cairosvg

    CAIROSVG_AVAILABLE = True
except ImportError:
    CAIROSVG_AVAILABLE = False


# ----------------------------------------------------------------------------
# SVG cache
# ----------------------------------------------------------------------------
_SVG_CACHE = {}


def parse_args():
    p = argparse.ArgumentParser(description="Generate finetune dataset for ElecFans LOGO (SVG)")
    p.add_argument(
        "--input_dir",
        type=str,
        default="/home/kaga/Desktop/watermaker remover/20251201/no_watermark_20251201",
        help="Input clean images directory",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default="/home/kaga/Desktop/watermaker remover/watermark_removal_project/data/finetune",
        help="Output dataset root dir. Will create images/ and masks/",
    )
    p.add_argument(
        "--num_variants",
        type=int,
        default=8,
        help="Variants per image (how many watermarked versions for each clean image)",
    )
    p.add_argument(
        "--num_images",
        type=int,
        default=0,
        help="Limit number of images (0 means all)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # ElecFans LOGO SVG tuning (copied from generate_svg_with_masks.py)
    p.add_argument("--logo_min_width", type=float, default=0.30, help="Min logo width ratio")
    p.add_argument("--logo_max_width", type=float, default=0.50, help="Max logo width ratio")
    p.add_argument("--alpha_min", type=float, default=0.80, help="Min alpha")
    p.add_argument("--alpha_max", type=float, default=0.95, help="Max alpha")
    p.add_argument("--shadow_offset_ratio", type=float, default=0.01, help="Shadow offset ratio")
    p.add_argument("--shadow_min_offset", type=int, default=1, help="Shadow min offset)")
    p.add_argument(
        "--min_render_size",
        type=int,
        default=256,
        help="Min render size for cairosvg (avoid blurry tiny raster)",
    )

    p.add_argument(
        "--enable_visibility_validation",
        action="store_true",
        help="Enable visibility validation (default off for finetune) ",
    )
    p.add_argument(
        "--min_visible_coverage",
        type=float,
        default=0.0005,
        help="Min coverage for validation (only when enabled)",
    )

    return p.parse_args()


def ensure_dirs(images_dir: str, masks_dir: str):
    Path(images_dir).mkdir(parents=True, exist_ok=True)
    Path(masks_dir).mkdir(parents=True, exist_ok=True)


def get_image_files(directory: str):
    supported = {".jpg", ".jpeg", ".png"}
    files = []
    for root, _, filenames in os.walk(directory):
        for fn in filenames:
            if Path(fn).suffix.lower() in supported:
                files.append(os.path.join(root, fn))
    files.sort()
    return files


def load_image(image_path: str) -> Image.Image | None:
    try:
        img = Image.open(image_path)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        return img
    except Exception:
        return None


def validate_watermark_visibility(overlay_array: np.ndarray, w: int, h: int, min_alpha: float, min_cov: float):
    alpha_channel = overlay_array[:, :, 3]

    visible_pixels = np.sum(alpha_channel > 0)
    total_pixels = w * h
    coverage = visible_pixels / total_pixels if total_pixels > 0 else 0

    if visible_pixels > 0:
        avg_alpha = float(np.mean(alpha_channel[alpha_channel > 0]) / 255.0)
    else:
        avg_alpha = 0.0

    return {
        "is_visible": bool((coverage >= min_cov) and (avg_alpha >= min_alpha)),
        "avg_alpha": avg_alpha,
        "coverage": coverage,
    }


def load_svg_as_rgba(svg_path: str, target_width: int | None = None, target_height: int | None = None, min_render_size: int = 256):
    if not CAIROSVG_AVAILABLE:
        return None

    cache_key = (svg_path, target_width, target_height, min_render_size)
    if cache_key in _SVG_CACHE:
        return _SVG_CACHE[cache_key].copy()

    try:
        if not os.path.exists(svg_path):
            return None

        render_width = target_width
        render_height = target_height

        # ensure minimum raster size
        if target_width is not None and target_width < min_render_size:
            render_width = min_render_size
            render_height = None

        png_bytes = io.BytesIO()
        kwargs = {"url": svg_path, "write_to": png_bytes}
        if render_width is not None:
            kwargs["output_width"] = render_width
        if render_height is not None:
            kwargs["output_height"] = render_height

        cairosvg.svg2png(**kwargs)
        png_bytes.seek(0)

        img = Image.open(png_bytes).convert("RGBA")

        if target_width is not None and render_width != target_width:
            img = img.resize((target_width, int(target_width * img.height / img.width)), Image.Resampling.LANCZOS)

        _SVG_CACHE[cache_key] = img.copy()
        return img

    except Exception:
        return None


def elecfans_logo_svg_corner(
    w: int,
    h: int,
    *,
    elecfans_logo_svg: str,
    logo_min_width: float,
    logo_max_width: float,
    alpha_min: float,
    alpha_max: float,
    shadow_offset_ratio: float,
    shadow_min_offset: int,
    min_render_size: int,
):
    """Copied from generate_svg_with_masks.py (with args passed in)."""
    if not CAIROSVG_AVAILABLE or not os.path.exists(elecfans_logo_svg):
        return None

    try:
        logo_w = int(w * random.uniform(logo_min_width, logo_max_width))

        svg_rgba = load_svg_as_rgba(elecfans_logo_svg, target_width=logo_w, min_render_size=min_render_size)
        if svg_rgba is None:
            return None

        logo_h = svg_rgba.height

        # shadow: black
        try:
            shadow_offset = max(shadow_min_offset, int(logo_w * shadow_offset_ratio))
            shadow_img = svg_rgba.copy()
            shadow_arr = np.array(shadow_img)
            shadow_arr[:, :, 0:3] = 0
            shadow_img = Image.fromarray(shadow_arr)

            # foreground: white
            fg_img = svg_rgba.copy()
            fg_arr = np.array(fg_img)
            fg_arr[:, :, 0:3] = 255
            fg_img = Image.fromarray(fg_arr)

            new_w = logo_w + shadow_offset
            new_h = logo_h + shadow_offset
            combined = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
            combined.paste(shadow_img, (shadow_offset, shadow_offset), shadow_img)
            combined.paste(fg_img, (0, 0), fg_img)
            svg_rgba = combined
        except Exception:
            pass

        # alpha
        alpha = random.uniform(alpha_min, alpha_max)
        svg_array = np.array(svg_rgba)
        mask = svg_array[:, :, 3] > 0
        svg_array[mask, 3] = (svg_array[mask, 3] * alpha).astype(np.uint8)
        svg_rgba = Image.fromarray(svg_array, "RGBA")

        overlay = Image.new("RGBA", (w, h), (255, 255, 255, 0))

        # bottom-right only
        margin = max(10, int(w * 0.015))
        pos = (max(0, w - logo_w - margin), max(0, h - logo_h - margin))

        overlay.paste(svg_rgba, pos, svg_rgba)
        return overlay

    except Exception:
        return None


def generate_variant(
    image_path: str,
    output_index: int,
    *,
    elecfans_logo_svg: str,
    args,
):
    image = load_image(image_path)
    if image is None:
        return False, None, None, None

    w, h = image.size
    stem = Path(image_path).stem
    output_name = f"{stem}_wm_{output_index + 1}"

    overlay = elecfans_logo_svg_corner(
        w,
        h,
        elecfans_logo_svg=elecfans_logo_svg,
        logo_min_width=args.logo_min_width,
        logo_max_width=args.logo_max_width,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        shadow_offset_ratio=args.shadow_offset_ratio,
        shadow_min_offset=args.shadow_min_offset,
        min_render_size=args.min_render_size,
    )

    if overlay is None:
        return False, None, None, None

    if args.enable_visibility_validation:
        overlay_array = np.array(overlay)
        vis = validate_watermark_visibility(
            overlay_array,
            w,
            h,
            min_alpha=min(args.alpha_min, args.alpha_max),
            min_cov=args.min_visible_coverage,
        )
        if not vis["is_visible"]:
            return False, None, None, None

    # composite
    watermarked = Image.alpha_composite(image, overlay).convert("RGB")

    # mask
    overlay_array = np.array(overlay)
    alpha_mask = overlay_array[:, :, 3] > 0
    mask_array = np.zeros((h, w), dtype=np.uint8)
    mask_array[alpha_mask] = 255
    mask = Image.fromarray(mask_array, mode="L")

    return True, watermarked, mask, output_name


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    elecfans_logo_svg = os.path.join(script_dir, "logos/elecfans-logo.svg")

    if not CAIROSVG_AVAILABLE:
        raise SystemExit("cairosvg 未安装：请先 pip install cairosvg")

    if not os.path.exists(elecfans_logo_svg):
        raise SystemExit(f"找不到SVG文件: {elecfans_logo_svg}")

    images_dir = os.path.join(args.data_dir, "images")
    masks_dir = os.path.join(args.data_dir, "masks")
    ensure_dirs(images_dir, masks_dir)

    image_files = get_image_files(args.input_dir)
    if args.num_images and args.num_images > 0:
        image_files = image_files[: args.num_images]

    if not image_files:
        raise SystemExit(f"未找到输入图片: {args.input_dir}")

    total_written = 0

    for i, image_path in enumerate(image_files):
        success_count = 0
        # for each image, try enough attempts to reach num_variants
        max_attempts = args.num_variants * 2
        attempts = 0

        while success_count < args.num_variants and attempts < max_attempts:
            ok, watermarked, mask, output_name = generate_variant(
                image_path,
                success_count,
                elecfans_logo_svg=elecfans_logo_svg,
                args=args,
            )
            attempts += 1
            if not ok:
                continue

            wm_path = os.path.join(images_dir, f"{output_name}.jpg")
            mask_path = os.path.join(masks_dir, f"{output_name}_mask.png")

            quality = random.randint(80, 95)
            watermarked.save(wm_path, "JPEG", quality=quality)
            mask.save(mask_path, "PNG")

            success_count += 1
            total_written += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(image_files):
            print(f"进度: {i + 1}/{len(image_files)} 已处理，累计写入 {total_written} 样本")

    print(f"完成! 共生成 {total_written} 个样本")
    print(f"images: {images_dir}")
    print(f"masks : {masks_dir}")


if __name__ == "__main__":
    main()
