#!/usr/bin/env python3
"""\
微调模型推理脚本（ElecFans finetune）

特点：
- 加载 `checkpoints_finetune/finetune_elecfans_best.pth`
- 支持单张图片或文件夹批量推理
- 支持普通推理（整体resize到512）与分块推理（保持高分辨率细节）

用法：
  python finetune_inference.py --input <image_or_dir>
  python finetune_inference.py --input <dir> --tiled --tile-size 512 --overlap 64

输出：默认写入 `results_finetune/`。
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.optimized_unet import AttentionResUNet


def parse_args():
    p = argparse.ArgumentParser(description="Finetune inference (ElecFans)")
    p.add_argument(
        "--input",
        "-i",
        default="/home/kaga/Desktop/watermaker remover/20251201/transparent_watermark_20251201",
        help="Input image path or directory (default: transparent watermark folder)",
    )
    p.add_argument(
        "--output",
        "-o",
        default="/home/kaga/Desktop/watermaker remover/watermark_removal_project/results_finetune",
        help="Output directory (default: watermark_removal_project/results_finetune)",
    )
    p.add_argument(
        "--checkpoint",
        "-c",
        default=None,
        help="Checkpoint path (default: checkpoints_finetune/finetune_elecfans_best.pth)",
    )

    # Preprocess size (for non-tiled)
    p.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Inference size for non-tiled mode (square). Must match training size (default 512)",
    )

    # Tiled inference
    p.add_argument(
        "--tiled",
        action="store_true",
        help="Enable tiled inference (keep original resolution)",
    )
    p.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tile size for tiled inference (default 512)",
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Overlap pixels for tiled inference (default 64)",
    )

    return p.parse_args()


def _to_tensor_bgr(image_bgr: np.ndarray) -> torch.Tensor:
    # BGR -> RGB
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # normalize to [-1, 1]
    x = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    # HWC -> CHW
    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    return t


def _to_bgr_uint8(output_tensor: torch.Tensor) -> np.ndarray:
    y = output_tensor.squeeze(0).detach().cpu().numpy()
    y = ((y * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    y = y.transpose(1, 2, 0)  # HWC
    y = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
    return y


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = AttentionResUNet(n_channels=3, n_classes=3)

    ckpt = torch.load(checkpoint_path, map_location=device)

    # support multiple checkpoint formats
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # remove possible module. prefix
    new_state = {}
    for k, v in state_dict.items():
        new_state[k.replace("module.", "")] = v

    model.load_state_dict(new_state, strict=True)
    model = model.to(device)
    model.eval()

    return model


class TiledInferencer:
    def __init__(self, model, device: torch.device, tile_size: int = 512, overlap: int = 64):
        self.model = model
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap

    def _create_weight_mask(self) -> np.ndarray:
        if self.overlap <= 0:
            return np.ones((self.tile_size, self.tile_size, 1), dtype=np.float32)

        ramp = np.linspace(0, 1, self.overlap, dtype=np.float32)
        w = np.ones((self.tile_size, self.tile_size), dtype=np.float32)
        w[: self.overlap, :] *= ramp[:, None]
        w[-self.overlap :, :] *= ramp[::-1][:, None]
        w[:, : self.overlap] *= ramp[None, :]
        w[:, -self.overlap :] *= ramp[::-1][None, :]
        return w[:, :, None]

    def process(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]

        if h <= self.tile_size and w <= self.tile_size:
            padded = cv2.copyMakeBorder(
                image_bgr,
                0,
                self.tile_size - h,
                0,
                self.tile_size - w,
                cv2.BORDER_REFLECT_101,
            )
            with torch.no_grad():
                t = _to_tensor_bgr(padded).to(self.device)
                out = self.model(t)
            out_bgr = _to_bgr_uint8(out)
            return out_bgr[:h, :w]

        n_tiles_h = max(1, int(np.ceil((h - self.overlap) / self.stride)))
        n_tiles_w = max(1, int(np.ceil((w - self.overlap) / self.stride)))

        padded_h = self.stride * n_tiles_h + self.overlap
        padded_w = self.stride * n_tiles_w + self.overlap

        pad_h = padded_h - h
        pad_w = padded_w - w

        padded = cv2.copyMakeBorder(image_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

        out_sum = np.zeros((padded_h, padded_w, 3), dtype=np.float32)
        w_sum = np.zeros((padded_h, padded_w, 1), dtype=np.float32)
        weight = self._create_weight_mask()

        self.model.eval()
        with torch.no_grad():
            for i in range(n_tiles_h):
                for j in range(n_tiles_w):
                    y0 = i * self.stride
                    x0 = j * self.stride
                    y1 = y0 + self.tile_size
                    x1 = x0 + self.tile_size

                    tile = padded[y0:y1, x0:x1]
                    if tile.shape[0] != self.tile_size or tile.shape[1] != self.tile_size:
                        tile = cv2.resize(tile, (self.tile_size, self.tile_size))

                    t = _to_tensor_bgr(tile).to(self.device)
                    out = self.model(t)
                    out_tile = _to_bgr_uint8(out).astype(np.float32)

                    out_sum[y0:y1, x0:x1] += out_tile * weight
                    w_sum[y0:y1, x0:x1] += weight

        w_sum = np.maximum(w_sum, 1e-8)
        out = out_sum / w_sum
        out = out[:h, :w]
        return out.clip(0, 255).astype(np.uint8)


def iter_input_paths(input_path: str):
    p = Path(input_path)
    if p.is_file():
        return [p]

    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = [x for x in p.iterdir() if x.suffix.lower() in exts]
        files.sort()
        return files

    raise FileNotFoundError(f"Input not found: {input_path}")


def main():
    args = parse_args()
    config = Config()

    if args.output is None:
        args.output = os.path.join(config.PROJECT_ROOT, "results_finetune")

    if args.checkpoint is None:
        args.checkpoint = os.path.join(config.PROJECT_ROOT, "checkpoints_finetune", "finetune_elecfans_best.pth")

    os.makedirs(args.output, exist_ok=True)

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    model = load_model(args.checkpoint, device)

    tiled = None
    if args.tiled:
        tiled = TiledInferencer(model, device, tile_size=args.tile_size, overlap=args.overlap)

    paths = iter_input_paths(args.input)
    print(f"Found {len(paths)} images")

    for p in tqdm(paths, desc="Inference"):
        img = cv2.imread(str(p))
        if img is None:
            continue

        if args.tiled:
            out = tiled.process(img)
        else:
            # resize to training size
            original_size = (img.shape[1], img.shape[0])
            resized = cv2.resize(img, (args.image_size, args.image_size))
            with torch.no_grad():
                t = _to_tensor_bgr(resized).to(device)
                y = model(t)
            out_small = _to_bgr_uint8(y)
            out = cv2.resize(out_small, original_size)

        out_path = os.path.join(args.output, p.name)
        if out_path.lower().endswith((".jpg", ".jpeg")):
            cv2.imwrite(out_path, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(out_path, out)

    print(f"Done. Results saved to: {args.output}")


if __name__ == "__main__":
    main()
