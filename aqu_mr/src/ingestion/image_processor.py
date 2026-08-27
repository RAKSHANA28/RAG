"""
Image and Visual Feature Processing for AQU-MR
Extracts visual features, thumbnails, and aligns visual regions with OCR context.
"""

from typing import List, Dict, Any, Optional
import os
import numpy as np
from PIL import Image
from ..utils.schemas import ImageItem, BoundingBox


class ImageProcessor:
    """Processes document images and computes visual descriptors with OCR alignment."""

    def __init__(self, thumbnail_size=(512, 512)):
        self.thumbnail_size = thumbnail_size
        self._clip_model = None
        self._clip_preprocess = None
        self._init_clip()

    def _init_clip(self):
        """Attempt to load CLIP or sentence-transformers vision model if available."""
        try:
            import torch
            import clip
            model, preprocess = clip.load("ViT-B/32", device="cpu")
            self._clip_model = model
            self._clip_preprocess = preprocess
        except Exception:
            self._clip_model = None

    def compute_visual_embedding(self, image: Image.Image) -> List[float]:
        """
        Computes normalized visual embedding.
        Uses CLIP if available, otherwise high-dimensional spatial-color layout descriptor.
        """
        if self._clip_model and self._clip_preprocess:
            try:
                import torch
                img_tensor = self._clip_preprocess(image).unsqueeze(0)
                with torch.no_grad():
                    features = self._clip_model.encode_image(img_tensor)
                    features = features / features.norm(dim=-1, keepdim=True)
                    return features.squeeze(0).cpu().numpy().tolist()
            except Exception:
                pass

        # Robust pure-Python / numpy spatial layout descriptor (32 bins: color, luminance & spatial quad density)
        return self._compute_spatial_descriptor(image)

    def _compute_spatial_descriptor(self, image: Image.Image) -> List[float]:
        """Computes a 64-dimensional spatial layout & color histogram feature vector."""
        img_rgb = image.convert('RGB').resize((64, 64))
        arr = np.array(img_rgb, dtype=np.float32) / 255.0  # shape: (64, 64, 3)

        # 1. Color channel histograms (8 bins per channel = 24 dims)
        r_hist, _ = np.histogram(arr[:, :, 0], bins=8, range=(0.0, 1.0), density=True)
        g_hist, _ = np.histogram(arr[:, :, 1], bins=8, range=(0.0, 1.0), density=True)
        b_hist, _ = np.histogram(arr[:, :, 2], bins=8, range=(0.0, 1.0), density=True)

        # 2. Quadrant mean luminance & variance (4 quadrants * 4 metrics = 16 dims)
        h, w, _ = arr.shape
        quads = [
            arr[:h//2, :w//2],
            arr[:h//2, w//2:],
            arr[h//2:, :w//2],
            arr[h//2:, w//2:]
        ]
        quad_features = []
        for q in quads:
            lum = 0.299 * q[:, :, 0] + 0.587 * q[:, :, 1] + 0.114 * q[:, :, 2]
            quad_features.extend([
                float(np.mean(lum)),
                float(np.std(lum)),
                float(np.mean(q[:, :, 0])),
                float(np.mean(q[:, :, 2]))
            ])

        # 3. Horizontal and vertical edge projections (24 dims)
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        h_proj = np.mean(gray, axis=1) # 64 -> pool to 12
        v_proj = np.mean(gray, axis=0) # 64 -> pool to 12
        h_pooled = [float(np.mean(h_proj[k*5:(k+1)*5])) for k in range(12)]
        v_pooled = [float(np.mean(v_proj[k*5:(k+1)*5])) for k in range(12)]

        combined = np.concatenate([r_hist, g_hist, b_hist, quad_features, h_pooled, v_pooled])
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        return combined.tolist()

    def process_page_image(
        self,
        doc_id: str,
        page_num: int,
        image: Image.Image,
        save_dir: str,
        ocr_context: str = ""
    ) -> ImageItem:
        """
        Saves page image / thumbnail and constructs an ImageItem with visual features & OCR context.
        """
        os.makedirs(save_dir, exist_ok=True)
        img_filename = f"{doc_id}_page_{page_num}.png"
        img_path = os.path.join(save_dir, img_filename)
        
        # Save image
        image.save(img_path, format="PNG")

        # Extract visual embedding
        features = self.compute_visual_embedding(image)
        w, h = image.size

        return ImageItem(
            image_id=f"{doc_id}_p{page_num}_img",
            doc_id=doc_id,
            page_num=page_num,
            image_path=img_path,
            caption=f"Page {page_num} Visual Layout",
            visual_features=features,
            ocr_context=ocr_context,
            bbox=BoundingBox(x0=0, y0=0, x1=w, y1=h),
            metadata={"width": w, "height": h}
        )
