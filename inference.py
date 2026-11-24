"""Inference and Grad-CAM helpers aligned to the original Kaggle notebook.
Model expects RGB images at 224x224, pixel range 0..255 (no 1/255 scaling).
"""
from __future__ import annotations
import numpy as np
from PIL import Image

CLASS_NAMES = ("black_pod_rot", "healthy", "pod_borer")
LABELS = {"black_pod_rot": "Busuk Buah", "healthy": "Buah Sehat", "pod_borer": "Penggerek Buah"}


def prepare_image(img: Image.Image):
    rgb = img.convert("RGB")
    arr = np.asarray(rgb.resize((224, 224), Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR), dtype=np.float32)
    return arr[None, ...]


def predict(model, img: Image.Image):
    batch = prepare_image(img)
    raw = np.asarray(model.predict(batch, verbose=0))[0]
    if raw.shape != (3,):
        raise ValueError(f"Output model {raw.shape}; aplikasi memerlukan 3 kelas.")
    if not np.all(np.isfinite(raw)):
        raise ValueError("Output model mengandung nilai yang tidak valid.")
    # Notebook ends in a 3-unit softmax.
    idx = int(np.argmax(raw))
    return CLASS_NAMES[idx], {k: float(v) for k, v in zip(CLASS_NAMES, raw)}


def gradcam(model, img: Image.Image):
    """Compute Grad-CAM for common notebook topology; returns heatmap [0..1].

    For a nested EfficientNet model, follow the original classifier head using
    the same layer objects with training=False. This preserves weights.
    """
    import tensorflow as tf
    x = tf.convert_to_tensor(prepare_image(img))
    backbone = next((ly for ly in model.layers if isinstance(ly, tf.keras.Model) and 'efficientnet' in ly.name.lower()), None)
    if backbone is not None:
        conv = backbone.get_layer('top_conv')
        features = tf.keras.Model(backbone.input, [conv.output, backbone.output])
        with tf.GradientTape() as tape:
            fmap, z = features(x, training=False)
            tape.watch(fmap)
            # Passing classifier head through the last convolution is essential.
            # Nested topology can have a gap between intermediate fmap and z;
            # recompute downstream backbone layers starting at top_conv.
            pos = backbone.layers.index(conv)
            z = fmap
            for layer in backbone.layers[pos+1:]:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                z = layer(z, training=False)
            base_index = model.layers.index(backbone)
            for layer in model.layers[base_index+1:]:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                z = layer(z, training=False)
            cls = tf.argmax(z[0]); score = z[:, cls]
        grads = tape.gradient(score, fmap)
    else:
        conv = model.get_layer('top_conv')
        extractor = tf.keras.Model(model.inputs, [conv.output, model.output])
        with tf.GradientTape() as tape:
            fmap, result = extractor(x, training=False)
            cls = tf.argmax(result[0]); score = result[:, cls]
        grads = tape.gradient(score, fmap)
    if grads is None:
        raise RuntimeError("Grad-CAM tidak tersedia untuk arsitektur model ini.")
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    hmap = tf.reduce_sum(fmap[0] * weights, axis=-1)
    hmap = tf.maximum(hmap, 0)
    hmap = hmap / (tf.reduce_max(hmap) + 1e-8)
    return hmap.numpy()


def overlay(img: Image.Image, heatmap: np.ndarray, alpha: float = 0.38):
    from matplotlib import colormaps
    original = img.convert('RGB')
    cmap = colormaps['turbo']
    heat = (cmap(np.clip(heatmap, 0, 1))[..., :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat).resize(original.size, Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR)
    return Image.blend(original, heat_img, alpha)
