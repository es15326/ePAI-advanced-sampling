'''
python -W ignore fig_interannotator_agreement.py --root_r1 /Users/zongwei.zhou/ASU\ Dropbox/Zongwei\ Zhou/Apps/Overleaf/NeurIPS\ 2025\ PanTS/src/fig_interannotator_agreement/interannotator_agreement/r1 --root_r2 /Users/zongwei.zhou/ASU\ Dropbox/Zongwei\ Zhou/Apps/Overleaf/NeurIPS\ 2025\ PanTS/src/fig_interannotator_agreement/interannotator_agreement/r2
'''

import os
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

def normalize_ct(ct_slice, ct_min, ct_max):
    ct_slice = np.clip(ct_slice, ct_min, ct_max)
    ct_slice = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-5)
    return ct_slice

from scipy.ndimage import binary_dilation
from skimage.morphology import disk

def overlay_boundary(ct_slice, seg, boundary_color, alpha, linewidth):
    seg = seg.astype(np.uint8)
    boundary = find_boundaries(seg, mode='outer')

    if linewidth > 1:
        boundary = binary_dilation(boundary, structure=disk(linewidth))

    overlay = np.stack([ct_slice]*3, axis=-1)

    # Fill region
    for i in range(3):
        overlay[..., i] = np.where(seg > 0,
                                   overlay[..., i] * (1 - alpha) + boundary_color[i] * alpha,
                                   overlay[..., i])

    # Draw boundary
    for i in range(3):
        overlay[..., i][boundary] = boundary_color[i]

    return overlay

def crop_center(img, mask, zoom=2.5):
    indices = np.argwhere(mask > 0)
    if indices.size == 0:
        return img  # no crop if no mask
    center = indices.mean(axis=0).astype(int)
    h, w = img.shape[:2]
    box_size = int(min(h, w) / zoom / 2)
    cx, cy = center
    x1, x2 = max(cx - box_size, 0), min(cx + box_size, h)
    y1, y2 = max(cy - box_size, 0), min(cy + box_size, w)
    return img[x1:x2, y1:y2]

def save_png(img, output_path, suffix, grayscale=False):
    img = np.rot90(img)  # transpose visually
    img = np.fliplr(img)  # left-right flip
    if grayscale:
        plt.imsave(output_path.replace(".png", f"_{suffix}.png"), img, cmap='gray')
    else:
        plt.imsave(output_path.replace(".png", f"_{suffix}.png"), img)

def process_case(case_id, root_dir_r1, root_dir_r2, output_dir, zoom_factor, ct_min, ct_max, boundary_color, linewidth, alpha):
    ct_path = os.path.join(root_dir_r1, case_id, "ct.nii.gz")
    seg_r1_path = os.path.join(root_dir_r1, case_id, "segmentations", "pancreatic_lesion.nii.gz")
    seg_r2_path = os.path.join(root_dir_r2, case_id, "segmentations", "pancreatic_lesion.nii.gz")

    ct_img = nib.load(ct_path).get_fdata()
    seg_r1 = nib.load(seg_r1_path).get_fdata()
    seg_r2 = nib.load(seg_r2_path).get_fdata()

    ct_img = np.transpose(ct_img, (2, 0, 1))
    seg_r1 = np.transpose(seg_r1, (2, 0, 1))
    seg_r2 = np.transpose(seg_r2, (2, 0, 1))

    z_indices = np.where(seg_r1 + seg_r2 > 0)[0]
    if len(z_indices) == 0:
        return
    z = z_indices[len(z_indices) // 2]

    ct_slice = normalize_ct(ct_img[z], ct_min, ct_max)
    overlaid_r1 = overlay_boundary(ct_slice, seg_r1[z], boundary_color, alpha, linewidth)
    overlaid_r2 = overlay_boundary(ct_slice, seg_r2[z], boundary_color, alpha, linewidth)

    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, f"{case_id}.png")

    # Crop zoomed region
    cropped_ct = crop_center(ct_slice, seg_r1[z] + seg_r2[z], zoom=zoom_factor)
    cropped_r1 = crop_center(overlaid_r1, seg_r1[z], zoom=zoom_factor)
    cropped_r2 = crop_center(overlaid_r2, seg_r2[z], zoom=zoom_factor)

    # Save individual PNGs
    save_png(cropped_ct, base_path, "ct", grayscale=True)
    save_png(cropped_r1, base_path, "r1")
    save_png(cropped_r2, base_path, "r2")

def main():
    parser = argparse.ArgumentParser(description="Visualize inter-annotator tumor boundaries.")
    parser.add_argument('--root_r1', type=str, required=True, help='Path to Annotator 1 directory')
    parser.add_argument('--root_r2', type=str, required=True, help='Path to Annotator 2 directory')
    parser.add_argument('--output_path', type=str, default='interannotator_agreement', help='Directory to save PNGs')
    parser.add_argument('--zoom', type=float, default=3.5, help='Zoom-in factor (default: 2.5x)')
    parser.add_argument('--ct_min', type=float, default=-100, help='CT window minimum value (default: -100)')
    parser.add_argument('--ct_max', type=float, default=200, help='CT window maximum value (default: 200)')
    parser.add_argument('--boundary_color', type=float, nargs=3, default=[1.0, 0.0, 0.0], help='Boundary color as RGB float values (default: red [1, 0, 0])')
    parser.add_argument('--linewidth', type=int, default=1.5, help='Line width for boundary overlay (default: 3)')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha blending for filled mask color (default: 0.2)')

    args = parser.parse_args()

    case_ids = [cid for cid in os.listdir(args.root_r1)
                if not cid.startswith('.') and os.path.isdir(os.path.join(args.root_r1, cid))]

    for cid in tqdm(case_ids, total=len(case_ids), ncols=80):
        process_case(cid, args.root_r1, args.root_r2, args.output_path, args.zoom,
                     args.ct_min, args.ct_max, args.boundary_color, args.linewidth, args.alpha)

if __name__ == '__main__':
    main()
