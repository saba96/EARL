import torch
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import re
import io
from PIL import Image

def pad_tensor_list(tensor_list, max_length, pad_value=0):
    # right
    # return torch.stack([torch.cat([t, torch.full((max_length - t.shape[0],), pad_value, dtype=t.dtype)]) for t in tensor_list])
    # left padding
    return torch.stack([
        torch.cat([torch.full((max_length - t.shape[0],), pad_value, dtype=t.dtype, device=t.device), t])  # Left-padding
        for t in tensor_list
    ])

# def visualize_bboxes_and_keypoints(bboxes, keypoints_dict, image, edit_id, save_dir, prefix='', show_skeleton=True):
def visualize_bboxes_and_keypoints(bboxes, keypoints_dict, image):
    """
    Visualize bounding boxes and keypoints on an image.
    
    Parameters:
        bboxes (dict): Bounding boxes with labels {label: [x_min, y_min, x_max, y_max]}.
        keypoints_dict (dict): Keypoints {person_id: {point_name: (x, y)}}.
        image (PIL.Image.Image): Image to visualize on.
        edit_id (str): Identifier for the visualization.
        save_dir (str): Directory to save the visualization.
        prefix (str): Optional prefix for the saved file name.
        show_skeleton (bool): Whether to connect keypoints with skeleton connections.
    """
    w, h = image.size
    
    fig, ax = plt.subplots()
    ax.imshow(image)

    # Plot bounding boxes
    for label, bbox in bboxes.items():
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.text(x_min, y_min - 5, label, color='blue', fontsize=5, weight='bold', va='bottom')
        ax.add_patch(rect)

    
    for (x, y) in keypoints_dict.values():
        ax.add_patch(plt.Circle((x, y), radius=3, color="red"))
                
    # # Optionally draw skeleton
    # if show_skeleton:
    #     for part_a, part_b in SKELETON_CONNECTIONS:
    #         if part_a in kp_dict and part_b in kp_dict:
    #             x1, y1 = kp_dict[part_a]
    #             x2, y2 = kp_dict[part_b]
    #             ax.plot([x1, x2], [y1, y2], color="green", linewidth=2)

    ###########################
    # Save the result
    # visualized_output = f'{save_dir}/{edit_id}_{prefix}.png'
    # plt.savefig(visualized_output)
    # plt.close()
    # return visualized_output
    ###########################

    buf = io.BytesIO() 
    fig.savefig(buf) 
    buf.seek(0) 
    img = Image.open(buf) 
    plt.close(fig)
    return img

def extract_bboxes_and_keypoints(text: str) -> Tuple[Dict[str, List[int]], Dict[str, Tuple[int, int]]]:
    bbox_pattern = re.compile(
        r'[\[\(]\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*[\)\]]'
    )
    
    bboxes = {}
    for i, match in enumerate(bbox_pattern.findall(text), start=1):
        bboxes[f"bbox {i}"] = [int(n) for n in match]

    keypoint_pattern = re.compile(r"[\[\(]\s*(-?\d+)\s*,\s*(-?\d+)\s*[\)\]]")
    
    keypoints = {}
    for i, match in enumerate(keypoint_pattern.findall(text), start=1):
        if len(match) == 2:
            keypoints[f"keypoint {i}"] = (int(match[0]), int(match[1]))

    return bboxes, keypoints

def plot_overview(original, edited, edit_instruction):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original)
    ax[0].set_title("Original Image", fontsize=10, fontweight='bold', color='blue')
    ax[0].axis("off")
    ax[1].imshow(edited)
    ax[1].set_title("Edited Image", fontsize=10, fontweight='bold', color='green')
    ax[1].axis("off")
    fig.suptitle(edit_instruction, fontsize=14, fontweight='bold', ha='center', y=0.92, color='black', wrap=True)

    buf = io.BytesIO() 
    fig.savefig(buf) 
    buf.seek(0) 
    img = Image.open(buf) 
    plt.close(fig)
    return img 

def smart_resize(image, image_area: int = 720 * 720):
    w, h = image.size
    current_area = h * w
    target_ratio = (image_area / current_area) ** 0.5

    th = int(round(h * target_ratio))
    tw = int(round(w * target_ratio))

    image = image.resize((tw, th))
    return image


def update_bbox_in_text(img, text):
    """Extract bbox from text and update it based on image resize."""
    patterns = [
        r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
        r'\[(\d+),\s*(\d+)\],\s*\[(\d+),\s*(\d+)\]',
        r'\(\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)\)'
    ]
    
    # Find and extract bbox
    for pattern in patterns:
        if match := re.search(pattern, text):
            # Get scale factors from image resize
            img_resized = smart_resize(img.copy(), image_area=256 * 256)
            w_scale = img_resized.size[0] / img.size[0]
            h_scale = img_resized.size[1] / img.size[1]
            
            # Scale coordinates
            x1, y1, x2, y2 = map(int, match.groups())
            scaled_bbox = f'[[{int(x1*w_scale)}, {int(y1*h_scale)}], [{int(x2*w_scale)}, {int(y2*h_scale)}]]'
            
            # Update text
            return re.sub(pattern, scaled_bbox, text), img_resized
            
    return text, img
