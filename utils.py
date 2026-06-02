import os
import time
import re
import torch

from qwen_vl_utils import smart_resize
import torch.distributed as dist


PROJECT_ROOT = "/home/jiapinglin/workspace/agent/Re-Prefill"
RANK_SAVE_DIR = os.path.join("rst_tmp_rank", time.strftime('%Y%m%d_%H_%M'))
model_path_map = {
    "qwen3vl8b": "/data/jiapinglin/models/Qwen3-VL-8B-Instruct",
    "qwen3vl32b": "/root/autodl-tmp/models/Qwen3-VL-32B-Instruct",
    "maiui8b": "/data/jiapinglin/models/MAI-UI-8B",
    "guiowl8b": "/data/jiapinglin/models/GUI-Owl-1.5-8B"
}
json_file_dir_map = {
    "sspro": "/data/jiapinglin/datasets/agent/screenspot-pro/annotations",
    "osworld": "/data/jiapinglin/datasets/agent/OSWorld-G/benchmark/classification_result.json",
    "mmbench": "/data/jiapinglin/datasets/agent/MMBench-GUI/L2_annotations.json",
    "ssv2": "/data/jiapinglin/datasets/agent/screenspot-v2"
}
base_image_dir_map = {
    "sspro": "/data/jiapinglin/datasets/agent/screenspot-pro/images",
    "osworld": "/data/jiapinglin/datasets/agent/OSWorld-G/benchmark/images",
    "mmbench": "/data/jiapinglin/datasets/agent/MMBench-GUI/offline_images",
    "ssv2": "/data/jiapinglin/datasets/agent/screenspot-v2/images",
    "uivision": "/data/jiapinglin/datasets/agent/ui-vision/images"
}


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
        return rank, world_size, gpu
    else:
        print("Not using distributed mode")
        return 0, 1, 0


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def locate_image_start_end(image_mask):
    assert image_mask.dim() == 2
    N, M = image_mask.shape

    # 第一个 True
    first_idx = image_mask.float().argmax(dim=1)

    # 最后一个 True（反转后再 argmax）
    last_idx = M - 1 - image_mask.flip(dims=[1]).float().argmax(dim=1)

    return torch.stack([first_idx, last_idx], dim=1).cpu()


def extract_coordinates(raw_string):
    """从模型输出中提取坐标"""
    try:
        matches = re.findall(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)", raw_string)
        return [tuple(map(int, match)) for match in matches][0]
    except:
        return (0, 0)


def _is_point_in_rectangle(point, bbox):
    """检查点是否在矩形边界框内"""
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _is_point_in_polygon(point, polygon):
    """检查点是否在多边形内"""
    x, y = point
    n = len(polygon) // 2
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i * 2], polygon[i * 2 + 1]
        xj, yj = polygon[j * 2], polygon[j * 2 + 1]

        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i

    return inside


def is_point_in_target(point, box_coordinates, box_type="bbox"):
    if box_type == "bbox":
        # bbox格式: [x1, y1, x2, y2]
        return _is_point_in_rectangle(point, box_coordinates)

    elif box_type == "polygon":
        # polygon格式: [x1, y1, x2, y2, ...]
        return _is_point_in_polygon(point, box_coordinates)

    elif box_type == "refusal":
        # refusal格式: 所有点坐标应为负值
        return all(coord < 0 for coord in point)

    else:
        print(f"Warning: Unknown box_type: {box_type}")
        return False


def resize_image(image, processor, resize=False):
    if resize:
        zoomed_img = image.resize((image.width * 2, image.height * 2))
    else:
        zoomed_img = image
    resized_height, resized_width = smart_resize(
        zoomed_img.height,
        zoomed_img.width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=processor.image_processor.max_pixels,
    )
    resized_image = zoomed_img.resize((resized_width, resized_height))

    scale_x = zoomed_img.width / resized_width
    scale_y = zoomed_img.height / resized_height
    return resized_image, resized_height, resized_width, scale_x, scale_y


def custom_collate_fn(batch):
    return batch


def select_key_visual_tokens(vis_attention, high_thresh=0.9, high_ratio_low=0.3):
    if len(vis_attention) > 36:
        vis_attention = vis_attention[32:43]  # for 32B
    else:
        vis_attention = vis_attention[18:24]
    layer_num, token_num = vis_attention.shape

    mean_scores = vis_attention.mean(dim=0)

    high_threshold = torch.quantile(vis_attention, high_thresh)
    high_steps = (vis_attention > high_threshold).sum(dim=0)
    ratio = high_steps.float() / layer_num

    mean_low = torch.quantile(mean_scores, 0.6)

    mask = (
            # (mean_scores > mean_low) &
            (ratio >= high_ratio_low)
    )

    selected_indices = torch.where(mask)[0]
    return selected_indices
