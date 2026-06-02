import os
from PIL import Image

from utils import (
    cleanup_distributed,
    is_point_in_target,
    RANK_SAVE_DIR
)
from rp_qwen3vl import parse_args, DistributedRunner, GUIGroundingEngine


def create_sub_region(prediction, width, height, subimage_width=1288, subimage_height=728):
    sub_w = min(subimage_width, width)
    sub_h = min(subimage_height, height)
    center_x, center_y = prediction

    left = max(0, center_x - sub_w / 2)
    top = max(0, center_y - sub_h / 2)
    right = min(width, left + sub_w)
    bottom = min(height, top + sub_h)

    if right - left < sub_w:
        if left == 0:
            right = sub_w
        else:
            left = right - sub_w
    if bottom - top < sub_h:
        if top == 0:
            bottom = sub_h
        else:
            top = bottom - sub_h

    region = (int(left), int(top), int(right), int(bottom))
    sub_region = {'region': region, 'center': (center_x, center_y)}

    return sub_region


class GUIGroundingEngineWZoomin(GUIGroundingEngine):
    def __init__(self, args, device):
        super(GUIGroundingEngineWZoomin, self).__init__(args, device)
        self.version = "wZoomin"

    def get_sub_region(self, image, instruction, subimage_width=1288, subimage_height=728):
        width, height = image.size
        coord, output_text = self.process_subimage(image, instruction)
        selected_region = create_sub_region(coord, width, height,
                                             subimage_width=subimage_width, subimage_height=subimage_height)

        return selected_region

    def process_orig_image(self, json_data):
        try:
            img_path = os.path.join(self.args.base_image_dir, json_data["img_filename"])
            if not os.path.exists(img_path):
                return None

            image = Image.open(img_path)
            width, height = image.size
            bbox = json_data["bbox"]
            box_type = json_data["box_type"] if "box_type" in json_data else "bbox"
            instruction = json_data["instruction"]

            print(f"Processing: {json_data['img_filename']}")

            result = {
                "filename": json_data["img_filename"],
                "instruction": instruction,
                "target_bbox": bbox,
                "final_prediction": None,
            }
            if self.benchmark in ["sspro", "ssv2", "mmbench"]:
                result.update({
                    "id": json_data["id"],
                    "application": json_data.get("application", "unknown"),
                    "platform": json_data.get("platform", "unknown"),
                    "ui_type": json_data.get("ui_type", "unknown"),
                    "group": json_data.get("group", "unknown")
                })

            selected_region = self.get_sub_region(image, instruction)
            sub_region = selected_region["region"]
            left, top, right, bottom = sub_region
            subimage = image.crop(sub_region)

            coord, output_text = self.process_subimage(subimage, instruction, left, top, resize=True)
            pred_x, pred_y = coord
            in_bbox = is_point_in_target((pred_x, pred_y), bbox, box_type=box_type)

            prediction = {
                "point": (pred_x, pred_y),
                "in_bbox": in_bbox,
                "output": output_text,
                "region": sub_region
            }
            result["final_prediction"] = prediction

            print(
                f"Final prediction: {result['final_prediction']['point']} in bbox: {result['final_prediction']['in_bbox']}")
            print("=" * 50)

            return result

        except Exception as e:
            print(f"Error processing {json_data.get('img_filename', 'unknown')}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def main():
    args = parse_args()
    runner = DistributedRunner(args, GUIGroundingEngineWZoomin, rank_save_dir=RANK_SAVE_DIR)
    runner.run()
    cleanup_distributed()


if __name__ == "__main__":
    main()
