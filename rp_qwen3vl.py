import json
import os
import time
import torch
import math
import argparse
from PIL import Image
from qwen_vl_utils import process_vision_info, smart_resize
from transformers import AutoProcessor

from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import torch.distributed as dist

from rp_qwen3vl_osworldg_utils import load_dataset as load_dataset_osworldg
from rp_qwen3vl_osworldg_utils import calculate_detailed_statistics as calculate_detailed_statistics_osworldg
from utils import (
    RANK_SAVE_DIR,
    PROJECT_ROOT,
    base_image_dir_map,
    json_file_dir_map,
    model_path_map,
    setup_distributed,
    cleanup_distributed,
    extract_coordinates,
    is_point_in_target,
    resize_image,
    custom_collate_fn,
)


class ScreenSpotDataset(Dataset):
    def __init__(self, json_data_list, base_image_dir):
        self.json_data_list = json_data_list
        self.base_image_dir = base_image_dir

    def __len__(self):
        return len(self.json_data_list)

    def __getitem__(self, idx):
        return self.json_data_list[idx]


class GUIGroundingEngine:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.benchmark = args.benchmark
        self.model_name = args.model_path.split('/')[-1]

        self.model, self.processor = self.load_model()
        self.dataset = self.load_dataset()

        self.version = "woZoomin"
        self.SYSTEM_PROMPT = '''
You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {height} and width {width}. For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)
'''.strip()

    def load_model(self):
        model_path = self.args.model_path
        from qwen3_vl import Qwen3VLConfig, Qwen3VLForConditionalGeneration
        config = Qwen3VLConfig.from_pretrained(model_path)
        setattr(config.text_config, "continuity_layer", self.args.continuity_layer)

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=f"cuda:{self.device}"
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=3136,
            max_pixels=4096 * 2160
        )

        return model, processor

    def load_dataset(self):
        json_file_dir = self.args.json_file_dir
        json_data_list = []

        if self.benchmark in ["sspro", "ssv2"]:
            json_file_paths = [f for f in os.listdir(json_file_dir) if f.endswith(('.json', '.jsonl'))]
            for json_file_path in json_file_paths:
                full_path = os.path.join(json_file_dir, json_file_path)
                with open(full_path, 'r') as f:
                    if json_file_path.endswith('.jsonl'):
                        json_data_list.extend([json.loads(line) for line in f])
                    else:
                        json_data_list.extend(json.load(f))
            if self.benchmark == "sspro":
                for jd in json_data_list:
                    jd["ui_type"] = jd["group"] + "_" + jd["ui_type"]
            elif self.benchmark == "ssv2":
                for j_idx, jd in enumerate(json_data_list):
                    ui_type = jd["img_filename"].split("_")[0] + "_"
                    jd["id"] = ui_type + str(j_idx)
                    jd["ui_type"] = ui_type + jd["data_type"]
                    jd["bbox"][2] += jd["bbox"][0]
                    jd["bbox"][3] += jd["bbox"][1]

        elif self.benchmark == "uivision":
            json_file = json_file_dir
            with open(json_file, 'r') as f:
                json_data_list.extend(json.load(f))
            for jd in json_data_list:
                if "image_path" in jd:
                    jd["img_filename"] = jd.pop("image_path")
                if "prompt_to_evaluate" in jd:
                    jd["instruction"] = jd.pop("prompt_to_evaluate")

        elif self.benchmark == "mmbench":
            json_file = json_file_dir
            with open(json_file, 'r') as f:
                json_data_list.extend(json.load(f))
            for jd in json_data_list:
                jd["id"] = str(jd["index"])
                jd["ui_type"] = jd["platform"] + "_" + jd["grounding_type"]
                if "image_path" in jd:
                    jd["img_filename"] = os.path.join(jd["platform"], jd.pop("image_path"))
                    jd["bbox"][0] = jd["bbox"][0] * jd["image_size"][0]
                    jd["bbox"][1] = jd["bbox"][1] * jd["image_size"][1]
                    jd["bbox"][2] = jd["bbox"][2] * jd["image_size"][0]
                    jd["bbox"][3] = jd["bbox"][3] * jd["image_size"][1]

        dataset = ScreenSpotDataset(json_data_list, self.args.base_image_dir)
        return dataset

    def prepare_inputs(self, instruction, image, img_height, img_width):
        system_message = {
            "role": "system",
            "content": self.SYSTEM_PROMPT.format(height=img_height, width=img_width)
        }

        user_message = {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction}
            ]
        }

        image_inputs, video_inputs = process_vision_info([system_message, user_message])
        text = self.processor.apply_chat_template([system_message, user_message], tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)

        return inputs

    def parse_coordinates_from_output(self, output_text, sub_img_width, sub_img_height, offset_x=0, offset_y=0):
        pred_x, pred_y = extract_coordinates(output_text)
        final_x = int(pred_x / 1000 * sub_img_width + offset_x)
        final_y = int(pred_y / 1000 * sub_img_height + offset_y)
        return final_x, final_y

    def process_subimage(self, subimage, instruction, offset_x=0, offset_y=0, resize=False):
        if hasattr(self.model, 'module'):
            original_model = self.model.module
        else:
            original_model = self.model

        resized_subimage, resized_height, resized_width, scale_x, scale_y = resize_image(subimage, self.processor,
                                                                                         resize=resize)

        inputs = self.prepare_inputs(instruction, resized_subimage, resized_height, resized_width)

        with torch.no_grad():
            generation_kwargs = {"custom_generate": PROJECT_ROOT}

            outputs = original_model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                              temperature=0.0, use_cache=True, pad_token_id=151645,
                                              **generation_kwargs)
            output_ids = outputs

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids,
                                                  skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        final_x, final_y = self.parse_coordinates_from_output(output_text,
                                                              subimage.width, subimage.height,
                                                              offset_x, offset_y)

        return (final_x, final_y), output_text

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

            pred_coord, pred_output = self.process_subimage(image, instruction)
            pred_x, pred_y = pred_coord
            pred_in_bbox = is_point_in_target((pred_x, pred_y), bbox, box_type=box_type)

            prediction = {
                "point": (pred_x, pred_y),
                "in_bbox": pred_in_bbox,
                "output": pred_output,
                "region": (0, 0, width, height),
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

    def run(self):
        pass

    def calculate_detailed_statistics(self, results):
        total = len(results)

        final_correct = sum(1 for r in results if r["final_prediction"] and r["final_prediction"]["in_bbox"])
        final_accuracy = final_correct / total if total > 0 else 0

        rst_stats = {
            "summary": {
                "total": int(total),
                "final_correct": int(final_correct),
                "final_accuracy": float(final_accuracy),
            }
        }
        print("\n" + "=" * 80)
        print("FINAL RESULTS:")
        print("=" * 80)
        print(f"Total images processed: {total}")
        print(f"Final prediction accuracy: {final_accuracy:.4f} ({final_correct}/{total})")

        if self.benchmark in ["sspro", "ssv2", "mmbench"]:
            ui_type_stats = {}
            for result in results:
                ui_type = result.get("ui_type", "unknown")
                if ui_type not in ui_type_stats:
                    ui_type_stats[ui_type] = {
                        "count": 0,
                        "final_correct": 0,
                    }

                ui_type_stats[ui_type]["count"] += 1
                if result["final_prediction"] and result["final_prediction"]["in_bbox"]:
                    ui_type_stats[ui_type]["final_correct"] += 1

            for stats_dict in [ui_type_stats]:
                for key in stats_dict:
                    stats = stats_dict[key]
                    count = stats["count"]
                    if count > 0:
                        stats["final_accuracy"] = stats["final_correct"] / count

            rst_stats["by_ui_type"] = ui_type_stats

            print("\n" + "=" * 40)
            print("ACCURACY BY UI TYPE:")
            print("=" * 40)
            for ui_type, ui_stat in sorted(ui_type_stats.items()):
                print(f"{ui_type}:")
                print(f"  Count: {ui_stat['count']}")
                print(
                    f"  Final accuracy: {ui_stat.get('final_accuracy', 0):.4f} ({ui_stat['final_correct']}/{ui_stat['count']})")
            print("=" * 80)

        rst_stats["detailed_results"] = results
        return rst_stats

    def save_results(self, results):
        output_data = self.calculate_detailed_statistics(results)

        output_name_item = [self.version]

        if os.path.split(self.args.json_file_dir)[1] == "classification_result.json":
            output_file = f"rst_{self.benchmark}/{self.model_name}/classification_result/{'_'.join(output_name_item)}.json"
        elif self.benchmark == "uivision":
            suffix = self.args.json_file_dir.split('/')[-1].replace('.json', '').split('_')[-1]
            output_file = f"rst_{self.benchmark}/{self.model_name}/{'_'.join(output_name_item)}/{suffix}.json"
        else:
            output_file = f"rst_{self.benchmark}/{self.model_name}/{'_'.join(output_name_item)}.json"

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Detailed results saved to {output_file}")


class DistributedRunner:
    def __init__(self, args, gui_grounding_engine, rank_save_dir):
        self.args = args
        self.rank_save_dir = rank_save_dir
        self.rank, self.world_size, self.gpu = setup_distributed()
        self.engine = gui_grounding_engine(args, self.gpu)

    def run(self):
        if self.rank == 0:
            print(f"Using {self.world_size} GPUs")

        if self.args.benchmark == "osworld":
            return self.run_osworldg()

        subset = self.build_rank_subset(dataset=self.engine.dataset)
        dataloader = DataLoader(
            subset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )

        local_results = []
        if self.rank == 0:
            pbar = tqdm(total=len(dataloader), desc="Processing images")
        for batch in dataloader:
            for json_data in batch:
                result = self.engine.process_orig_image(json_data)
                if result is not None:
                    local_results.append(result)
            if self.rank == 0:
                pbar.update(1)
        if self.rank == 0:
            pbar.close()
        torch.cuda.empty_cache()

        results = self.collect_subprocess_results(local_results)
        if self.rank == 0:
            self.engine.save_results(results)

        if self.world_size > 1:
            dist.destroy_process_group()

    def run_osworldg(self):
        category_data = load_dataset_osworldg(self.args.json_file_dir, self.rank)
        all_results = {}
        for category_name, samples in category_data.items():
            if self.rank == 0:
                print(f"\n{'=' * 60}")
                print(f"Processing category: {category_name}")
                print(f"{'=' * 60}")

            dataset = ScreenSpotDataset(samples, self.args.base_image_dir)

            subset = self.build_rank_subset(dataset=dataset)
            dataloader = DataLoader(
                subset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True,
                collate_fn=custom_collate_fn
            )

            local_results = []
            if self.rank == 0:
                pbar = tqdm(total=len(dataloader), desc=f"Processing {category_name}")

            for batch in dataloader:
                for json_data in batch:
                    result = self.engine.process_orig_image(json_data)
                    if result is not None:
                        local_results.append(result)
                if self.rank == 0:
                    pbar.update(1)
            if self.rank == 0:
                pbar.close()
            torch.cuda.empty_cache()

            category_results = self.collect_subprocess_results(local_results, category_name)

            if self.rank == 0:
                all_results[category_name] = category_results

        if self.rank == 0:
            self.engine.calculate_detailed_statistics = calculate_detailed_statistics_osworldg
            self.engine.save_results(all_results)

        if self.world_size > 1:
            dist.destroy_process_group()

    def build_rank_subset(self, dataset):
        total_size = len(dataset)

        per_rank = math.ceil(total_size / self.world_size)
        start = self.rank * per_rank
        end = min(start + per_rank, total_size)

        indices = list(range(start, end))
        return Subset(dataset, indices)

    def collect_subprocess_results(self, local_results, category_name=''):
        if self.world_size == 1:
            return local_results

        os.makedirs(os.path.join(self.rank_save_dir, category_name), exist_ok=True)
        local_path = os.path.join(self.rank_save_dir, category_name, f"results_rank{self.rank}.json")
        with open(local_path, "w") as f:
            json.dump(local_results, f, indent=2)

        merged_results = []
        if self.rank == 0:
            expected_files = [
                os.path.join(self.rank_save_dir, category_name, f"results_rank{i}.json")
                for i in range(self.world_size)
            ]
            waiting_count = 0
            while waiting_count < 1000:
                if all(os.path.exists(f) for f in expected_files):
                    break
                time.sleep(5)
                waiting_count += 1
            time.sleep(2)

            for part_path in expected_files:
                with open(part_path, "r") as f:
                    merged_results.extend(json.load(f))

            if self.args.benchmark == "sspro":
                merged_results.sort(key=lambda x: x["id"])

        return merged_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    parser.add_argument('--benchmark', type=str, default="sspro", choices=["sspro", "ssv2", "uivision", "osworld", "mmbench"],
                        help='Benchmark options: ["sspro", "ssv2", "uivision", "osworld", "mmbench"]')
    parser.add_argument('--model_name', type=str, default="qwen3vl8b",
                        choices=["qwen3vl8b", "qwen3vl32b", "maiui8b", "guiowl8b"])
    parser.add_argument('--continuity_layer', type=int, default=2)
    parser.add_argument('--model_path', type=str, default=None, help='Path to the pre-trained model')
    parser.add_argument('--json_file_dir', type=str, default=None, help='Path to the JSON file with annotations')
    parser.add_argument('--base_image_dir', type=str, default=None, help='Base directory for images')

    args = parser.parse_args()
    if args.json_file_dir is None:
        args.json_file_dir = json_file_dir_map[args.benchmark]
    if args.base_image_dir is None:
        args.base_image_dir = base_image_dir_map[args.benchmark]
    if args.model_path is None:
        args.model_path = model_path_map[args.model_name]

    return args


def main():
    args = parse_args()
    runner = DistributedRunner(args, GUIGroundingEngine,
                               rank_save_dir=RANK_SAVE_DIR)
    runner.run()
    cleanup_distributed()


if __name__ == "__main__":
    main()
