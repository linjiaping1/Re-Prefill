import os
import json


def load_dataset(json_file, rank=0):
    with open(json_file, 'r') as f:
        data = json.load(f)["classified"]

    category_data = {}
    for category_name, samples in data.items():
        for jd in samples:
            jd["img_filename"] = jd.pop("image_path")
            jd["bbox"] = jd.pop("box_coordinates")
            if jd["box_type"] == "bbox":
                x, y, width, height = jd["bbox"]
                jd["bbox"] = [x, y, x + width, y + height]
        category_data[category_name] = samples

    if rank == 0:
        print(f"Loaded {len(category_data)} categories:")
        for category_name, samples in category_data.items():
            print(f"  {category_name}: {len(samples)} samples")

    return category_data


def calculate_detailed_statistics(all_results):
    category_stats = {}
    overall_stats = {
        "total": 0,
        "final_correct": 0,
        "final_accuracy": 0,
    }

    for category_name, results in all_results.items():
        total = len(results)

        final_correct = sum(1 for r in results if r["final_prediction"] and r["final_prediction"]["in_bbox"])
        final_accuracy = final_correct / total if total > 0 else 0

        category_stats[category_name] = {
            "total": int(total),
            "final_correct": int(final_correct),
            "final_accuracy": float(final_accuracy),
        }

        overall_stats["total"] += total
        overall_stats["final_correct"] += final_correct

    overall_final_accuracy = overall_stats["final_correct"] / overall_stats["total"] if overall_stats[
                                                                                            "total"] > 0 else 0
    overall_stats["final_accuracy"] = float(overall_final_accuracy)

    print("\n" + "=" * 80)
    print("CATEGORY-WISE RESULTS:")
    print("=" * 80)

    for category_name, stats in category_stats.items():
        print(f"\n{category_name}:")
        print(f"  Total: {stats['total']}")
        print(f"  Final prediction accuracy: {stats['final_accuracy']:.4f} ({stats['final_correct']}/{stats['total']})")

    print("\n" + "=" * 80)
    print("OVERALL RESULTS:")
    print("=" * 80)
    print(f"Total images processed: {overall_stats['total']}")
    print(
        f"Overall Final prediction accuracy: {overall_stats['final_accuracy']:.4f} ({overall_stats['final_correct']}/{overall_stats['total']})")
    print("=" * 80)

    output_data = {
        "summary": {
            "overall": overall_stats,
            "categories": category_stats,
        },
        "detailed_results": all_results
    }
    return output_data


def save_results(all_results, args):
    output_data = calculate_detailed_statistics(None, all_results)
    model_name = args.model_path.split('/')[-1]
    output_file = f"rst_osworld/{args.method_name}_{model_name}_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {output_file}")
