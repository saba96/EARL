import os
os.environ["VLLM_USE_V1"] = "0"

import sys
sys.path.append('../')
sys.path.append('../VIEScore/src/')
sys.path.append('../evaluation/')

from constants import openai_key
from viescore.metric import VIEScore
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
import argparse
from collections import defaultdict



def calculate_viescore(json_path):
    vie_score = VIEScore(backbone="gpt4omini", task="tie", key_path=openai_key)
    vie_sc_scores = []
    vie_pq_scores = []
    vie_all_scores = []
    vie_sc_descriptions = []
    vie_pq_descriptions = []
    edit_success = []
    overedit = []
    looks_natural = []
    no_artifact = []
    output_path = json_path.replace('.json', '_viescore.json')
    grouped_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    source_stats = defaultdict(lambda: defaultdict(list))
    task_type_stats = defaultdict(lambda: defaultdict(list))
    with open(json_path, 'r') as f, open(output_path, 'w') as fout:
        lines = f.readlines()
        for idx, line in tqdm(enumerate(lines), total=len(lines)):
            entry = json.loads(line)
            orig_path = entry["original_image"]
            edit_path = entry["edited_image"]
            instr = entry["edit_instruction"]

            if "source" in entry:
                source = entry["source"]
            else:
                source = "unknown"

            if "task_type" in entry:
                task_type = entry["task_type"]
            elif "task" in entry:
                task_type = entry["task"]
            else:
                task_type = "unknown"

            # Load images
            orig_img = Image.open(orig_path).convert("RGB")
            edit_img = Image.open(edit_path).convert("RGB")
            print(f"Sample {idx}: orig size {orig_img.size}, edit size {edit_img.size}, instruction: {instr}")
            
            try:
                summarized, fine_grained_scores, descriptive  = vie_score.evaluate(image_prompts=[orig_img, edit_img], 
                                                                                                text_prompt=instr, 
                                                                                                extract_all_score=False)
            
                vie_sc, vie_pq, vie_all = summarized
                vie_sc_desc = descriptive['SC']['reasoning']
                vie_pq_desc = descriptive['PQ']['reasoning']
            except:
                print(f"Error in sample {idx}")
                vie_sc, vie_pq, vie_all = 0, 0, 0
                vie_sc_desc = ""
                vie_pq_desc = ""
                fine_grained_scores = {'edit_success': 0, 'overedit': 0, 'looks_natural': 0, 'no_artifact': 0}

            vie_sc_scores.append(vie_sc)
            vie_pq_scores.append(vie_pq)
            vie_all_scores.append(vie_all)
            vie_sc_descriptions.append(vie_sc_desc)
            vie_pq_descriptions.append(vie_pq_desc)
            edit_success.append(fine_grained_scores['edit_success'])
            overedit.append(fine_grained_scores['overedit'])
            looks_natural.append(fine_grained_scores['looks_natural'])
            no_artifact.append(fine_grained_scores['no_artifact'])

            # Write per-sample result
            result = {
                "original_image": orig_path,
                "edited_image": edit_path,
                "edit_instruction": instr,
                "vie_sc": vie_sc,
                "vie_pq": vie_pq,
                "vie_all": vie_all,
                "vie_sc_description": vie_sc_desc,
                "vie_pq_description": vie_pq_desc,
                "edit_success": fine_grained_scores['edit_success'],
                "overedit": fine_grained_scores['overedit'],
                "looks_natural": fine_grained_scores['looks_natural'],
                "no_artifact": fine_grained_scores['no_artifact'],
                "source": source,
                "task_type": task_type
            }
            fout.write(json.dumps(result) + "\n")

            for score_name, value in [
                ('vie_sc', vie_sc),
                ('vie_pq', vie_pq),
                ('vie_all', vie_all),
                ('edit_success', fine_grained_scores['edit_success']),
                ('overedit', fine_grained_scores['overedit']),
                ('looks_natural', fine_grained_scores['looks_natural']),
                ('no_artifact', fine_grained_scores['no_artifact']),
            ]:
                grouped_stats[source][task_type][score_name].append(value)

            # For each score, append to the source and task_type stats
            for score_name, value in [
                ('vie_sc', vie_sc),
                ('vie_pq', vie_pq),
                ('vie_all', vie_all),
                ('edit_success', fine_grained_scores['edit_success']),
                ('overedit', fine_grained_scores['overedit']),
                ('looks_natural', fine_grained_scores['looks_natural']),
                ('no_artifact', fine_grained_scores['no_artifact']),
            ]:
                source_stats[source][score_name].append(value)
                task_type_stats[task_type][score_name].append(value)

            # if idx > 10:
            #     break

    # Print mean and std for each score type
    def print_stats(name, arr):
        arr = np.array(arr)
        print(f"{name}: mean={arr.mean():.4f}, std={arr.std():.4f}")

    print_stats("VIE_SC", vie_sc_scores)
    print_stats("VIE_PQ", vie_pq_scores)
    print_stats("VIE_ALL", vie_all_scores)
    print_stats("EDIT_SUCCESS", edit_success)
    print_stats("OVEREDIT", overedit)
    print_stats("LOOKS_NATURAL", looks_natural)
    print_stats("NO_ARTIFACT", no_artifact)

    # Collect stats in a dictionary
    stats = {
        "VIE_SC": {"mean": float(np.mean(vie_sc_scores)), "std": float(np.std(vie_sc_scores))},
        "VIE_PQ": {"mean": float(np.mean(vie_pq_scores)), "std": float(np.std(vie_pq_scores))},
        "VIE_ALL": {"mean": float(np.mean(vie_all_scores)), "std": float(np.std(vie_all_scores))},
        "EDIT_SUCCESS": {"mean": float(np.mean(edit_success)), "std": float(np.std(edit_success))},
        "OVEREDIT": {"mean": float(np.mean(overedit)), "std": float(np.std(overedit))},
        "LOOKS_NATURAL": {"mean": float(np.mean(looks_natural)), "std": float(np.std(looks_natural))},
        "NO_ARTIFACT": {"mean": float(np.mean(no_artifact)), "std": float(np.std(no_artifact))}
    }

    stats_path = output_path.replace('.json', '_stats.json')
    with open(stats_path, 'w') as stats_file:
        json.dump(stats, stats_file, indent=2)
    print(f"Stats written to {stats_path}")

    # Compute stats for source and task_type
    source_and_task_type_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for source, scores_dict in grouped_stats.items():
        for task_type, scores_dict in scores_dict.items():
            for score_name, values in scores_dict.items():
                arr = np.array(values)
                source_and_task_type_stats[source][task_type][score_name] = {
                    "mean": float(arr.mean()) if len(arr) > 0 else None,
                    "std": float(arr.std()) if len(arr) > 0 else None,
                    "count": int(len(arr))
                }

    source_and_task_type_stats_path = output_path.replace('.json', '_source_and_tasktype_stats.json')
    with open(source_and_task_type_stats_path, 'w') as f:
        json.dump(source_and_task_type_stats, f, indent=2)
    print(f"Source and task type stats written to {source_and_task_type_stats_path}")


    # Compute stats for each source
    source_stats_summary = {}
    for source, scores_dict in source_stats.items():
        source_stats_summary[source] = {}
        for score_name, values in scores_dict.items():
            arr = np.array(values)
            source_stats_summary[source][score_name] = {
                "mean": float(arr.mean()) if len(arr) > 0 else None,
                "std": float(arr.std()) if len(arr) > 0 else None,
                "count": int(len(arr))
            }

    # Compute stats for each task_type
    task_type_stats_summary = {}
    for task_type, scores_dict in task_type_stats.items():
        task_type_stats_summary[task_type] = {}
        for score_name, values in scores_dict.items():
            arr = np.array(values)
            task_type_stats_summary[task_type][score_name] = {
                "mean": float(arr.mean()) if len(arr) > 0 else None,
                "std": float(arr.std()) if len(arr) > 0 else None,
                "count": int(len(arr))
            }

    # Save and print source stats
    source_stats_path = output_path.replace('.json', '_source_stats.json')
    with open(source_stats_path, 'w') as f:
        json.dump(source_stats_summary, f, indent=2)
    print(f"Source stats written to {source_stats_path}")

    # Save and print task_type stats
    task_type_stats_path = output_path.replace('.json', '_tasktype_stats.json')
    with open(task_type_stats_path, 'w') as f:
        json.dump(task_type_stats_summary, f, indent=2)
    print(f"Task type stats written to {task_type_stats_path}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate VIEScore stats for a results JSONL file.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the input JSONL file.")
    args = parser.parse_args()
    calculate_viescore(json_path=args.json_path)
