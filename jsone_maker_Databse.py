import os
import json
import csv

folder = "json_pages"  # your folder with JSON files
csv_output = "dataset.json"
jsonl_output = "training_data.jsonl"

# CSV Writer
with open(csv_output, "w", newline='', encoding='utf-8') as f_csv, open(jsonl_output, "w", encoding='utf-8') as f_jsonl:
    writer = csv.writer(f_csv)
    writer.writerow(["page_name", "json_output"])

    for file in os.listdir(folder):
        if file.endswith(".json"):
            page_name = os.path.splitext(file)[0]
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                raw_json = json.load(f)
                writer.writerow([page_name, json.dumps(raw_json)])
                f_jsonl.write(json.dumps({"page_name": page_name, "json_output": raw_json}) + "\n")
