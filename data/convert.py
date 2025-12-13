import json

# 读取 time_list.json
json_path = "./time_list.json"
with open(json_path, "r") as f:
    time_data = json.load(f)

# 读取 code_KodCode_data.jsonl，添加 time 字段
code_json_path = "./code_KodCode_data.jsonl"
output_jsonl_path = "./code_KodCode_data_with_time.jsonl"

with open(code_json_path, "r") as fin, open(output_jsonl_path, "w") as fout:
    for line in fin:
        item = json.loads(line)
        code_id = item.get("id")
        if code_id is not None and code_id in time_data:
            item["timeout"] = time_data[code_id][0]
        json.dump(item, fout, ensure_ascii=False)
        fout.write("\n")