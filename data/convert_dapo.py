import json

def process_jsonl(input_file, output_file):


    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        data = json.load(f_in)  # 读取整个 JSON 列表
        for item in data:
            new_item = {
                "id": item["id"],
                "domain": item["domain"],
                "source": item["sources"],  # 将sources字段映射到source
                "difficulty": "5.0",  # 所有difficulty设为5
                "prompt": item["prompt"],
                "messages": item["messages"],
                "ground_truth": item["ground_truth"],
                "case_type": item["case_type"],
                "test_case_function": item["test_case_function"],
                "test_cases": item["test_cases"],
                "tag": item["sources"]  # tag字段与source相同
            }
            f_out.write(json.dumps(new_item, ensure_ascii=False) + '\n')

# 使用示例
process_jsonl('math_dapo_with_difficulty_subjects_3.json', 'math_dapo_new.jsonl')
