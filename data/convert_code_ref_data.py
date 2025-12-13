import json
from collections import defaultdict
import random

source_data_file = 'code_KodCode_data_with_time.jsonl'
target_data_file = 'code_KodCode_data_with_time_e2e.jsonl'
ref_file = 'code_KodCode_data_e2e.jsonl'



ref_ids = []

with open(ref_file, 'r', encoding='utf-8') as f_in:
    for line in f_in:
        ref_ids.append(json.loads(line)['id'])
    
    ref_ids = set(ref_ids)
print(len(ref_ids))




with open(source_data_file, 'r', encoding='utf-8') as f_in:
    with open(target_data_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            id = json.loads(line)['id']
            if id in ref_ids:
                f_out.write(line)

    
# # 创建模板数据
# template_data_content = {
#     "id": None,
#     "source": "RLVR",
#     "difficulty": "0",
#     "prompt": "You are a chatbot who can solve problems. Please solve the following problem and give your thought process. Please reason step by step, and before giving the final result, you should output \"Therefore, the answer is\", and then give your final answer.",
#     "messages": "[{\"content\": \"You are a chatbot who can solve problems. Please solve the following problem and give your thought process. Please reason step by step, and before giving the final result, you should output \\\"Therefore, the answer is\\\", and then give your final answer.\", \"role\": \"system\"}, {\"content\": \"An adult male tested positive for HIV antibodies in the blood using the ELISA test during a health check. Confirmation tests should use\", \"role\": \"user\"}]",
#     "ground_truth": "Western blot",
#     "case_type": "",
#     "test_case_function": "",
#     "test_cases": "",
#     "tag": "RLVR"
# }

# # 打开输出文件
# with open(target_data_file, 'w', encoding='utf-8') as f_out, \
#      open(source_half_file, 'w', encoding='utf-8') as f_source_half:

#     for difficulty in sorted(difficulty_to_data.keys()):
#         data_list = difficulty_to_data[difficulty]
        
#         # Shuffle 数据
#         random.shuffle(data_list)
        
#         half_idx = len(data_list) // 2  # 取整数部分作为分界点

#         # 前一半：使用原有逻辑转换后写入 target_data_file
#         for i in range(half_idx):
#             math_data = data_list[i]
#             messages_str = math_data['messages']
#             messages = json.loads(messages_str)
#             prompts = messages[-1]["content"]

#             cur_message_template = json.loads(template_data_content['messages'])
#             cur_message_template[-1]["content"] = prompts

#             template_data_content_copy = template_data_content.copy()
#             template_data_content_copy['messages'] = json.dumps(cur_message_template)
#             template_data_content_copy['id'] = math_data['id']
#             template_data_content_copy['difficulty'] = math_data['difficulty']
#             template_data_content_copy['ground_truth'] = math_data['ground_truth']

#             f_out.write(json.dumps(template_data_content_copy, ensure_ascii=False) + '\n')

#         # 后一半：原样写入 source_half_file
#         for j in range(half_idx, len(data_list)):
#             f_source_half.write(json.dumps(data_list[j], ensure_ascii=False) + '\n')
