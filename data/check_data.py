import hashlib
import base64
import json

dict = {}

# for line in open('math_deepmath_deal.jsonl'):
# for line in open('math_dapo_new.jsonl'):
for line in open('code_KodCode_data_with_time_e2e.jsonl'):
# for line in open('math_deepmath_deal_e2e.jsonl'):
# for line in open('llm_judge_math_rlvr_e2e.jsonl'):
# for line in open('code_rl_data_v2.4_top2048_shortest_correct_test_cases_e2e.jsonl'):
    dataset_item = json.loads(line)
    prompt_digest = hashlib.md5(dataset_item['prompt'].encode() + dataset_item['messages'].encode()).digest()
    prompt_hash = base64.urlsafe_b64encode(prompt_digest).decode().rstrip('=')
    if prompt_hash in dict:
        print(dataset_item)
        print(dict[prompt_hash])
        print('-------------------')
        print('-------------------')
    dict[prompt_hash] = dataset_item
