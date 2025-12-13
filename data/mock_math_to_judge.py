import json

source_data_file = 'math_dapo_e2e.jsonl'
target_data_file = 'llm_judge_dapo_rlvr_e2e.jsonl'


i = 0
with open(source_data_file, 'r', encoding='utf-8') as f_in, \
     open(target_data_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:

        template_data_content = {
            "id": None, 
            "source": "RLVR", 
            "domain": "llm_judge",
            "difficulty": "0",
            "prompt": "You are a chatbot who can solve problems. Please solve the following problem and give your thought process. Please reason step by step, and before giving the final result, you should output \"Therefore, the answer is\", and then give your final answer.", 
            "messages": "[{\"content\": \"You are a chatbot who can solve problems. Please solve the following problem and give your thought process. Please reason step by step, and before giving the final result, you should output \\\"Therefore, the answer is\\\", and then give your final answer.\", \"role\": \"system\"}, {\"content\": \"An adult male tested positive for HIV antibodies in the blood using the ELISA test during a health check. Confirmation tests should use\", \"role\": \"user\"}]", 
            "ground_truth": "Western blot", 
            "case_type": "", 
            "test_case_function": "", 
            "test_cases": "", 
            "tag": "RLVR"
        }
        math_data = json.loads(line)
        messages_str = math_data['messages']
        messages = json.loads(messages_str)
        prompts = messages[-1]["content"]

        cur_message_template = json.loads(template_data_content['messages'])
        cur_message_template[-1]["content"] = prompts

        template_data_content['messages'] = json.dumps(cur_message_template)
        # .replace("\\\\", "\\")
        template_data_content['id'] = math_data['id']
        template_data_content['difficulty'] = math_data['difficulty']
        template_data_content['ground_truth'] = math_data['ground_truth']
        # template_data_content["messages"] = template_data_content["messages"].replace("\\\\", "\\")

        f_out.write(json.dumps(template_data_content, ensure_ascii=False) + '\n')

