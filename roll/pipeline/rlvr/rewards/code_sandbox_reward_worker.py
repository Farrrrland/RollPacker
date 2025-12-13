"""
Code Sandbox Reward Worker for evaluating code solutions.
This module provides functionality to test code solutions in a sandbox environment
and compute rewards based on test case results.
"""

import asyncio
import json
import re
import time
import traceback
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any

import aiohttp
import ray
import torch

from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.rlvr.rlvr_config import RewardConfig
from roll.utils.local_code.evaluator import codegen_metrics
from roll.utils.local_code.extract_utils import extract_code_generation
from roll.utils.logging import get_logger
from roll.utils.constants import FixedTimeOut


def modified_text(text: str) -> str:
    text = re.sub(r"(\n) +(?=\S|\n)", r"\1", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip().replace(" \n", "\n")


def remove_entrypoints(code: str, language: str = "python") -> str:
    """
    Remove entry points and example usage from code.
    
    Args:
        code: Source code to process
        language: Programming language of the code
        
    Returns:
        Processed code with entry points removed
    """
    if language == "python":
        if 'if __name__ == "__main__":' in code:
            try:
                next_line = code.index('if __name__ == "__main__":')
                code = code[:next_line].strip()
            except ValueError:
                pass
        elif "if __name__ == '__main__':" in code:
            try:
                next_line = code.index("if __name__ == '__main__':")
                code = code[:next_line].strip()
            except ValueError:
                pass
    elif language == "cpp":
        if "int main()" in code:
            try:
                next_line = code.index("int main()")
                code = code[:next_line].strip()
            except ValueError:
                pass
    elif language == "go":
        code = code.replace("package main", "")

    if "# Example usage" in code:
        try:
            next_line = code.index("# Example usage")
            code = code[:next_line].strip()
        except ValueError:
            pass

    if language == "python" and "def" in code:
        lines = code.strip().split("\n")
        cleaned_lines = [line for line in lines if line.startswith(" ") or line.startswith("def ")]
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        code = "\n".join(cleaned_lines) + "\n"
        
    return code


class CodeTester:
    """
    Class for testing code solutions using a sandbox api.
    """
    def __init__(self, sandbox_url: str):
        self.DEFAULT_TIMEOUT = 30
        self.SOLUTION_IMPORTS = {
            "python": "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\nfrom typing import List, Dict, Any, Optional, Tuple\nfrom math import inf\n"
        }
        self.sandbox_url = sandbox_url
        self.logger = get_logger()

    def check_format(self, prompt_id: str, text: str) -> Tuple[int, int]:
        has_think_tag = 1 if "</think>" in text else 0
        has_code_block = 1 if "```" in text else 0
        return has_code_block, has_think_tag

    def extract_code_blocks(self, prompt: str, text: str, case_type: str = "input") -> Tuple[Optional[List[str]], Optional[List[str]], str]:
        """
        Extract code blocks from the response text.
        """
        if "<|begin_of_solution|>" in text:
            text = text.split("<|begin_of_solution|>")[-1].strip()
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()

        if "```" in text:
            code_pattern = r"```(cpp|python|python3|java|javascript|ruby|go)\s*\n*(.*?)```"
            matches = re.findall(code_pattern, text, re.DOTALL)
            codes = []
            langs = []
            if matches:
                for lang, code in matches:
                    codes.append(code.strip())
                    langs.append(lang)

            if len(codes) == 0:
                code_pattern = r"```\s*\n*(.*?)```"
                matches = re.findall(code_pattern, text, re.DOTALL)
                codes = [match.strip() for match in matches]
                langs = ["python"] * len(codes)  # Default to Python
        elif len(text) > 4000:
            return None, None, "No code block found"
        else:
            codes = [text.strip()]
            langs = ["python"] * len(codes)
            
        if not codes:
            return None, None, "No code block found"

        if case_type != "input" and "```python\ndef " in prompt:
            codes = [remove_entrypoints(code, lang) for code, lang in zip(codes, langs)]
            
        return codes, langs, ""

    def format_sandbox_test(self, test_code: str, code_language: str, case_type: str, test_cases: List[Dict]) -> Tuple[Optional[List[Dict]], str]:
        """
        Format test cases for sandbox testing.
        """
        test_cases_final = []

        if not code_language:
            code_language = "python"
            
        if not test_cases:
            return None, "test case is empty"
            
        if case_type == "text":
            # Text-based test cases
            for case in test_cases:
                assert_code = case["assert_code"] if "assert_code" in case else case
                case_code = self.SOLUTION_IMPORTS.get(code_language, "") + '\n' + assert_code
                cur_test_code = f"\'\'\'{test_code}\'\'\'"
                case_code = case_code.replace("{response}", cur_test_code)
                
                test_cases_final.append({
                    "compile_timeout": self.DEFAULT_TIMEOUT,
                    "run_timeout": self.DEFAULT_TIMEOUT,
                    "code": case_code,
                    "language": code_language,
                    "stdin": "",
                    "expected_stdout": ""
                })
        elif case_type in ("assert", "pytest"):
            # Assert or pytest test cases
            if case_type == "pytest":
                code_language = "pytest"
                
            for case in test_cases:
                assert_code = case["assert_code"] if "assert_code" in case else case
                case_code = self.SOLUTION_IMPORTS.get(code_language, "") + test_code + "\n" + assert_code
                
                test_cases_final.append({
                    "compile_timeout": self.DEFAULT_TIMEOUT,
                    "run_timeout": self.DEFAULT_TIMEOUT,
                    "code": case_code,
                    "language": code_language,
                    "stdin": "",
                    "expected_stdout": "",
                })
        else:
            # Standard input/output test cases
            for test_case in test_cases:
                try:
                    test_cases_final.append({
                        "compile_timeout": self.DEFAULT_TIMEOUT,
                        "run_timeout": self.DEFAULT_TIMEOUT,
                        "code": self.SOLUTION_IMPORTS.get(code_language, "") + test_code,
                        "language": code_language,
                        "stdin": test_case["stdin"],
                        "expected_stdout": test_case["expected_stdout"],
                    })
                except Exception as e:
                    self.logger.error(f"Error formatting test case: {e}")
                    self.logger.error(traceback.format_exc())
                    
        return test_cases_final, ""

    async def process_single_test(
        self, curid: str, session: aiohttp.ClientSession, test_case: Dict, max_retries: int = 3
    ) -> Tuple[Dict, List[Dict]]:
        """
        Process a single test case using the sandbox.
        """
        results = []
        
        for _ in range(2):
            retries = 0
            result = None
            
            while retries < max_retries:
                try:
                    headers = {
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "eagleeye-traceid": str(uuid.uuid4()),
                    }
                    
                    async with session.post(self.sandbox_url, headers=headers, json=test_case) as response:
                        if response.status == 200:
                            result = await response.json()

                            if "status" in result and result["status"] == "SandboxError":
                                error_msg = result.get("message", "Unknown sandbox error")
                                self.logger.warning(f"curid: {curid}, Sandbox error: {error_msg}, retry: {retries+1}/{max_retries}")
                                retries += 1
                                await asyncio.sleep(1)
                                continue
                            else:
                                break
                        else:
                            error_text = await response.text()
                            self.logger.warning(
                                f"curid: {curid}, HTTP error {response.status}: {error_text}, retry: {retries+1}/{max_retries}"
                            )
                            retries += 1
                            await asyncio.sleep(1)
                except asyncio.TimeoutError:
                    self.logger.warning(f"curid: {curid}, Request timeout, retry: {retries+1}/{max_retries}")
                    retries += 1
                    await asyncio.sleep(1)
                except Exception as e:
                    self.logger.error(f"curid: {curid}, Sandbox error: {e}, retry: {retries+1}/{max_retries}")
                    retries += 1
                    await asyncio.sleep(1)
                    
            if result:
                results.append(result)

        return test_case, results

    async def sandbox_test_async(self, prompt_id: str, test_cases: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Run sandbox tests asynchronously.
        """
        succeed_test_cases = []
        total_res = []
        concurrency_limit = 20

        connector = aiohttp.TCPConnector(
            limit=concurrency_limit,
            ttl_dns_cache=600,
            enable_cleanup_closed=True,
            force_close=False,
            keepalive_timeout=60,
            ssl=False,
        )
        timeout = aiohttp.ClientTimeout(total=90, connect=20, sock_read=30)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            semaphore = asyncio.Semaphore(concurrency_limit)
            async def bounded_process_single_test(test_case):
                async with semaphore:
                    return await self.process_single_test(prompt_id, session, test_case)
            tasks = [bounded_process_single_test(test_case) for test_case in test_cases]
            for task in asyncio.as_completed(tasks):
                try:
                    test_case, results = await task
                    if test_case and results:
                        succeed_test_cases.append(test_case)
                        total_res.append(results)
                except Exception as e:
                    self.logger.error(f"prompt_id: {prompt_id}, Task error: {e}")
                    
        return succeed_test_cases, total_res

    def sandbox_test(self, prompt_id: str, test_cases: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Run sandbox tests (synchronous wrapper for async implementation).
        """
        return asyncio.run(self.sandbox_test_async(prompt_id, test_cases))

    def sandbox_result_judge(self, test_cases: List[Dict], sandbox_results: List[Dict]) -> Tuple[int, List[str]]:
        """
        Judge the results of sandbox tests.
        
        Args:
            test_cases: List of test cases
            sandbox_results: List of sandbox results
            
        Returns:
            Tuple containing the number of passed tests and a list of error types
        """
        pass_test_number = 0
        error_types = []
        
        for i, responses in enumerate(sandbox_results):
            flag = "[No Pass]"
            
            for response in responses:
                status = response["status"]
                sandbox_output = ""
                
                if "expected_stdout" in test_cases[i]:
                    expected_output = test_cases[i].get("expected_stdout", "").strip()
                else:
                    expected_output = test_cases[i].get("expected_output", "").strip()
                    
                if status == "Success":
                    sandbox_output = response.get("run_result", {}).get("stdout", "").strip()
                    if "User customization applied." in sandbox_output:
                        sandbox_output = sandbox_output.replace("User customization applied.", "").strip()
                    if "User customization module loaded!" in sandbox_output:
                        sandbox_output = sandbox_output.replace("User customization module loaded!", "").strip()
                        
                    if "User" in sandbox_output:
                        if expected_output in sandbox_output:
                            flag = "[Pass]"
                    elif (
                        expected_output == ""
                        or sandbox_output == expected_output
                        or modified_text(sandbox_output) == modified_text(expected_output)
                    ):
                        flag = "[Pass]"
                        
                    if flag == "[No Pass]":
                        error_types.append("LogicError")
                        
                elif status == "Failed":
                    try:
                        stderr = response.get("run_result", {}).get("stderr", "").strip()
                        stderr = stderr.split("\n")[-1].split(":")[0].strip()
                        if stderr == "":
                            stderr = response.get("run_result", {}).get("return_code", "").strip()
                            error_types.append(f"ReturnCode: {stderr}")
                        else:
                            error_types.append(stderr)
                    except Exception as e:  # Added specific exception type
                        self.logger.error(f"Error processing stderr: {e}")
                        error_types.append("Others")
                        
            if flag == "[Pass]":
                pass_test_number += 1
                
        error_types = list(set(error_types))
        return pass_test_number, error_types

    def single_code_test(
        self,
        global_step: int,
        prompt_id: str,
        code: str,
        case_type: str,
        test_cases: List[Dict],
        prompt: str = "",
        flag: int = 0,
    ) -> Dict:
        """
        Test a single code solution.
        """
        info = {
            "global_step": global_step,
            "prompt_id": prompt_id,
            "pass_test_ratio": 0,
            "origin_response": code,
            "test_code": code,
            "test_cases_info": test_cases,
            "sandbox_responses": "",
            "succeed_test_cases_number": 0,
            "pass_test_number": 0,
            "validation": 1,
            "format_validation": 0,
            "format_think": 0,
            "error_info": [],
        }
        format_validation, format_think = self.check_format(prompt_id, code)
        info["format_validation"] = format_validation
        info["format_think"] = format_think
        
        start_time = time.time()
        
        if case_type == "text":
            codes = [code.strip()]
            code_langs = []
            error_info = ""
        else:
            codes, code_langs, error_info = self.extract_code_blocks(prompt, code, case_type)
            
        if error_info or not codes:
            info["error_info"] = ["extract_code_blocks error"]
            return info

        test_code = codes[-1]
        if code_langs:
            code_language = code_langs[-1]
        else:
            code_language = "python"  # Default to Python

        test_cases, error_info = self.format_sandbox_test(test_code, code_language, case_type, test_cases)
        if error_info or test_cases is None:
            info["error_info"] = ["format_sandbox_test error"]
            return info

        succeed_test_cases, responses = self.sandbox_test(prompt_id, test_cases)
        if not responses or not succeed_test_cases:
            info["error_info"] = ["sandbox error"]
            info["sandbox_responses"] = responses
            info["validation"] = 0
            return info
        pass_test_number, error_types = self.sandbox_result_judge(succeed_test_cases, responses)
        time_duration = time.time() - start_time
        self.logger.debug(
            f"prompt_id: {prompt_id}, total case number: {len(succeed_test_cases)} "
            f"pass case number: {pass_test_number} "
            f"pass rate: {pass_test_number / len(succeed_test_cases) if succeed_test_cases else 0}, "
            f"time: {time_duration}, detailed info: {error_info}"
        )
        pass_test_ratio = pass_test_number / len(succeed_test_cases) if succeed_test_cases else 0
        info["test_code"] = test_code
        info["test_cases_info"] = test_cases
        info["sandbox_responses"] = responses
        info["succeed_test_cases_number"] = len(succeed_test_cases)
        info["pass_test_number"] = pass_test_number
        info["pass_test_ratio"] = pass_test_ratio
        info["error_info"] = error_types
        
        return info

def cal_http_sandbox(global_step: int, prompt_id: str, prompt: str, response: str, 
                    case_type: str, test_cases: List[Dict], url: str) -> Tuple[int, Dict, str]:
    """
    Calculate rewards using HTTP sandbox.
    """
    codetester = CodeTester(url)
    info = codetester.single_code_test(global_step, prompt_id, response, case_type, test_cases, prompt)

    validation = info.get("validation", 0)
    pass_test_ratio = info.get("pass_test_ratio", 0)
    
    error_info = info.get('error_info', [""])
    error_info_str = error_info[0] if error_info else ""
    
    if validation == 0:
        info["error"] = "Sandbox validation failed"
        info["error_code"] = -1
        info["error_message"] = "Compilation Error"
        return -1, info, "Sandbox validation failed"
    elif error_info and not pass_test_ratio >= 1:
        error_type = error_info[0] if error_info else "Unknown Error"
        if "SyntaxError" in error_type or "IndentationError" in error_type:
            info["error"] = error_type
            info["error_code"] = -1
            info["error_message"] = "Compilation Error"
        else:
            info["error"] = error_type
            info["error_code"] = -2
            info["error_message"] = "Wrong Answer"
        return -1, info, error_type
        
    correct = 1 if pass_test_ratio >= 1 else 0
    return correct, info, error_info_str

def run_assert_tests(code: str, test_cases: List[Dict], timeout: float = 20) -> Tuple[bool, Dict]:
    """
    Run assert-style test cases against the provided code.
    """
    from roll.utils.local_code.execute_utils import BASE_IMPORTS, codeexecute_check_correctness
    
    all_passed = True
    error_info = {}
    for test_case in test_cases:
        assert_code = test_case["assert_code"]
        
        if "def test_" in assert_code:
            test_functions = []
            lines = assert_code.strip().split('\n')
            for line in lines:
                if line.strip().startswith("def test_"):
                    func_name = line.strip().split("(")[0].split("def ")[1]
                    test_functions.append(func_name)
            
            test_runner = "\n\n# Test runner\ntry:\n"
            for func_name in test_functions:
                test_runner += f"    {func_name}()\n"
            test_runner += "    print('All tests passed!')\nexcept AssertionError as e:\n    print(f'Test failed: {e}')\n    exit(1)"

            full_code = f"{BASE_IMPORTS}\n{code}\n{assert_code}\n{test_runner}"
        else:
            full_code = f"{BASE_IMPORTS}\n{code}\n{assert_code}"
        
        try:
            passed = codeexecute_check_correctness(full_code, timeout=timeout)
            if not passed:
                all_passed = False
                error_info = {
                    'error': 'AssertionError in test case',
                    'error_code': -2,
                    'error_message': 'Wrong Answer'
                }
                break
        except SyntaxError as e:
            all_passed = False
            error_info = {
                'error': str(e),
                'error_code': -1,
                'error_message': 'Compilation Error'
            }
            break
        except Exception as e:
            all_passed = False
            error_info = {
                'error': str(e),
                'error_code': -3,
                'error_message': 'Runtime Error'
            }
            break
    
    return all_passed, error_info

def run_check_based_tests(extracted_code: str, test_cases: List[Dict], timeout: float = FixedTimeOut) -> Tuple[bool, float, Dict]:
    """
    Run check-based test cases against the provided code.
    """
    import tempfile
    import subprocess
    
    error_info = {}
    if len(test_cases) == 0:
        return True, 1, error_info
    test_case = test_cases[0]
    
    test_code_lines = [x for x in test_case['assert_code'].split("\n") if x != ""]
    entry_point = test_case.get('entry_point', 'candidate')
    import_prefix = test_case.get('import_prefix', '')
    
    solution = import_prefix + extracted_code
    
    all_passed = True
    total_tests = len(test_code_lines) - 1 if len(test_code_lines) > 1 else 1
    passed_tests = 0
    
    for i in range(1, len(test_code_lines)) if len(test_code_lines) > 1 else [0]:
        cur_solution = solution
        if len(test_code_lines) > 1:
            cur_solution += "\n" + test_code_lines[0] + "\n" + test_code_lines[i]
        else:
            cur_solution += "\n" + test_code_lines[0]
        cur_solution += f"\ncheck({entry_point})"
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as temp_file:
                temp_file.write(cur_solution)
                temp_file.flush()
                result = subprocess.run(
                    ['python', temp_file.name],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                if result.returncode == 0:
                    passed_tests += 1
                else:
                    all_passed = False
                    error_message = result.stderr.strip() if result.stderr else "Check failed"
                    error_info = {
                        'error': error_message,
                        'error_code': -2,
                        'error_message': 'Wrong Answer'
                    }
        except Exception as e:
            all_passed = False
            error_info = {
                'error': str(e),
                'error_code': -1 if 'SyntaxError' in str(e) else -2,
                'error_message': 'Compilation Error' if 'SyntaxError' in str(e) else 'Runtime Error'
            }
    
    pass_ratio = passed_tests / total_tests if total_tests > 0 else 0
    return all_passed, pass_ratio, error_info

def run_text_tests(response: str, test_cases: List[Dict]) -> Tuple[bool, Dict]:
    """
    Run text-based test cases against the provided response.
    """
    error_info = {}
    all_passed = True
    
    for test_case in test_cases:
        assert_code = test_case["assert_code"] if "assert_code" in test_case else test_case
        quoted_response = f"\'\'\'{response}\'\'\'"
        case_code = assert_code.replace("{response}", quoted_response)
        full_code = case_code
        
        try:
            exec(full_code)
            passed = True
        except AssertionError as e:
            passed = False
            error_info = {
                'error': str(e),
                'error_code': -2,
                'error_message': 'Wrong Answer'
            }
        except Exception as e:
            print(f"Error executing test case: {e}")
            passed = False
            error_info = {
                'error': str(e),
                'error_code': -1 if 'SyntaxError' in str(e) else -3,
                'error_message': 'Compilation Error' if 'SyntaxError' in str(e) else 'Runtime Error'
            }
            
        if not passed:
            all_passed = False
            break
    
    return all_passed, error_info

def run_io_tests(extracted_code: str, test_cases: List[Dict], func_name: str = None, 
                num_process_evaluate: int = 1, timeout: float = 20) -> bool:
    """
    Run input/output test cases against the provided code.
    """
    from roll.utils.local_code.evaluator import codegen_check_correctness
    
    if func_name == "None":
        func_name = ""
        
    evaluation_sample = {
        "inputs": [t["stdin"] for t in test_cases],
        "outputs": [t["expected_stdout"] for t in test_cases],
        "fn_name": func_name,
    }
    res, metadata = codegen_check_correctness(evaluation_sample, extracted_code, timeout=timeout, debug=True)
    
    all_passed = int(all(x == 1 for x in res))
    
    return all_passed, metadata

def cal_local_test(global_step: int, prompt_id: str, prompt_txt: str, response: str, 
                  case_type: str, test_cases: List[Dict], func_name: str = None, 
                  num_process_evaluate: int = 1, timeout: float = 20) -> Tuple[int, Dict, str]:
    """
    Calculate rewards using local testing.
    
    Reference implementation from: https://github.com/LiveCodeBench/LiveCodeBench/tree/main/lcb_runner
    
    Supports multiple testing modes:
    1. Input/output testing: Test cases contain stdin/expected_stdout pairs
    2. Assert testing: Test cases contain simple assert statements
    3. Pytest testing: Test cases contain pytest-style test functions
    4. Text testing: Test cases contain assert code that validates text responses
    5. Check-based testing: Test cases contain import_prefix, test_code, and entry_point
    """
    from roll.utils.local_code.execute_utils import BASE_IMPORTS
    
    info = {
        "global_step": global_step,
        "prompt_id": prompt_id,
        "pass_test_ratio": 0,
        "prompt_txt": prompt_txt,
        "origin_response": response,
        "test_code": response,
        "format_validation": 0,
        "format_think": 0,
        "validation": 1,
    }
    if "</think>" in response:
        info['format_think'] = 1
    if "```" in response:
        info['format_validation'] = 1
    
    if response == "":
        info["error"] = "Empty Response"
        info["error_code"] = -1
        info["error_message"] = "Empty Response"
        return 0, info, ""
    
    extracted_code = ""
    if "<|begin_of_solution|>" in response:
        response = response.split("<|begin_of_solution|>")[-1].strip()
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()
    if case_type != "text":
        try:
            extracted_code = extract_code_generation(response)
            info["test_code"] = extracted_code
        except Exception as e:
            info["error"] = str(e)
            info["error_code"] = -1
            info["error_message"] = "Extract Response Error"
            return 0, info, ""
    
    extracted_code = BASE_IMPORTS + extracted_code
    
    try:
        correct = 0
        if case_type == "check_based":
            try:
                all_passed, pass_ratio, error_info = run_check_based_tests(extracted_code, test_cases, timeout)
                info.update(error_info)
                info["pass_test_ratio"] = pass_ratio
                if all_passed:
                    correct = 1
            except TypeError as e:
                info["error"] = "Test environment error: " + str(e)
                info["error_code"] = -4
                info["error_message"] = "Environment Error"
                return 0, info, "Environment Error"
                
        elif case_type == "text":
            try:
                all_passed, error_info = run_text_tests(response, test_cases)
                info.update(error_info)
                if all_passed:
                    info["pass_test_ratio"] = 1
                    correct = 1
            except TypeError as e:
                info["error"] = "Test environment error: " + str(e)
                info["error_code"] = -4
                info["error_message"] = "Environment Error"
                return 0, info, "Environment Error"
                
        elif case_type == "assert" or case_type == "pytest":
            try:
                all_passed, error_info = run_assert_tests(extracted_code, test_cases, timeout=timeout)
                info.update(error_info)
                if all_passed:
                    info["pass_test_ratio"] = 1
                    correct = 1
            except TypeError as e:
                info["error"] = "Test environment error: " + str(e)
                info["error_code"] = -4
                info["error_message"] = "Environment Error"
                return 0, info, "Environment Error"
                
        else:
            try:
                all_passed, metadata = run_io_tests(extracted_code, test_cases, func_name, 
                                     num_process_evaluate, timeout)
                if metadata:
                    info.update(metadata)
                if all_passed:
                    info["pass_test_ratio"] = 1
                    correct = 1
                elif "error_message" not in info and not all_passed:
                    info["error"] = "Test failed"
                    info["error_code"] = -2
                    info["error_message"] = "Wrong Answer"
            except TypeError as e:
                info["error"] = "Test environment error: " + str(e)
                info["error_code"] = -4
                info["error_message"] = "Environment Error"
                return 0, info, "Environment Error"
    except Exception as e:
        info["error"] = f"Unexpected error: {str(e)}"
        info["error_code"] = -5
        info["error_message"] = "Unexpected Error"
        return 0, info, str(e)
    if correct and info.get("error_message", "") == "":
        info['error_message'] = "Succeed"

    return correct, info, info["error_message"]

@ray.remote(concurrency_groups={"multi_thread": 1024})
class CodeSandboxRewardWorker(Worker):
    """
    Worker for computing rewards based on code sandbox testing.
    """

    def __init__(self, worker_config: RewardConfig):
        """
        Initialize the CodeSandboxRewardWorker.
        
        Args:
            worker_config: Configuration for the worker
        """
        super().__init__(worker_config=worker_config)
        self.worker_config = worker_config
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None
        self.use_local = self.worker_config.use_local
        self.url = self.worker_config.code_url
        self.max_resp_len = 10000
        
        # from tools.timer import Timer
        # self.timer = Timer()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        """
        Initialize the worker with pipeline configuration.
        
        Args:
            pipeline_config: Configuration for the pipeline
        """
        super().initialize(pipeline_config)

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    @ray.method(concurrency_group="multi_thread")
    def compute_rewards(self, data: DataProto) -> DataProto:
        """
        Compute rewards for code solutions.
        """
        global_step = data.meta_info.get("global_step", 0)
        request_id = data.meta_info.get("request_id", -1)
        origin_prompt_id = data.non_tensor_batch["id"]
        self.logger.info(f"[TIMER][compute_rewards_codesandbox_start][request_id={request_id}][origin_prompt_id={origin_prompt_id}][time={time.time()}]")

        verify_answer = []
        error_infos = []
        format_validations = []
        have_thinks = []
        prompts_text_list = self.tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=True)
        response_text_list = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
        start_compute_reward_list = []
        end_compute_reward_list = []
        reward_score_list = []
        test_cases_list = []

        self.logger.info(f"[TIMER][batch_decode_end][request_id={request_id}][origin_prompt_id={origin_prompt_id}][time={time.time()}]")


        self.logger.info(f"non_tensor_batch keys: {list(data.non_tensor_batch.keys())}")

        if "reference_time" in data.non_tensor_batch:
            reference_time_list = data.non_tensor_batch["reference_time"]
        else:
            reference_time_list = [None] * len(prompts_text_list)

        for i, (prompt_id, prompt_txt, response, case_type, test_cases, test_case_function, tag, reference_time) in enumerate(
            zip(
                data.non_tensor_batch["id"],
                prompts_text_list,
                response_text_list,
                data.non_tensor_batch["case_type"],
                data.non_tensor_batch["test_cases"],
                data.non_tensor_batch["test_case_function"],
                data.non_tensor_batch["tag"],
                reference_time_list,
            )
        ):
            if isinstance(test_cases, str):
                test_cases = json.loads(test_cases)
            
            import os
            self.logger.info(f"ADAPTIVE_TIMEOUT: {os.environ.get('ADAPTIVE_TIMEOUT', '0')}")
            self.logger.info(f"reference_time: {reference_time}")
            self.logger.info(f"reference_time_list: {reference_time_list}")
            if os.environ.get("ADAPTIVE_TIMEOUT", "0") == "1" and reference_time is not None:
                timeout = min(max(2 * reference_time, 2), FixedTimeOut)
                self.logger.info(f"timeout: {timeout}")
            else:
                timeout = FixedTimeOut
                self.logger.info(f"timeout: {timeout}")

            if "livecodebench" in tag or self.use_local:
                start_time = time.time()
                # self.timer.update_timestamp(request_id, prompt_id, "start_compute_reward", start_time)
                self.logger.info(f"[TIMER][cal_local_test_start][request_id={request_id}][origin_prompt_id={origin_prompt_id}][time={time.time()}]")
                correct, info, error_info = cal_local_test(
                    global_step, prompt_id, prompt_txt, response, case_type, test_cases, test_case_function, timeout=timeout
                )
                self.logger.info(f"[TIMER][cal_local_test_end][request_id={request_id}][origin_prompt_id={origin_prompt_id}][time={time.time()}]")
                end_time = time.time()
                start_compute_reward_list.append(start_time)
                end_compute_reward_list.append(end_time)
                reward_score_list.append(correct)

                # self.timer.update_timestamp(request_id, prompt_id, "end_compute_reward", compute_rewards_end)
                # self.timer.update_kv(request_id, "reward_score", correct)

            else:
                correct, info, error_info = cal_http_sandbox(
                    global_step, prompt_id, prompt_txt, response, case_type, test_cases, self.url
                )
            info['tag'] = tag
            self.logger.debug(f"{json.dumps(info, ensure_ascii=False)}")

            error_infos.append(error_info)
            verify_answer.append(correct)
            format_validations.append(info.get("format_validation", 0))
            have_thinks.append(info.get("have_think", 0))
            test_cases_list.append(len(test_cases))
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        response_level_rewards = torch.tensor(verify_answer, dtype=torch.float16)
        scores = torch.tensor(verify_answer, dtype=torch.float16)
        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": response_level_rewards,
                "scores": scores,
                # "start_compute_reward_list": torch.tensor(start_compute_reward_list, dtype=torch.float64),
                # "end_compute_reward_list": torch.tensor(end_compute_reward_list, dtype=torch.float64),
                # "reward_score_list": torch.tensor(reward_score_list, dtype=torch.float16),
                # "test_cases_list": torch.tensor(test_cases_list, dtype=torch.int32),
            }
        )

        self.logger.info(f"[TIMER][compute_rewards_codesandbox_end][request_id={request_id}][origin_prompt_id={origin_prompt_id}][time={time.time()}][scores={scores}][error_infos={error_infos}]")
        return output

    # def get_timer(self):
    #     return self.timer

    # def clear_timer(self):
    #     self.timer.clear_timer()
