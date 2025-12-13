import asyncio
import subprocess
import json
import os
import time
from datetime import datetime


import re
import subprocess

import subprocess

def get_sm_utilization_once():
    proc = subprocess.Popen(
        ['nvidia-smi', 'dmon', '-s', 'ut', '-o', 'T'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    sm_data = []
    header_lines = 0
    count = 0
    # dmon输出前两行为表头，后面每行为每块卡
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        if line.startswith('#'):
            header_lines += 1
            continue
        if line:  # 数据行
            parts = line.split()
            if len(parts) >= 3:
                sm_data.append({
                    'index': parts[1],
                    'sm_utilization': int(parts[2])
                })
            count += 1
        # 通常卡数个数据行，全部读到就够了
        # 这里以系统有8卡为例：
        if count >= 8:  # 你的卡数多少就写多少，或根据实际行判断break
            break
    proc.terminate()
    return sm_data



def _parse_nvidia_smi_output(output):
    gpu_lines = []

    lines = output.strip().splitlines()
    for line in lines:
        # Remove commas and extra spaces
        parts = [field.strip() for field in line.split(',')]

        if len(parts) < 5:
            continue  # skip invalid lines

        try:
            gpu_index = int(parts[0])
            name = parts[1].strip()
            gpu_util = int(parts[2].replace('%', '').strip())
            mem_used = int(parts[3].replace('MiB', '').strip())
            mem_total = int(parts[4].replace('MiB', '').strip())

            gpu_info = {
                "index": str(gpu_index),
                "name": name,
                "utilization.gpu": gpu_util,
                "memory.used": mem_used,
                "memory.total": mem_total
            }
            gpu_lines.append(gpu_info)
        except (ValueError, IndexError) as e:
            print(f"Error parsing line: {line} | Error: {e}")
            continue

    return {"gpus": gpu_lines}


class GPUTracker:
    def __init__(self, interval=1, output_dir=".", filename="gpu_usage_log.json"):
        self.interval = interval
        self.running = False
        self.data_log = []
        self.output_dir = output_dir
        self.filename = filename
        self.full_path = os.path.join(output_dir, filename)

    

    def _get_gpu_info(self):
        # nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
        simple_cmd = ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader']
        simple_result = subprocess.run(simple_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if simple_result.returncode != 0:
            print("Error running simple nvidia-smi query:", simple_result.stderr)
            return None
        nv_smi_output = _parse_nvidia_smi_output(simple_result.stdout)
        length_of_gpu = len(nv_smi_output['gpus'])
        sm_utils = get_sm_utilization_once()[:length_of_gpu]
        for output in nv_smi_output['gpus']: 
            for sm_util in sm_utils: 
                if sm_util['index'] == output['index']: 
                    output['sm_utilization'] = sm_util['sm_utilization']
        

        return nv_smi_output

    async def _monitor_task(self):
        while self.running:
            gpus = self._get_gpu_info()
            if gpus:
                entry = {
                    "timestamp": time.time(),
                    "gpus": gpus
                }
                self.data_log.append(entry)
                # print(f"[{datetime.now()}] Recorded GPU data.")
            else:
                print(f"[{datetime.now()}] Failed to retrieve GPU data.")

            await asyncio.sleep(self.interval)

    async def start(self):
        """ 启动异步监控 """
        if not self.running:
            self.running = True
            self.task = asyncio.create_task(self._monitor_task())
            print("GPU tracker started asynchronously.")

    def stop(self, filename):
        if self.running:
            self.running = False
            self.save_to_file(filename)
            print("GPU tracker stopped and data saved.")

    def save_to_file(self, filename):
        # os.makedirs(self.output_dir, exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.data_log, f, indent=4)
        print(f"Data saved to: {self.full_path}")
        self.data_log = list()

import threading
if __name__ == "__main__":
    
    tracker = GPUTracker(
        interval=0.2,
        output_dir="./logs",
        filename="gpu_usage.json"
    )
    print(tracker._get_gpu_info())
    import pdb; pdb.set_trace()

    loop = asyncio.new_event_loop()

    def run_async():
        asyncio.set_event_loop(loop)
        loop.run_forever()
    for i in range(1): 
        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()

        # 在事件循环中启动监控任务
        asyncio.run_coroutine_threadsafe(tracker.start(), loop)

        print("Main thread continues to run...")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping...")
            tracker.stop(filename='output/log.json')
            loop.call_soon_threadsafe(loop.stop)
            thread.join()
