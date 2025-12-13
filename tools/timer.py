class Timer:
    def __init__(self):
        self._timings = {}

    def update_timestamp(self, request_id, prompt_id, key, time_stamp):
        """
        Update the timing information for a given prompt_id and key.
        _timings is a dict of the form:
        {
            request_id: {
                key1: time_stamp1,
                key2: time_stamp2,
                ...
            },
            ...
        }
        """
        if request_id not in self._timings:
            self._timings[request_id] = {}
        self._timings[request_id][key] = time_stamp
        if "prompt_id" not in self._timings[request_id]:
            self._timings[request_id]["prompt_id"] = prompt_id
        else:
            assert self._timings[request_id]["prompt_id"] == prompt_id

    def update_gpu_work_id(self, request_id, gpu_work_id):
        if request_id not in self._timings:
            self._timings[request_id] = {}
        self._timings[request_id]["gpu_work_id"] = gpu_work_id

    def update_reward_worker_id(self, request_id, reward_worker_id):
        if request_id not in self._timings:
            self._timings[request_id] = {}
        self._timings[request_id]["reward_worker_id"] = reward_worker_id
    
    def update_kv(self, request_id, key, value):
        if request_id not in self._timings:
            self._timings[request_id] = {}
        self._timings[request_id][key] = value

    def clear_timer(self):
        self._timings = {}