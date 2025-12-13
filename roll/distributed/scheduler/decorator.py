"""
ref: https://github.com/volcengine/verl/blob/main/single_controller/base/decorator.py
"""

import gc
import os
import traceback
from enum import Enum, auto
from functools import wraps, partial
from itertools import chain
from typing import Tuple, List, Dict
from more_itertools import chunked
import ray
import torch
import asyncio

from roll.distributed.scheduler.protocol import DataProto, ObjectRefWrap
from roll.utils.logging import get_logger

logger = get_logger()

BIND_WORKER_METHOD_FLAG = "BIND_WORKER_METHOD_FLAG"


class Dispatch(Enum):
    """
    dispatch 负责处理Cluster的输入list如何拆分分配到各worker上
    """

    ONE_TO_ALL = auto()
    ONE_TO_ALL_ONE = auto()
    ALL_TO_ALL = auto()
    DP_MP_COMPUTE = auto()
    DP_MP_DISPATCH_FIRST = auto()
    ONE_TO_FIRST_HALF = auto()
    ONE_TO_SECOND_HALF = auto()
    ONE_TO_HALF = auto()
    DP_MP_COMPUTE_FIRST_HALF = auto()
    DP_MP_COMPUTE_SECOND_HALF = auto()
    ONE_TO_ALL_ONE_SECOND_HALF = auto()
    ONE_TO_ALL_ONE_FIRST_HALF = auto()
    DP_MP_DISPATCH_FIRST_SECOND_HALF = auto()
    DP_MP_DISPATCH_FIRST_COLLECT_ALL = auto()
    ONE_TO_RANKS = auto()


class Execute(Enum):
    ALL = 0
    RANK_ZERO = 1
    FIRST_HALF = 2
    SECOND_HALF = 3
    RANK = 4


SPECIAL_KEYS = ['_ranks']

def _split_args_kwargs(chunks, *args, **kwargs):
    """
    arg: List, 将List分成dp份
    """

    def split(arg, chunks):
        if isinstance(arg, list):
            return list(chunked(arg, len(arg) // chunks))
        else:
            assert hasattr(arg, "chunk"), f"Argument {arg} does not have a 'chunk' method."
            # import pdb; pdb.set_trace() 
            return arg.chunk(chunks=chunks)

    splitted_args = []
    for arg in args:
        splitted_args.append(split(arg, chunks))

    splitted_kwargs = {}
    for key, val in kwargs.items():
        if key in SPECIAL_KEYS: 
            splitted_kwargs[key] = val
        else:
            splitted_kwargs[key] = split(val, chunks)

    return splitted_args, splitted_kwargs


def dispatch_one_to_all(cluster, *args, **kwargs):
    """
    假定输入arg是一个值，分发到所有的worker上
    """
    args = tuple([arg] * cluster.world_size for arg in args)
    kwargs = {k: [v] * cluster.world_size for k, v in kwargs.items()}
    return args, kwargs

def dispatch_one_to_first_half(worker_group, *args, **kwargs):
    # 获取目标 ranks
    assert "_ranks" in kwargs, f"_ranks is not in kwargs"
    target_ranks = kwargs.get("_ranks", list(range(worker_group.world_size // 2)))
    args = tuple([arg] * len(target_ranks) for arg in args)
    kwargs = {k: [v] * len(target_ranks) for k, v in kwargs.items()}
    return args, kwargs

def dispatch_one_to_half(worker_group, *args, **kwargs):
    # 获取目标 ranks
    assert "_ranks" in kwargs, f"_ranks is not in kwargs"
    target_ranks = kwargs.get("_ranks", list(range(worker_group.world_size // 2)))
    args = tuple([arg] * len(target_ranks) for arg in args)
    kwargs = {k: [v] * len(target_ranks) for k, v in kwargs.items()}
    return args, kwargs


# def dispatch_one_to_second_half(worker_group, *args, **kwargs):
#     args = tuple([arg] * (worker_group.world_size // 2) for arg in args)
#     kwargs = {k: [v] * (worker_group.world_size // 2) for k, v in kwargs.items()}
#     print("dispatch_one_to_second_half is ", worker_group.world_size // 2)
#     return args, kwargs


def dispatch_one_to_second_half(worker_group, *args, **kwargs):
    # 如果没有指定 ranks，默认是后半部分
    assert "_ranks" in kwargs, f"_ranks is not in kwargs"
    target_ranks = kwargs.get("_ranks", list(range(worker_group.world_size // 2, worker_group.world_size)))
    args = tuple([arg] * len(target_ranks) for arg in args)
    kwargs = {k: [v] * len(target_ranks) for k, v in kwargs.items()}
    return args, kwargs


def dispatch_one_to_ranks(worker_group, *args, **kwargs):
    target_ranks = kwargs.get("_ranks", None)
    assert target_ranks is not None, "dispatch_one_to_ranks requires _ranks param to be explicitly provided."
    args = tuple([arg] * len(target_ranks) for arg in args)
    kwargs = {k: [v] * len(target_ranks) for k, v in kwargs.items()}
    return args, kwargs

def collect_ranks_to_one(cluster, output, *args, **kwargs):
    if isinstance(output[0], ray.ObjectRef):
        output_in_dp = []
        for global_rank in range(cluster.world_size):
            output_in_dp.append(ObjectRefWrap(output[global_rank], collected=global_rank == 0))
        return output_in_dp
    return output[0]

def collect_all_to_all(cluster, output):
    """
    collect 所有worker的输出
    """
    assert len(output) == cluster.world_size
    return output

def collect_all_to_half(cluster, output, _ranks=None):
    """
    collect 所有worker的输出
    """
    # assert len(output) == cluster.world_size // 2
    return output


def collect_all_to_one(cluster, output):
    """
    collect 所有worker的输出
    """
    assert len(output) == cluster.world_size

    if isinstance(output[0], ray.ObjectRef):
        output_in_dp = []
        for global_rank in range(cluster.world_size):
            output_in_dp.append(ObjectRefWrap(output[global_rank], collected=global_rank == 0))
        return output_in_dp
    return output[0]


def collect_all_to_one_second_half(cluster, output, _ranks=None):
    """
    collect 所有worker的输出
    """
    assert len(output) == cluster.world_size // 2
    
    if isinstance(output[0], ray.ObjectRef):
        output_in_dp = []
        if _ranks is None:
            for global_rank in range(cluster.world_size // 2, cluster.world_size, 1):
                output_in_dp.append(ObjectRefWrap(output[global_rank], collected=global_rank == cluster.world_size // 2))
        else: 
            for idx, global_rank in enumerate(_ranks):
                output_in_dp.append(ObjectRefWrap(output[idx], collected=global_rank == _ranks[0]))
        return output_in_dp # FIXME: 这里的output_in_dp是一个list，里面的每个元素都是ObjectRefWrap
    return output[0]


def collect_all_to_one_first_half(cluster, output, _ranks=None):
    """
    collect 所有worker的输出
    """

    if isinstance(output[0], ray.ObjectRef):
        output_in_dp = []
        if _ranks is None:
            for global_rank in range(cluster.world_size // 2):
                output_in_dp.append(ObjectRefWrap(output[global_rank], collected=global_rank == 0))
        else:
            for idx, global_rank in enumerate(_ranks):
                output_in_dp.append(ObjectRefWrap(output[idx], collected=global_rank == 0))
        return output_in_dp # FIXME: 这里的output_in_dp是一个list，里面的每个元素都是ObjectRefWrap
    return output[0]


def dispatch_all_to_all(cluster, *args, **kwargs):
    """
    假定输入arg是List, len(arg) = cluster.world_size
    """
    for arg in args:
        assert isinstance(arg, (Tuple, List)) and len(arg) == cluster.world_size
    for k, v in kwargs.items():
        assert isinstance(v, (Tuple, List)) and len(v) == cluster.world_size
    return args, kwargs


def _dispatch_dp_mp_compute(cluster, _dispatch_first, *args, **kwargs):
    """
    将输入chunk成dp_world_size份，按dp_rank为每个worker组织数据 -> 同一dp_rank收到的数据都是相同的
    """
    splitted_args, splitted_kwargs = _split_args_kwargs(cluster.dp_size, *args, **kwargs)
    all_args = []

    def get_arg_by_rank_info(arg, rank_info):
        local_dp_rank = rank_info.dp_rank
        if (
            _dispatch_first
            and isinstance(arg[local_dp_rank], DataProto)
            and not (rank_info.tp_rank == 0 and rank_info.cp_rank == 0 and rank_info.pp_rank == 0)
        ):
            return DataProto(batch=None, meta_info=arg[local_dp_rank].meta_info)
        return arg[local_dp_rank]

    for arg in splitted_args:
        assert isinstance(arg, (Tuple, List)) and len(arg) == cluster.dp_size
        transformed_args = []
        for i in range(cluster.world_size):
            local_rank_info = cluster.get_rank_info(rank=i)
            transformed_args.append(get_arg_by_rank_info(arg, local_rank_info))
        all_args.append(transformed_args)
    all_args = tuple(all_args)

    all_kwargs = {}
    for k, v in splitted_kwargs.items():
        assert isinstance(v, (Tuple, List)) and len(v) == cluster.dp_size
        transformed_v = []
        for i in range(cluster.world_size):
            local_rank_info = cluster.get_rank_info(rank=i)
            transformed_v.append(get_arg_by_rank_info(v, local_rank_info))
        all_kwargs[k] = transformed_v
    return all_args, all_kwargs


def _dispatch_dp_mp_compute_first_half(cluster, _dispatch_first, target_ranks, *args, **kwargs):
    """
    将输入chunk成dp_world_size份，按dp_rank为每个worker组织数据 -> 同一dp_rank收到的数据都是相同的
    """
    split_size = int(cluster.dp_size * len(target_ranks) // cluster.world_size)
    splitted_args, splitted_kwargs = _split_args_kwargs(split_size, *args, **kwargs)
    all_args = []

    def get_arg_by_rank_info(arg, rank_info):
        local_dp_rank = rank_info.dp_rank
        if (
            _dispatch_first
            and isinstance(arg[local_dp_rank], DataProto)
            and not (rank_info.tp_rank == 0 and rank_info.cp_rank == 0 and rank_info.pp_rank == 0)
        ):
            return DataProto(batch=None, meta_info=arg[local_dp_rank].meta_info)
        return arg[local_dp_rank]

    for arg in splitted_args:
        assert isinstance(arg, (Tuple, List)) and len(arg) == split_size
        transformed_args = []
        for i in range(len(target_ranks)):
            local_rank_info = cluster.get_rank_info(rank=target_ranks[i]-target_ranks[0])
            transformed_args.append(get_arg_by_rank_info(arg, local_rank_info))
        all_args.append(transformed_args)
    all_args = tuple(all_args)

    all_kwargs = {}
    for k, v in splitted_kwargs.items():
        if k in SPECIAL_KEYS: 
            all_kwargs[k] = v
            continue
        assert isinstance(v, (Tuple, List)) and len(v) == split_size
        transformed_v = []
        for i in range(len(target_ranks)):
            local_rank_info = cluster.get_rank_info(rank=target_ranks[i]-target_ranks[0])
            transformed_v.append(get_arg_by_rank_info(v, local_rank_info))
        all_kwargs[k] = transformed_v
    return all_args, all_kwargs


def _dispatch_dp_mp_compute_second_half(cluster, _dispatch_first, target_ranks, *args, **kwargs):
    """
    将输入chunk成dp_world_size份，按dp_rank为每个worker组织数据 -> 同一dp_rank收到的数据都是相同的
    """
    split_size = int(cluster.dp_size * len(target_ranks) // cluster.world_size)
    splitted_args, splitted_kwargs = _split_args_kwargs(split_size, *args, **kwargs)
    all_args = []

    def get_arg_by_rank_info(arg, rank_info):
        local_dp_rank = rank_info.dp_rank
        if (
            _dispatch_first
            and isinstance(arg[local_dp_rank], DataProto)
            and not (rank_info.tp_rank == 0 and rank_info.cp_rank == 0 and rank_info.pp_rank == 0)
        ):
            return DataProto(batch=None, meta_info=arg[local_dp_rank].meta_info)
        print(f"loca_dp rank {local_dp_rank}, rank_info {rank_info}", flush=True)
        return arg[local_dp_rank]

    for arg in splitted_args:
        assert isinstance(arg, (Tuple, List)) and len(arg) == split_size
        transformed_args = []
        # for i in range(cluster.world_size // 2):
        for i in range(len(target_ranks)):
            # local_rank_info = cluster.get_rank_info(rank=i+target_ranks[0])
            local_rank_info = cluster.get_rank_info(rank=target_ranks[i]-target_ranks[0])
            transformed_args.append(get_arg_by_rank_info(arg, local_rank_info))
        all_args.append(transformed_args)
    all_args = tuple(all_args)

    all_kwargs = {}
    for k, v in splitted_kwargs.items():
        if k in SPECIAL_KEYS: 
            all_kwargs[k] = v
            continue
        assert isinstance(v, (Tuple, List)) and len(v) == split_size
        transformed_v = []
        # for i in range(cluster.world_size // 2):
        for i in range(len(target_ranks)):
            # local_rank_info = cluster.get_rank_info(rank=i+target_ranks[0])
            local_rank_info = cluster.get_rank_info(rank=target_ranks[i]-target_ranks[0])
            transformed_v.append(get_arg_by_rank_info(v, local_rank_info))
        all_kwargs[k] = transformed_v
    
    return all_args, all_kwargs


def dispatch_dp_mp_compute(cluster, *args, **kwargs):
    return _dispatch_dp_mp_compute(cluster, False, *args, **kwargs)


def dispatch_dp_mp_dispatch_first(cluster, *args, **kwargs):
    return _dispatch_dp_mp_compute(cluster, True, *args, **kwargs)


def dispatch_dp_mp_compute_first_half(cluster, *args, **kwargs):
    half_world_size = cluster.world_size // 2
    assert "_ranks" in kwargs, "_ranks is required for dispatch_dp_mp_compute_first_half"
    target_ranks = kwargs.get("_ranks", list(range(half_world_size)))
    return _dispatch_dp_mp_compute_first_half(cluster, False, target_ranks, *args, **kwargs)

def dispatch_dp_mp_compute_second_half(cluster, *args, **kwargs):
    half_world_size = cluster.world_size // 2
    assert "_ranks" in kwargs, "_ranks is required for dispatch_dp_mp_compute_second_half"
    target_ranks = kwargs.get("_ranks", list(range(half_world_size, cluster.world_size)))
    return _dispatch_dp_mp_compute_second_half(cluster, False, target_ranks, *args, **kwargs)


def collect_dp_mp_compute_first_half(cluster, output, _ranks=None):
    """
    只需要搜集tp=0, pipeline_last_stage的结果
    输入输出都是list, 是batch维度的
    """
    assert _ranks is not None, "_ranks must be provided for collect_dp_mp_compute_first_half"
    output_in_dp = []
    for idx, global_rank in enumerate(_ranks):
        local_rank_info = cluster.get_rank_info(rank=global_rank)
        if local_rank_info.tp_rank == 0 and local_rank_info.is_pipeline_last_stage and local_rank_info.cp_rank == 0:
            output_in_dp.append(output[idx])
    if isinstance(output[0], list):
        return list(chain.from_iterable(output_in_dp))
    elif isinstance(output[0], DataProto):
        return DataProto.concat(output_in_dp)
    elif isinstance(output[0], ray.ObjectRef):
        # 处理block=False情况下，dp内的可能完成时间不一致问题
        output_in_dp = []
        # for global_rank in range(cluster.world_size // 2):
        for idx, global_rank in enumerate(_ranks):
            local_rank_info = cluster.get_rank_info(rank=global_rank)
            collected = False
            if (
                local_rank_info.tp_rank == 0
                and local_rank_info.is_pipeline_last_stage
                and local_rank_info.cp_rank == 0
            ):
                collected = True
            # output_in_dp.append(ObjectRefWrap(output[global_rank], collected=collected))
            output_in_dp.append(ObjectRefWrap(output[idx], collected=collected))
        return output_in_dp
    else:
        raise NotImplementedError(f"output type {type(output[0])}")


def collect_dp_mp_compute(cluster, output):
    """
    只需要搜集tp=0, pipeline_last_stage的结果
    输入输出都是list, 是batch维度的
    """
    output_in_dp = []
    for global_rank in range(cluster.world_size):
        local_rank_info = cluster.get_rank_info(rank=global_rank)
        if local_rank_info.tp_rank == 0 and local_rank_info.is_pipeline_last_stage and local_rank_info.cp_rank == 0:
            output_in_dp.append(output[global_rank])
    if isinstance(output[0], list):
        return list(chain.from_iterable(output_in_dp))
    elif isinstance(output[0], DataProto):
        return DataProto.concat(output_in_dp)
    elif isinstance(output[0], ray.ObjectRef):
        # 处理block=False情况下，dp内的可能完成时间不一致问题
        output_in_dp = []
        for global_rank in range(cluster.world_size):
            local_rank_info = cluster.get_rank_info(rank=global_rank)
            collected = False
            if (
                local_rank_info.tp_rank == 0
                and local_rank_info.is_pipeline_last_stage
                and local_rank_info.cp_rank == 0
            ):
                collected = True
            output_in_dp.append(ObjectRefWrap(output[global_rank], collected=collected))
        return output_in_dp
    else:
        raise NotImplementedError(f"output type {type(output[0])}")


predefined_dispatch_mode_fn = {
    Dispatch.ONE_TO_ALL: {
        "dispatch_fn": dispatch_one_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.ONE_TO_FIRST_HALF: {
        "dispatch_fn": dispatch_one_to_first_half,
        "collect_fn": collect_all_to_half,
    },
    Dispatch.ONE_TO_HALF: {
        "dispatch_fn": dispatch_one_to_half,
        "collect_fn": collect_all_to_half,
    },
    Dispatch.ONE_TO_ALL_ONE: {
        "dispatch_fn": dispatch_one_to_all,
        "collect_fn": collect_all_to_one,
    },
    Dispatch.ONE_TO_ALL_ONE_SECOND_HALF: {
        "dispatch_fn": dispatch_one_to_second_half,
        "collect_fn": collect_all_to_one_second_half,
    },
    Dispatch.ONE_TO_ALL_ONE_FIRST_HALF: {
        "dispatch_fn": dispatch_one_to_first_half,
        "collect_fn": collect_all_to_one_first_half,
    },
    Dispatch.ALL_TO_ALL: {
        "dispatch_fn": dispatch_all_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.DP_MP_COMPUTE: {
        "dispatch_fn": dispatch_dp_mp_compute,
        "collect_fn": collect_dp_mp_compute,
    },
    Dispatch.DP_MP_COMPUTE_FIRST_HALF: {
        "dispatch_fn": dispatch_dp_mp_compute_first_half,
        "collect_fn": collect_dp_mp_compute_first_half,
    },
    Dispatch.DP_MP_COMPUTE_SECOND_HALF: {
        "dispatch_fn": dispatch_dp_mp_compute_second_half,
        "collect_fn": collect_dp_mp_compute_first_half,
    },
    Dispatch.DP_MP_DISPATCH_FIRST: {
        "dispatch_fn": dispatch_dp_mp_dispatch_first,
        "collect_fn": collect_dp_mp_compute,
    },
    Dispatch.DP_MP_DISPATCH_FIRST_SECOND_HALF: {
        "dispatch_fn": dispatch_dp_mp_compute_second_half,
        "collect_fn": collect_dp_mp_compute_first_half,
    },
    Dispatch.DP_MP_DISPATCH_FIRST_COLLECT_ALL: {
        "dispatch_fn": dispatch_dp_mp_dispatch_first,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.ONE_TO_RANKS: {
        "dispatch_fn": dispatch_one_to_ranks,
        "collect_fn": collect_ranks_to_one,
    },
}


def get_predefined_dispatch_fn(dispatch_mode):
    return predefined_dispatch_mode_fn[dispatch_mode]


predefined_execute_mode_fn = {
    Execute.ALL: {"execute_fn_name": "execute_all"},
    Execute.RANK_ZERO: {"execute_fn_name": "execute_rank_zero"},
    Execute.FIRST_HALF: {"execute_fn_name": "execute_first_half"},
    Execute.SECOND_HALF: {"execute_fn_name": "execute_second_half"},
    Execute.RANK: {"execute_fn_name": "execute_ranks"},
}


def get_predefined_execute_fn(execute_mode):
    """
    Note that here we only asks execute_all and execute_rank_zero to be implemented
    Leave the choice of how these two functions handle argument 'blocking' to users
    """
    return predefined_execute_mode_fn[execute_mode]


def func_generator(cls, method_name, dispatch_fn, collect_fn, execute_fn):
    def func(*args, blocking=True, **kwargs):
        try:
            if method_name == "initialize":
                setattr(cls, "initialized", True)

            args, kwargs = dispatch_fn(cls, *args, **kwargs)
            output = execute_fn(method_name, *args, **kwargs)
            if blocking:
                timeout = None
                if "roll_RPC_TIMEOUT" in os.environ:
                    timeout = int(os.environ.get("roll_RPC_TIMEOUT"))
                output = ray.get(output, timeout=timeout)
            if '_ranks' in kwargs: 
                output = collect_fn(cls, output, _ranks=kwargs['_ranks'])
            else:
                output = collect_fn(cls, output)
            return output
        except Exception as e:
            print(f"[GEN FUNC ERROR]")
            import traceback; traceback.print_exc()
    return func


def _check_dispatch_mode(dispatch_mode):
    assert isinstance(
        dispatch_mode, (Dispatch, Dict)
    ), f"dispatch_mode must be a Dispatch or a Dict. Got {dispatch_mode}"
    if isinstance(dispatch_mode, Dict):
        necessary_keys = ["dispatch_fn", "collect_fn"]
        for key in necessary_keys:
            assert key in dispatch_mode, f"key {key} should be in dispatch_mode if it is a dictionary"


def _check_execute_mode(execute_mode):
    assert isinstance(execute_mode, Execute), f"execute_mode must be a Execute. Got {execute_mode}"


def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, clear_cache=True):
    _check_dispatch_mode(dispatch_mode)
    _check_execute_mode(execute_mode)

    def decorator(func):
        is_async = asyncio.iscoroutinefunction(func)
        attrs = {"dispatch_mode": dispatch_mode, "execute_mode": execute_mode}
        if is_async:
            @wraps(func)
            async def inner_async(*args, **kwargs):
                try:
                    result = await func(*args, **kwargs)
                    if clear_cache:
                        try:
                            torch._C._cuda_clearCublasWorkspaces()
                            gc.collect()
                            torch.cuda.empty_cache()
                        except Exception as oe:
                            pass

                except Exception as e:
                    logger.error(str(e))
                    logger.error(traceback.format_exc())
                    raise e
                return result

            setattr(inner_async, BIND_WORKER_METHOD_FLAG, attrs)
            return inner_async
        else:
            @wraps(func)
            def inner(*args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    if clear_cache:
                        try:
                            torch._C._cuda_clearCublasWorkspaces()
                            gc.collect()
                            torch.cuda.empty_cache()
                        except Exception as oe:
                            pass

                except Exception as e:
                    logger.error(str(e))
                    logger.error(traceback.format_exc())
                    raise e
                return result

            setattr(inner, BIND_WORKER_METHOD_FLAG, attrs)
            return inner

    return decorator
