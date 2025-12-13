# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import contextlib
from typing import Iterator, List, Union
import os 
import torch
import torch.distributed as dist
from torch.distributed import _coalescing_manager
import warnings
from contextlib import nullcontext

from megatron.core.utils import is_torch_min_version
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBucketGroup, shard_buffer


if is_torch_min_version("1.13.0"):
    dist_all_gather_func = torch.distributed.all_gather_into_tensor
    dist_reduce_scatter_func = torch.distributed.reduce_scatter_tensor
else:
    dist_all_gather_func = torch.distributed._all_gather_base
    dist_reduce_scatter_func = torch.distributed._reduce_scatter_base



def start_grad_sync(self):
    """
    Initiates grad sync (all-reduce or reduce-scatter) communication operations
    for all buckets in the bucket group.

    When ddp_config.overlap_grad_reduce is set to True, dispatches an asynchronous
    communication call. When ddp_config.overlap_grad_reduce is set to False, makes
    synchronous call.
    """
    
    if (eval(os.environ.get('ddp_no_async', "False"))): 
        # for idx, bucket in enumerate(self.buckets):
        #     print(f"autoscaling state, idx is {idx}, rank is {dist.get_rank()}, bucket norm {bucket.grad_data.norm()}, max {bucket.grad_data.max()}, mean {bucket.grad_data.abs().mean()}, shape {bucket.grad_data.shape}", flush=True)
        return
    assert (
        self.grad_reduce_handle is None
    ), 'Should not have multiple communication calls outstanding at once'

    if self.ddp_config.check_for_nan_in_grad or self.ddp_config.check_for_large_grads:
        self.check_grads(
            check_for_nan_or_inf=self.ddp_config.check_for_nan_in_grad,
            check_for_large=self.ddp_config.check_for_large_grads,
        )

    # gradient_scaling_factor already takes into account whether we are computing
    # an average or sum in the data-parallel collective.
    for bucket in self.buckets:
        if bucket.gradient_scaling_factor != 1.0:
            bucket.grad_data *= bucket.gradient_scaling_factor

    # Decide reduce_op.
    reduce_op = torch.distributed.ReduceOp.SUM
    if self.ddp_config.average_in_collective:
        reduce_op = torch.distributed.ReduceOp.AVG

    # We use the following stream synchronization for the gradient reduction
    # within and across DistOpt instances.

    # Compute Stream: -------------Gradient compute-------------------
    # Comm. Stream:   ------(wait for NCCL)-----(wait for NCCL)-------
    # NCCL Stream:          -------RS------     -------AR------

    # Use async communications only when overlap_grad_reduce is True.
    async_op = (
        self.ddp_config.overlap_grad_reduce
        and self.ddp_config.num_distributed_optimizer_instances == 1
    )
    if (
        self.ddp_config.num_distributed_optimizer_instances > 1
        and self.ddp_config.overlap_grad_reduce
    ):
        # Assign a communication stream if we have multiple DistOpt instances and we
        # need to overlap communication.
        stream_context = torch.cuda.stream(self.communication_stream)

        # The RS/AR communication stream needs to wait for the default stream
        # to complete its gradient computation before launching the next
        # gradient reduction collective.
        self.communication_stream.wait_stream(torch.cuda.default_stream())
    else:
        stream_context = nullcontext()

    if self.ddp_config.use_distributed_optimizer:
        communication_group = self.intra_distributed_optimizer_instance_group
    else:
        communication_group = self.data_parallel_group

    if eval(os.environ.get('ddp_no_async', "False")): 
        if not hasattr(self, 'second_half_data_parallel_group'):
            all_ranks = sorted(torch.distributed.get_process_group_ranks(communication_group))
            mid = len(all_ranks) // 2
            self.sub_ranks = all_ranks[mid:]
            self.second_half_data_parallel_group = dist.new_group(ranks=self.sub_ranks, backend='nccl')
        communication_group = self.second_half_data_parallel_group
    

        
    # Coalesce communication kernels across buckets in the bucket group.
    if (eval(os.environ.get('ddp_no_async', "False"))): 
        for bucket in self.buckets:
            torch.distributed.all_reduce(
                bucket.grad_data, op=reduce_op, group=communication_group, async_op=async_op
            )
        dist.barrier(communication_group)
    else: 
        with stream_context, _coalescing_manager(communication_group, async_ops=async_op) as cm:
            for bucket in self.buckets:
                if self.ddp_config.use_distributed_optimizer:
                    local_data_view = shard_buffer(
                        bucket.grad_data, self.intra_distributed_optimizer_instance_size
                    )[self.intra_distributed_optimizer_instance_rank]
                    dist_reduce_scatter_func(
                        local_data_view,
                        bucket.grad_data,
                        op=reduce_op,
                        group=communication_group,
                        async_op=async_op,
                    )
                else:
                    torch.distributed.all_reduce(
                        bucket.grad_data, op=reduce_op, group=communication_group, async_op=async_op
                    )

    # With multiple DistOpt instances, we need to all-reduce across instances.
    if (
        self.ddp_config.use_distributed_optimizer
        and self.ddp_config.num_distributed_optimizer_instances > 1
    ):

        assert self.inter_distributed_optimizer_instance_group is not None
        # Create a new coalescing manager for the inter-instance all-reduce.
        with stream_context, _coalescing_manager(
            self.inter_distributed_optimizer_instance_group, async_ops=async_op
        ) as cm:
            for bucket in self.buckets:
                local_data_view = shard_buffer(
                    bucket.grad_data, self.intra_distributed_optimizer_instance_size
                )[self.intra_distributed_optimizer_instance_rank]

                torch.distributed.all_reduce(
                    local_data_view,
                    op=reduce_op,
                    group=self.inter_distributed_optimizer_instance_group,
                    async_op=async_op,
                )

    if async_op:
        self.grad_reduce_handle = cm
    else:
        # When using `_coalescing_manager`, even if a synchronous op (async_op=False) is used,
        # `cm` is not None, which is different from when `_coalescing_manager` is not used in
        # which case the torch.distributed._reduce_scatter_base() will return None. In order to
        # maintain consistency with prior code, we need to manually set communication handle to
        # None.
        self.grad_reduce_handle = None


_ParamAndGradBucketGroup.start_grad_sync = start_grad_sync

from megatron.core.transformer.cuda_graphs import is_graph_capturing
from megatron.core import DistributedDataParallel

def _make_backward_post_hook(self, param: torch.nn.Parameter):
    """
    Creates a backward post-hook to dispatch an all-reduce / reduce-scatter when
    ready (i.e., when all grads in a bucket have been computed in all microbatches
    in a batch).
    """

    def hook(*unused):
        if is_graph_capturing():
            return

        if param in self.param_to_bucket_group:
            assert param.requires_grad
            if self.ddp_config.overlap_grad_reduce:
                assert (
                    param.grad is not None
                ), 'param.grad being None is not safe when overlap_grad_reduce is True'
            if param.grad is not None and (
                not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
            ):
                param.main_grad.add_(param.grad.data)
            param.grad = None
            grad_sum = 0
            for sub_param, bucket_group in self.param_to_bucket_group.items(): 
                import pdb; pdb.set_trace()
                for sub_bucket in bucket_group.buckets: 
                    grad_sum += sub_bucket.grad_data.norm().item()
            print(f"rank is {dist.get_rank()}, gradm sum {grad_sum}", flush=True)
            if self.ddp_config.overlap_grad_reduce:
                self.param_to_bucket_group[param].register_grad_ready(param)

    return hook

DistributedDataParallel.__make_backward_post_hook = _make_backward_post_hook