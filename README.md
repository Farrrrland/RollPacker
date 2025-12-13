<div align="center">

<!-- <img src="assets/roll.jpeg" width="40%" alt="ROLL Logo"> -->

# RollPacker: Taming Long-Tail Rollouts for RL Post-Training with Tail Batching

Artifact Evaluation Guidelines for NSDI'26 [Paper #677](./files/nsdi26fall-paper677.pdf)

</div>

## Testbed Environment and Experiments Description
We provide you with scripts and configurations to reproduce key results in our paper, including:
- End-to-end evaluation results in Figure 9 and 10.
- Sensitive analysis in Figure 11.
- Micro-benchmarks in Figure 13(b-c) and Table 3.

*Note 1*: The evaluation results shown in the paper were conducted on NVIDIA H800 clusters with 16-128 GPUs.
Providing such infrastructure is extremely expensive, so we only provide you with resources of *eight H800 GPUs* to train a **Qwen2.5-7B** model with 8K response length, which is the minimal model size exhibited in the paper.
Micro-benchmarks in Figure 12, as well as the large-scale analysis in Figure 14, are not provided, since they require larger model sizes and more resources.
Reviewers with access to large-scale resources may replicate these experiments.

*Note 2*: The minimal scale of the evaluation results in our paper is performed with 16 H800 GPUs, while we can only provide eight H800 GPUs; the absolute performance will not be an exact match, but they provide approximately the same trends.

*Note 3*: We are still organizing available resources to provide you with for artifact evaluation. The final API of our provided resource is still TBA.



## Environment Setup

### Option #1: Using our provided Docker image.
We stronly recommend using our provided docker images, thus no additional environment setup is required.
```bash
# Docker image TBD.
```
You will find the source code located in `/root/code/RollPacker` of the Docker image.

### Option #2: Build your own environment based on ROLL.
`RollPacker` is build on [ROLL](https://github.com/alibaba/ROLL). So you can follow ROLL's [set up instructions](https://github.com/alibaba/ROLL?tab=readme-ov-file#-get-started) to build your own environment for the **RLVR Pipeline**.

## Get Start

