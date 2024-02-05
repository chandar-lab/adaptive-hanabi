# Adaptive Hanabi

## Introduction

This repo contains the implementation of the benchmark described
in our paper [Towards Few-shot Coordination: Revisiting Ad-hoc Teamplay Challenge In the Game of Hanabi]    (https://proceedings.mlr.press/v232/nekoei23b.html).

The codebase is mostly based on [off-belief-learning](https://github.com/facebookresearch/off-belief-learning) repo. 

`hanabi-learning-environment` is a modified version of the original
[HLE from Deepmind](https://github.com/deepmind/hanabi-learning-environment).


## Environment Setup

Please refer to the setup instruction oon [off-belief-learning](https://github.com/facebookresearch/off-belief-learning) repo. 

## Run the Code

To pre-train hanabi agents, 
```bash
cd pyhanabi
sh scripts/iql.sh
```

To finetune an agent to a pre-traind agent, use the following script:

```bash
cd pyhanabi
sh scripts/adaptation.sh
```

Note that, before running the script, `--load_model` and `--coop_agents` should be specified that shows the path to directory of the learner and cooperative partners checkpoints. 


## Download Models

To download the trained models used in the paper, go to `models` folder and run

```shell
sh download_pool.sh
```