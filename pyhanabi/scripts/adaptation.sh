#!/bin/bash

python adapt.py \
       --save_dir <save_dir> \
       --load_model <learner_model_dir> \
       --coop_agents <cooperative_partners_dir> \
       --num_thread 80 \
       --num_game_per_thread 80 \
       --method iql \
       --mode br \
       --shuffle_color 0 \
       --sad 0 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --gamma 0.999 \
       --seed 100 \
       --burn_in_frames 1000 \
       --replay_buffer_size 100000 \
       --batchsize 128 \
       --epoch_len 100 \
       --num_epoch 2000 \
       --num_player 2 \
       --num_lstm_layer 2 \
       --multi_step 3 \
       --train_device cuda:0 \
       --act_device cuda:1
