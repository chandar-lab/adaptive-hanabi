#!/bin/bash

source /home/mila/n/nekoeiha/anaconda3/bin/activate
conda activate obl
module load gcc/9.3.0
module load cuda/10.2/cudnn/7.6
export PYTHONPATH=/home/mila/n/nekoeiha/MILA/hanabi/off-belief-learning/:$PYTHONPATH
export LD_LIBRARY_PATH=/cvmfs/ai.mila.quebec/apps/x86_64/debian/gcc/9.3.0/lib64
export CUDNN_LIBRARY_PATH=/cvmfs/ai.mila.quebec/apps/arch/common/cudnn/10.2-v7.6/lib64/
export OMP_NUM_THREADS=1
# iql-op0-lstmlayers1-seed100/model_epoch1000.pthw
python adapt.py \
       --save_dir /home/mila/n/nekoeiha/scratch/hanabi_exps/test_adaptation \
       --load_model /home/mila/n/nekoeiha/scratch/hanabi_exps/test_cross_play/icml_OBL/icml_OBL5/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw \
       --coop_agents /home/mila/n/nekoeiha/scratch/hanabi_exps/test_cross_play/icml_OBL/icml_OBL5/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw \
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

#seeds=(102) # 6 7 8 9
#methods=('vdn')
#num_lstm_layers=(1 2)
#shuffle_colors=(0 1)
#
#echo 'Submitting SBATCH jobs...'
#
#for seed in ${seeds[@]}
#do
#  for method in ${methods[@]}
#  do
#    for num_lstm_layer in ${num_lstm_layers[@]}
#    do
#      for shuffle_color in ${shuffle_colors[@]}
#      do
#        echo "#!/bin/bash" >> temprun.sh
#        echo "#SBATCH --partition=long"  >> temprun.sh
#        echo "#SBATCH --cpus-per-task=2" >> temprun.sh
#        echo "#SBATCH --gres=gpu:2" >> temprun.sh
#        echo "#SBATCH --mem=128G" >> temprun.sh
#        echo "#SBATCH --time=71:59:00" >>  temprun.sh
#        echo "#SBATCH -o /home/mila/n/nekoeiha/scratch/hanabi_exps/logs/slurm-%j.out" >> temprun.sh
#
#
#        echo "source /home/mila/n/nekoeiha/anaconda3/bin/activate" >> temprun.sh
#        echo "conda activate obl" >> temprun.sh
#        echo "module load gcc/9.3.0" >> temprun.sh
#        echo "module load cuda/10.2/cudnn/7.6" >> temprun.sh
#
#        echo "export PYTHONPATH=/home/mila/n/nekoeiha/MILA/hanabi/off-belief-learning/:$PYTHONPATH" >> temprun.sh
#        echo "export LD_LIBRARY_PATH=/cvmfs/ai.mila.quebec/apps/x86_64/debian/gcc/9.3.0/lib64" >> temprun.sh
#        echo "export CUDNN_LIBRARY_PATH=/cvmfs/ai.mila.quebec/apps/arch/common/cudnn/10.2-v7.6/lib64/" >> temprun.sh
#        echo "export OMP_NUM_THREADS=1" >> temprun.sh
#
#        echo "python selfplay.py \
#             --save_dir /home/mila/n/nekoeiha/scratch/hanabi_exps/${method}-sad1-op${shuffle_color}-lstmlayers${num_lstm_layer}-seed${seed} \
#             --num_thread 80 \
#             --num_game_per_thread 80 \
#             --method ${method} \
#             --shuffle_color ${shuffle_color} \
#             --sad 1 \
#             --lr 6.25e-05 \
#             --eps 1.5e-05 \
#             --gamma 0.999 \
#             --seed ${seed} \
#             --burn_in_frames 10000 \
#             --replay_buffer_size 100000 \
#             --batchsize 128 \
#             --epoch_len 1000 \
#             --num_epoch 2000 \
#             --num_player 2 \
#             --net lstm \
#             --num_lstm_layer ${num_lstm_layer} \
#             --multi_step 3 \
#             --train_device cuda:0 \
#             --act_device cuda:1 " >> temprun.sh #
#
#        eval "sbatch temprun.sh"
#        rm temprun.sh
#      done
#    done
#  done
#done
#
#
