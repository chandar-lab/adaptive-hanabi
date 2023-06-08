#!/bin/bash

seeds=(101) # 102 103) # 6 7 8 9
# pretrained learner
agent1s=("iql-op0-lstmlayers1-seed100/model_epoch1000.pthw" "iql-op1-lstmlayers1-seed100/model_epoch1000.pthw" "vdn-op0-lstmlayers1-seed100/model_epoch1000.pthw" "vdn-op1-lstmlayers1-seed100/model_epoch1000.pthw" "icml_OBL/icml_OBL5/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_a/model0.pthw")
# eval partner
# medium diversity (collas main results)
#agent2s=("iql-op0-lstmlayers2-seed101/model_epoch1000.pthw" "iql-op1-lstmlayers2-seed100/model_epoch1000.pthw" "vdn-op0-lstmlayers2-seed100/model_epoch1000.pthw" "vdn-op1-lstmlayers2-seed100/model_epoch1000.pthw" "icml_OBL/icml_OBL5/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_d/model0.pthw")
# low diversity
#agent2s=("iql-op0-lstmlayers1-seed101/model_epoch1000.pthw" "vdn-op1-lstmlayers2-seed101/model_epoch1000.pthw") # iql-op0-lstmlayers1-seed100,iql-op0-lstmlayers2-seed101,iql-op1-lstmlayers1-seed100,icml_OBL5/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_d
# high diversity
agent2s=("iql-op1-lstmlayers2-seed101/model_epoch1000.pthw" "icml_OBL/icml_OBL1/OFF_BELIEF1_SHUFFLE_COLOR0_BZA1_BELIEF_a/model0.pthw") # iql-op0-lstmlayers2-seed101,vdn-op1-lstmlayers1-seed100,vdn-op1-lstmlayers2-seed100,icml_OBL5/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_d
# debug below
# agent1s=()
# agent2s=()

lrs=(0.0000625) # 0.001 0.000625 0.0001  0.0000625  0.00001 0.00000625
num_threads=(5) #10 20 40 80 160
for seed in ${seeds[@]}
do
for num_thread in ${num_threads[@]}
do
for lr in ${lrs[@]}
do
  for agent1 in ${agent1s[@]}
  do
    for agent2 in ${agent2s[@]}
    do
      echo "#!/bin/bash" >> temprun.sh
      echo "#SBATCH --partition=long"  >> temprun.sh
      echo "#SBATCH --cpus-per-task=2" >> temprun.sh
      echo "#SBATCH --gres=gpu:rtx8000:2" >> temprun.sh
      echo "#SBATCH --mem=128G" >> temprun.sh
      echo "#SBATCH --time=23:59:00" >>  temprun.sh
      echo "#SBATCH -o ${SCRATCH}/hanabi_exps/logs/slurm-%j.out" >> temprun.sh


      # echo "source /home/mila/n/nekoeiha/anaconda3/bin/activate" >> temprun.sh
      echo "module load anaconda/3" >> temprun.sh
      echo "conda activate hanabi_obl" >> temprun.sh
      echo "module load gcc/9.3.0" >> temprun.sh
      echo "module load cuda/10.2/cudnn/7.6" >> temprun.sh

      echo "export PYTHONPATH=${SCRATCH}/adaptive-hanabi/:$PYTHONPATH" >> temprun.sh
      echo "export LD_LIBRARY_PATH=/cvmfs/ai.mila.quebec/apps/x86_64/debian/gcc/9.3.0/lib64" >> temprun.sh
      echo "export CUDNN_LIBRARY_PATH=/cvmfs/ai.mila.quebec/apps/arch/common/cudnn/10.2-v7.6/lib64/" >> temprun.sh
      echo "export OMP_NUM_THREADS=1" >> temprun.sh

      echo "python ${SCRATCH}/adaptive-hanabi/pyhanabi/adapt.py \
         --save_dir ${SCRATCH}/hanabi_exps/test_adaptation/seed${seed}-${agent1}-${agent2} \
         --load_model ${SCRATCH}/hanabi_exps/test_cross_play/${agent1} \
         --coop_agents ${SCRATCH}/hanabi_exps/test_cross_play/${agent2} \
         --num_thread 10 \
         --num_game_per_thread 40 \
         --method iql \
         --mode br \
         --shuffle_color 0 \
         --sad 0 \
         --lr ${lr} \
         --eps 1.5e-05 \
         --gamma 0.999 \
         --seed ${seed} \
         --burn_in_frames 10000 \
         --replay_buffer_size 100000 \
         --batchsize 128 \
         --epoch_len 1000 \
         --num_epoch 2000 \
         --num_player 2 \
         --num_lstm_layer 2 \
         --multi_step 3 \
         --train_device cuda:0 \
         --act_device cuda:1" >> temprun.sh

      eval "sbatch temprun.sh"
      rm temprun.sh
    done
  done
done
done
done