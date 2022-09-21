GPU=$1
seed0=31
seed1=32
seed2=33
seed3=34
code=main_imp.py 

data_dir=/home/fsh/MILA/data/
dataset=MNIST_BIASED_0.9_val # MNIST_75sp_0.9 MNIST_75sp_0.95 fashion_0.8 fashion_0.9 fashion_0.95 kuzu_0.8 kuzu_0.9 kuzu_0.95
all_epochs=200
use_mask=1
swap_epochs=100 # 100 with L_G 200 without L_G
lambda_swap=10 # lambda_G
q=0.7
lambda_dis=1


str1="output_GCNII_"$dataset"_q_"$q"_lambda_swap_"$lambda_swap""

tmux new -s DisC_GCNII -d
tmux send-keys "source activate benchmark_gnn" C-m #replace DisC with your environment name

tmux send-keys "
CUDA_VISIBLE_DEVICES=2 \
python -u $code --config 'configs/superpixels_graph_classification_GCNII_MNIST_100k.json' \
--dataset $dataset \
--data_dir $data_dir \
--seed $seed0  \
--mask_epochs $all_epochs \
--swap_epochs $swap_epochs \
--lambda_swap $lambda_swap \
--use_mask $use_mask \
--q $q \
--lambda_dis $lambda_dis \
--out_dir $str1 &

#CUDA_VISIBLE_DEVICES=1 \
#python -u $code --config 'configs/superpixels_graph_classification_GCNII_MNIST_100k.json' \
#--dataset $dataset \
#--data_dir $data_dir \
#--seed $seed1  \
#--mask_epochs $all_epochs \
#--swap_epochs $swap_epochs \
#--lambda_swap $lambda_swap \
#--use_mask $use_mask \
#--q $q \
#--lambda_dis $lambda_dis \
#--out_dir $str1 &

#CUDA_VISIBLE_DEVICES=2 \
#python -u $code --config 'configs/superpixels_graph_classification_GCNII_MNIST_100k.json' \
#--dataset $dataset \
#--data_dir $data_dir \
#--seed $seed2  \
#--mask_epochs $all_epochs \
#--swap_epochs $swap_epochs \
#--lambda_swap $lambda_swap \
#--use_mask $use_mask \
#--q $q \
#--lambda_dis $lambda_dis \
#--out_dir $str1 &

#CUDA_VISIBLE_DEVICES=3 \
#python -u $code --config 'configs/superpixels_graph_classification_GCN_MNIST_100k.json' \
#--dataset $dataset \
#--data_dir $data_dir \
#--seed $seed3  \
#--mask_epochs $all_epochs \
#--swap_epochs $swap_epochs \
#--lambda_swap $lambda_swap \
#--use_mask $use_mask \
#--q $q \
#--lambda_dis $lambda_dis \
#--out_dir $str1 &
wait" C-m
