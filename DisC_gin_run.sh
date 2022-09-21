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
swap_epochs=100 # 100 with L_G and 200 without L_G
lambda_swap=15
q=0.7
lambda_dis=1


str1="output_GIN_"$dataset"_q_"$q"_lambda_swap_"$lambda_swap
tmux new -s DisC_GIN -d
tmux send-keys "source activate benchmark_gnn" C-m #replace benchmark_gnn with your environment name

tmux send-keys "
CUDA_VISIBLE_DEVICES=2 \
python -u $code --config 'configs/superpixels_graph_classification_GIN_MNIST_100k.json' \
--dataset $dataset \
--data_dir $data_dir \
--seed $seed0  \
--mask_epochs $all_epochs \
--swap_epochs $swap_epochs \
--use_mask $use_mask \
--q $q \
--lambda_swap $lambda_swap \
--lambda_dis $lambda_dis \
--out_dir $str1 &

#CUDA_VISIBLE_DEVICES=0 \
#python -u $code --config 'configs/superpixels_graph_classification_GIN_MNIST_100k.json' \
#--dataset $dataset \
#--data_dir $data_dir \
#--seed $seed1  \
#--mask_epochs $all_epochs \
#--swap_epochs $swap_epochs \
#--use_mask $use_mask \
#--lambda_swap $lambda_swap \
#--q $q \
#--lambda_dis $lambda_dis \
#--out_dir $str1 &

#CUDA_VISIBLE_DEVICES=1 \
#python -u $code --config 'configs/superpixels_graph_classification_GIN_MNIST_100k.json' \
#--dataset $dataset \
#--data_dir $data_dir \
#--seed $seed2  \
#--mask_epochs $all_epochs \
#--swap_epochs $swap_epochs \
#--use_mask $use_mask \
#--lambda_swap $lambda_swap \
#--q $q \
#--lambda_dis $lambda_dis \
#--out_dir $str1 &

#CUDA_VISIBLE_DEVICES=1 \
#python -u $code --config 'configs/superpixels_graph_classification_GIN_MNIST_100k.json' \
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
