
function train {
    echo experiment_name ${experiment_name}
    echo output_dir ${output_dir}
    mkdir -p ${output_dir}
    python train.py \
        --experiment_name ${experiment_name}\
        --train_dir lung_colon_dataset/train \
        --output_dir ${output_dir} \
        --remove_unused_columns False \
        --label_column_name label \
        --do_train \
        --do_eval \
        --learning_rate 2e-5 \
        --num_train_epochs 2 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --logging_strategy steps \
        --logging_steps 10 \
        --eval_strategy steps \
        --eval_steps 200 \
        --save_strategy epoch \
        --load_best_model_at_end False \
        --save_total_limit 3 \
        --seed 1337 \
        --ignore_mismatched_sizes True \
        --report_to wandb &> ${output_dir}/train.log
}


function train_linear_prob {
    output_dir=${output_dir}_linear_prob
    echo experiment_name ${experiment_name}
    echo output_dir ${output_dir}
    mkdir -p ${output_dir}
    python train.py \
        --experiment_name ${experiment_name}\
        --linear_prob True\
        --train_dir lung_colon_dataset/train \
        --output_dir ${output_dir} \
        --remove_unused_columns False \
        --label_column_name label \
        --do_train \
        --do_eval \
        --learning_rate 2e-5 \
        --num_train_epochs 2 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --logging_strategy steps \
        --logging_steps 10 \
        --eval_strategy steps \
        --eval_steps 200 \
        --save_strategy epoch \
        --load_best_model_at_end False \
        --save_total_limit 3 \
        --seed 1337 \
        --ignore_mismatched_sizes True \
        --report_to wandb &> ${output_dir}/train.log
}

EXPERIMENTS="
    vit_base
    vit_large
    vit_huge
    mae_base
    mae_large
    mae_huge
    clip_large
    clip_base
    mim_base
    mim_large
    biomedclip_base
"

for experiment in $EXPERIMENTS;
do
    echo Running ${experiment}
    experiment_name=${experiment} output_dir=./outputs/train_${experiment} \
        train
done



EXPERIMENTS="
    vit_base
    vit_large
    vit_huge
    mae_base
    mae_large
    mae_huge
    clip_large
    clip_base
    mim_base
    mim_large
    biomedclip_base
"

for experiment in $EXPERIMENTS;
do
    echo Running ${experiment}
    experiment_name=${experiment} output_dir=./outputs/train_${experiment} \
        train_linear_prob
done
