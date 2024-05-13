experiment=scratch
experiment_name=${experiment}
output_dir=./outputs/train_${experiment}_vitbase
mkdir -p ${output_dir}
python train.py \
    --experiment_name ${experiment_name}\
    --config_name google/vit-base-patch16-224-in21k \
    --image_processor_name google/vit-base-patch16-224-in21k \
    --train_dir lung_colon_dataset/train \
    --output_dir ${output_dir} \
    --remove_unused_columns False \
    --label_column_name label \
    --do_train \
    --do_eval \
    --learning_rate 1e-3 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end False \
    --save_total_limit 3 \
    --seed 1337 \
    --ignore_mismatched_sizes True \
    --report_to wandb &> ${output_dir}/train.log