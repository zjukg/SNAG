# 600 250
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python  main.py \
            --gpu           $1    \
            --eval_epoch    1  \
            --only_test     0   \
            --model_name    SNAG \
            --data_choice   $2 \
            --data_split    $3 \
            --data_rate     $4 \
            --epoch         1500 \
            --lr            5e-4  \
            --hidden_units  "300,300,300" \
            --save_model    0 \
            --batch_size    3500 \
            --semi_learn_step 5 \
	        --csls          \
	        --csls_k        3 \
	        --random_seed   $5 \
            --exp_name      SNAG_NOISE_${7}_NR_${8}_MR_${9}_v1 \
            --exp_id        ${7}_NR_${8}_MR_${9} \
            --workers       12 \
            --accumulation_steps 1 \
            --scheduler     cos \
            --attr_dim      300     \
            --img_dim       300     \
            --name_dim      300     \
            --char_dim      300     \
            --hidden_size   300     \
            --intermediate_size 400 \
            --tau           0.1     \
            --tau2          4.0     \
            --structure_encoder "gat" \
            --num_attention_heads 1 \
            --num_hidden_layers 1 \
            --use_surface   $6     \
            --use_intermediate 1   \
            --replay 0 \
            --ratio $7 \
            --il            \
	        --il_start      250 \
            --enable_sota \
            --add_noise 1 \
            --noise_ratio $8 \
            --mask_ratio $9 \
            # --w_img \
            # --unsup \
            # --unsup_mode $8 \
            # --unsup_k 3000  \