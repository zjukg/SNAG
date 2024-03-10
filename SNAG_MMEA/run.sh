# ---
bash run_snag.sh $1 DBP15K ja_en 0.3 3408 0 1.0 0.2 0.7
bash run_snag.sh $1 DBP15K ja_en 0.3 3408 0 0.6 0.2 0.7
bash run_snag.sh $1 DBP15K ja_en 0.3 3408 0 0.4 0.2 0.7

bash run_snag.sh $1 DBP15K zh_en 0.3 3408 0 1.0 0.2 0.7
bash run_snag.sh $1 DBP15K zh_en 0.3 3408 0 0.6 0.2 0.7
bash run_snag.sh $1 DBP15K zh_en 0.3 3408 0 0.4 0.2 0.7

bash run_snag.sh $1 DBP15K fr_en 0.3 3408 0 1.0 0.2 0.7
bash run_snag.sh $1 DBP15K fr_en 0.3 3408 0 0.6 0.2 0.7
bash run_snag.sh $1 DBP15K fr_en 0.3 3408 0 0.4 0.2 0.7

bash run_snag.sh $1 OEA_D_W_15K_V2 norm 0.2 3408 0 1.0 0.2 0.7
bash run_snag.sh $1 OEA_D_W_15K_V2 norm 0.2 3408 0 0.6 0.2 0.7
bash run_snag.sh $1 OEA_D_W_15K_V2 norm 0.2 3408 0 0.4 0.2 0.7

bash run_snag.sh $1 OEA_D_W_15K_V1 norm 0.2 3408 0 1.0 0.2 0.7
bash run_snag.sh $1 OEA_D_W_15K_V1 norm 0.2 3408 0 0.6 0.2 0.7
bash run_snag.sh $1 OEA_D_W_15K_V1 norm 0.2 3408 0 0.4 0.2 0.7

bash run_snag.sh $1 OEA_EN_FR_15K_V1 norm 0.2 3408 0 1.0 0.8 0.2
bash run_snag.sh $1 OEA_EN_FR_15K_V1 norm 0.2 3408 0 0.6 0.8 0.2
bash run_snag.sh $1 OEA_EN_FR_15K_V1 norm 0.2 3408 0 0.4 0.8 0.2

bash run_snag.sh $1 OEA_EN_DE_15K_V1 norm 0.2 3408 0 1.0 0.8 0.2
bash run_snag.sh $1 OEA_EN_DE_15K_V1 norm 0.2 3408 0 0.6 0.8 0.2
bash run_snag.sh $1 OEA_EN_DE_15K_V1 norm 0.2 3408 0 0.4 0.8 0.2


bash run_meaformer.sh $1 DBP15K ja_en 0.3 3408 0 1.0 0.2 0.7
bash run_meaformer.sh $1 DBP15K ja_en 0.3 3408 0 0.6 0.2 0.7
bash run_meaformer.sh $1 DBP15K ja_en 0.3 3408 0 0.4 0.2 0.7

bash run_meaformer.sh $1 DBP15K zh_en 0.3 3408 0 1.0 0.2 0.7
bash run_meaformer.sh $1 DBP15K zh_en 0.3 3408 0 0.6 0.2 0.7
bash run_meaformer.sh $1 DBP15K zh_en 0.3 3408 0 0.4 0.2 0.7

bash run_meaformer.sh $1 DBP15K fr_en 0.3 3408 0 1.0 0.2 0.7
bash run_meaformer.sh $1 DBP15K fr_en 0.3 3408 0 0.6 0.2 0.7
bash run_meaformer.sh $1 DBP15K fr_en 0.3 3408 0 0.4 0.2 0.7

bash run_meaformer.sh $1 OEA_EN_FR_15K_V1 norm 0.2 3408 0 1.0 0.2 0.7
bash run_meaformer.sh $1 OEA_EN_FR_15K_V1 norm 0.2 3408 0 0.6 0.2 0.7
bash run_meaformer.sh $1 OEA_EN_FR_15K_V1 norm 0.2 3408 0 0.4 0.2 0.7

bash run_meaformer.sh $1 OEA_D_W_15K_V1 norm 0.2 3408 0 1.0 0.2 0.7
bash run_meaformer.sh $1 OEA_D_W_15K_V1 norm 0.2 3408 0 0.6 0.2 0.7
bash run_meaformer.sh $1 OEA_D_W_15K_V1 norm 0.2 3408 0 0.4 0.2 0.7

bash run_meaformer.sh $1 OEA_EN_DE_15K_V1 norm 0.2 3408 0 1.0 0.2 0.7
bash run_meaformer.sh $1 OEA_EN_DE_15K_V1 norm 0.2 3408 0 0.6 0.2 0.7
bash run_meaformer.sh $1 OEA_EN_DE_15K_V1 norm 0.2 3408 0 0.4 0.2 0.7

bash run_meaformer.sh $1 OEA_D_W_15K_V2 norm 0.2 3408 0 1.0 0.2 0.7
bash run_meaformer.sh $1 OEA_D_W_15K_V2 norm 0.2 3408 0 0.6 0.2 0.7
bash run_meaformer.sh $1 OEA_D_W_15K_V2 norm 0.2 3408 0 0.4 0.2 0.7



bash run_mclea.sh $1 DBP15K fr_en 0.3 3408 0 1.0 0.2 0.7
bash run_mclea.sh $1 DBP15K fr_en 0.3 3408 0 0.6 0.2 0.7
bash run_mclea.sh $1 DBP15K fr_en 0.3 3408 0 0.4 0.2 0.7

bash run_mclea.sh $1 OEA_D_W_15K_V2 norm 0.2 3408 0 1.0 0.2 0.7
bash run_mclea.sh $1 OEA_D_W_15K_V2 norm 0.2 3408 0 0.6 0.2 0.7
bash run_mclea.sh $1 OEA_D_W_15K_V2 norm 0.2 3408 0 0.4 0.2 0.7

bash run_mclea.sh $1 DBP15K ja_en 0.3 3408 0 1.0 0.2 0.7
bash run_mclea.sh $1 DBP15K ja_en 0.3 3408 0 0.6 0.2 0.7
bash run_mclea.sh $1 DBP15K ja_en 0.3 3408 0 0.4 0.2 0.7

bash run_mclea.sh $1 DBP15K zh_en 0.3 3408 0 1.0 0.2 0.7
bash run_mclea.sh $1 DBP15K zh_en 0.3 3408 0 0.6 0.2 0.7
bash run_mclea.sh $1 DBP15K zh_en 0.3 3408 0 0.4 0.2 0.7

bash run_mclea.sh $1 OEA_D_W_15K_V1 norm 0.2 3408 0 1.0 0.2 0.7
bash run_mclea.sh $1 OEA_D_W_15K_V1 norm 0.2 3408 0 0.6 0.2 0.7
bash run_mclea.sh $1 OEA_D_W_15K_V1 norm 0.2 3408 0 0.4 0.2 0.7

bash run_mclea.sh $1 OEA_EN_DE_15K_V1 norm 0.2 3408 0 1.0 0.2 0.7
bash run_mclea.sh $1 OEA_EN_DE_15K_V1 norm 0.2 3408 0 0.6 0.2 0.7
bash run_mclea.sh $1 OEA_EN_DE_15K_V1 norm 0.2 3408 0 0.4 0.2 0.7

bash run_mclea.sh $1 OEA_EN_FR_15K_V1 norm 0.2 3408 0 1.0 0.2 0.7
bash run_mclea.sh $1 OEA_EN_FR_15K_V1 norm 0.2 3408 0 0.6 0.2 0.7
bash run_mclea.sh $1 OEA_EN_FR_15K_V1 norm 0.2 3408 0 0.4 0.2 0.7


bash run_eva.sh $1 DBP15K zh_en 0.3 3408 0 1.0 0.2 0.7
bash run_eva.sh $1 DBP15K zh_en 0.3 3408 0 0.6 0.2 0.7
bash run_eva.sh $1 DBP15K zh_en 0.3 3408 0 0.4 0.2 0.7

bash run_eva.sh $1 DBP15K fr_en 0.3 3408 0 1.0 0.2 0.7
bash run_eva.sh $1 DBP15K fr_en 0.3 3408 0 0.6 0.2 0.7
bash run_eva.sh $1 DBP15K fr_en 0.3 3408 0 0.4 0.2 0.7

bash run_eva.sh $1 OEA_D_W_15K_V2 norm 0.2 3408 0 1.0 0.2 0.7
bash run_eva.sh $1 OEA_D_W_15K_V2 norm 0.2 3408 0 0.6 0.2 0.7
bash run_eva.sh $1 OEA_D_W_15K_V2 norm 0.2 3408 0 0.4 0.2 0.7

bash run_eva.sh $1 OEA_D_W_15K_V1 norm 0.2 3408 0 1.0 0.2 0.7
bash run_eva.sh $1 OEA_D_W_15K_V1 norm 0.2 3408 0 0.6 0.2 0.7
bash run_eva.sh $1 OEA_D_W_15K_V1 norm 0.2 3408 0 0.4 0.2 0.7

bash run_eva.sh $1 OEA_EN_DE_15K_V1 norm 0.2 3408 0 1.0 0.2 0.7
bash run_eva.sh $1 OEA_EN_DE_15K_V1 norm 0.2 3408 0 0.6 0.2 0.7
bash run_eva.sh $1 OEA_EN_DE_15K_V1 norm 0.2 3408 0 0.4 0.2 0.7

bash run_eva.sh $1 OEA_EN_FR_15K_V1 norm 0.2 3408 0 1.0 0.2 0.7
bash run_eva.sh $1 OEA_EN_FR_15K_V1 norm 0.2 3408 0 0.6 0.2 0.7
bash run_eva.sh $1 OEA_EN_FR_15K_V1 norm 0.2 3408 0 0.4 0.2 0.7

bash run_eva.sh $1 DBP15K ja_en 0.3 3408 0 1.0 0.2 0.7
bash run_eva.sh $1 DBP15K ja_en 0.3 3408 0 0.6 0.2 0.7
bash run_eva.sh $1 DBP15K ja_en 0.3 3408 0 0.4 0.2 0.7
