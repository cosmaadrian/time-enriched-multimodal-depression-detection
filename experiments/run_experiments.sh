#!/bin/bash
set -e
cd ..
GROUP=experiments

#########################################
############# TLSTM #####################
#########################################

python main.py  --config_file configs/tlstm_config.yaml --name tlstm --group tlstm --fold 0 --dataset twitter --window_size 128 --mode dryrun --epochs 200 --batch_size 256
python main.py  --config_file configs/tlstm_config.yaml --name tlstm-fold-1 --group tlstm --fold 1 --dataset twitter --window_size 128 --mode run --epochs 200 --batch_size 256
python main.py  --config_file configs/tlstm_config.yaml --name tlstm-fold-2 --group tlstm --fold 2 --dataset twitter --window_size 128 --mode run --epochs 200 --batch_size 256
python main.py  --config_file configs/tlstm_config.yaml --name tlstm-fold-3 --group tlstm --fold 3 --dataset twitter --window_size 128 --mode run --epochs 200 --batch_size 256
python main.py  --config_file configs/tlstm_config.yaml --name tlstm-fold-4 --group tlstm --fold 4 --dataset twitter --window_size 128 --mode run --epochs 200 --batch_size 256

python main.py  --config_file configs/tlstm_config.yaml --name tlstm --group tlstm-reddit --dataset reddit --window_size 128 --mode run --epochs 200 --batch_size 256

############# EVALUATE #################

python evaluate.py  --config_file configs/tlstm_config.yaml --name tlstm --group tlstm --fold 0 --dataset twitter --window_size 128 --output_dir tlstm-twitter
python evaluate.py  --config_file configs/tlstm_config.yaml --name tlstm-fold-1 --group tlstm --fold 1 --dataset twitter --window_size 128 --output_dir tlstm-twitter
python evaluate.py  --config_file configs/tlstm_config.yaml --name tlstm-fold-2 --group tlstm --fold 2 --dataset twitter --window_size 128 --output_dir tlstm-twitter
python evaluate.py  --config_file configs/tlstm_config.yaml --name tlstm-fold-3 --group tlstm --fold 3 --dataset twitter --window_size 128 --output_dir tlstm-twitter
python evaluate.py  --config_file configs/tlstm_config.yaml --name tlstm-fold-4 --group tlstm --fold 4 --dataset twitter --window_size 128 --output_dir tlstm-twitter

python evaluate.py  --config_file configs/tlstm_config.yaml --name tlstm --group tlstm-reddit --dataset reddit --window_size 128 --output_dir tlstm-reddit

#########################################
############# END TLSTM #################
#########################################


#########################################
############# EmoBERTa Text-Only ########
#########################################

python main.py --config_file configs/text_modality.yaml --text_embeddings_type emoberta --name fold-0-twitter-emoberta-only-learned --group twitter-text-only --dataset twitter --fold 0 --position_embeddings learned --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/text_modality.yaml --text_embeddings_type emoberta --name fold-1-twitter-emoberta-only-learned --group twitter-text-only --dataset twitter --fold 1 --position_embeddings learned --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/text_modality.yaml --text_embeddings_type emoberta --name fold-2-twitter-emoberta-only-learned --group twitter-text-only --dataset twitter --fold 2 --position_embeddings learned --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/text_modality.yaml --text_embeddings_type emoberta --name fold-3-twitter-emoberta-only-learned --group twitter-text-only --dataset twitter --fold 3 --position_embeddings learned --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/text_modality.yaml --text_embeddings_type emoberta --name fold-4-twitter-emoberta-only-learned --group twitter-text-only --dataset twitter --fold 4 --position_embeddings learned --mode run --epochs 200 --batch_size 256

python evaluate.py --config_file configs/text_modality.yaml --text_embeddings_type emoberta --name fold-0-twitter-emoberta-only-learned --group twitter-text-only --dataset twitter --fold 0 --position_embeddings learned
python evaluate.py --config_file configs/text_modality.yaml --text_embeddings_type emoberta --name fold-1-twitter-emoberta-only-learned --group twitter-text-only --dataset twitter --fold 1 --position_embeddings learned
python evaluate.py --config_file configs/text_modality.yaml --text_embeddings_type emoberta --name fold-2-twitter-emoberta-only-learned --group twitter-text-only --dataset twitter --fold 2 --position_embeddings learned
python evaluate.py --config_file configs/text_modality.yaml --text_embeddings_type emoberta --name fold-3-twitter-emoberta-only-learned --group twitter-text-only --dataset twitter --fold 3 --position_embeddings learned
python evaluate.py --config_file configs/text_modality.yaml --text_embeddings_type emoberta --name fold-4-twitter-emoberta-only-learned --group twitter-text-only --dataset twitter --fold 4 --position_embeddings learned


python main.py --config_file configs/text_modality.yaml --text_embeddings_type emoberta --name reddit-emoberta-only-learned --group reddit-text-only --dataset reddit --position_embeddings learned --mode run --epochs 200 --batch_size 256
python evaluate.py --config_file configs/text_modality.yaml --text_embeddings_type emoberta --name reddit-emoberta-only-learned --group reddit-text-only --dataset reddit --position_embeddings learned

#########################################
############# END EmoBERTa Text-Only ####
#########################################

#########################################
############# TWITTER MULTIMODAL ####
#########################################
################################

python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-0-twitter-ws-32-clip-emoberta-time2vec --group $GROUP --dataset twitter --fold 0 --window_size 32  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-0-twitter-ws-64-clip-emoberta-time2vec   --group $GROUP --dataset twitter --fold 0 --window_size 64  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-0-twitter-ws-128-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 0 --window_size 128 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-0-twitter-ws-256-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 0 --window_size 256 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 64 --accumulation_steps 4
python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-0-twitter-ws-512-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 0 --window_size 512 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 32 --accumulation_steps 8

python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-0-twitter-ws-32-clip-emoberta-time2vec --group $GROUP --dataset twitter --fold 0 --window_size 32  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-0-twitter-ws-64-clip-emoberta-time2vec   --group $GROUP --dataset twitter --fold 0 --window_size 64  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-0-twitter-ws-128-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 0 --window_size 128 --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-0-twitter-ws-256-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 0 --window_size 256 --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-0-twitter-ws-512-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 0 --window_size 512 --position_embeddings time2vec --output_dir $GROUP

##############################

python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-1-twitter-ws-32-clip-emoberta-time2vec --group $GROUP --dataset twitter --fold 1 --window_size 32  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-1-twitter-ws-64-clip-emoberta-time2vec   --group $GROUP --dataset twitter --fold 1 --window_size 64  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-1-twitter-ws-128-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 1 --window_size 128 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-1-twitter-ws-256-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 1 --window_size 256 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 64 --accumulation_steps 4
python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-1-twitter-ws-512-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 1 --window_size 512 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 32 --accumulation_steps 8

python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-1-twitter-ws-32-clip-emoberta-time2vec --group $GROUP --dataset twitter --fold 1 --window_size 32  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-1-twitter-ws-64-clip-emoberta-time2vec   --group $GROUP --dataset twitter --fold 1 --window_size 64  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-1-twitter-ws-128-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 1 --window_size 128 --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-1-twitter-ws-256-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 1 --window_size 256 --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-1-twitter-ws-512-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 1 --window_size 512 --position_embeddings time2vec --output_dir $GROUP

##############################

python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-2-twitter-ws-32-clip-emoberta-time2vec --group $GROUP --dataset twitter --fold 2 --window_size 32  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-2-twitter-ws-64-clip-emoberta-time2vec   --group $GROUP --dataset twitter --fold 2 --window_size 64  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-2-twitter-ws-128-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 2 --window_size 128 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-2-twitter-ws-256-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 2 --window_size 256 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 64 --accumulation_steps 4
python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-2-twitter-ws-512-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 2 --window_size 512 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 32 --accumulation_steps 8

python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-2-twitter-ws-32-clip-emoberta-time2vec --group $GROUP --dataset twitter --fold 2 --window_size 32  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-2-twitter-ws-64-clip-emoberta-time2vec   --group $GROUP --dataset twitter --fold 2 --window_size 64  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-2-twitter-ws-128-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 2 --window_size 128 --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-2-twitter-ws-256-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 2 --window_size 256 --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-2-twitter-ws-512-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 2 --window_size 512 --position_embeddings time2vec --output_dir $GROUP

##############################

python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-3-twitter-ws-32-clip-emoberta-time2vec   --group $GROUP --dataset twitter --fold 3 --window_size 32  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-3-twitter-ws-64-clip-emoberta-time2vec   --group $GROUP --dataset twitter --fold 3 --window_size 64  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-3-twitter-ws-128-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 3 --window_size 128 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-3-twitter-ws-256-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 3 --window_size 256 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 64 --accumulation_steps 4
python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-3-twitter-ws-512-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 3 --window_size 512 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 32 --accumulation_steps 8

python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-3-twitter-ws-32-clip-emoberta-time2vec   --group $GROUP --dataset twitter --fold 3 --window_size 32  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-3-twitter-ws-64-clip-emoberta-time2vec   --group $GROUP --dataset twitter --fold 3 --window_size 64  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-3-twitter-ws-128-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 3 --window_size 128 --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-3-twitter-ws-256-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 3 --window_size 256 --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-3-twitter-ws-512-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 3 --window_size 512 --position_embeddings time2vec --output_dir $GROUP

###############################

python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-4-twitter-ws-32-clip-emoberta-time2vec   --group $GROUP --dataset twitter --fold 4 --window_size 32  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-4-twitter-ws-64-clip-emoberta-time2vec   --group $GROUP --dataset twitter --fold 4 --window_size 64  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-4-twitter-ws-128-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 4 --window_size 128 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-4-twitter-ws-256-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 4 --window_size 256 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 64 --accumulation_steps 4
python main.py  --config_file configs/combos/clip_emoberta.yaml --name fold-4-twitter-ws-512-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 4 --window_size 512 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 32 --accumulation_steps 8

python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-4-twitter-ws-32-clip-emoberta-time2vec   --group $GROUP --dataset twitter --fold 4 --window_size 32  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-4-twitter-ws-64-clip-emoberta-time2vec   --group $GROUP --dataset twitter --fold 4 --window_size 64  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-4-twitter-ws-128-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 4 --window_size 128 --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-4-twitter-ws-256-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 4 --window_size 256 --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name fold-4-twitter-ws-512-clip-emoberta-time2vec  --group $GROUP --dataset twitter --fold 4 --window_size 512 --position_embeddings time2vec --output_dir $GROUP

python main.py  --config_file configs/combos/dino_emoberta.yaml --name fold-0-twitter-ws-128-dino-emoberta-time2vec --group $GROUP --dataset twitter --fold 0 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/dino_emoberta.yaml --name fold-1-twitter-ws-128-dino-emoberta-time2vec --group $GROUP --dataset twitter --fold 1 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/dino_emoberta.yaml --name fold-2-twitter-ws-128-dino-emoberta-time2vec --group $GROUP --dataset twitter --fold 2 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/dino_emoberta.yaml --name fold-3-twitter-ws-128-dino-emoberta-time2vec --group $GROUP --dataset twitter --fold 3 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/dino_emoberta.yaml --name fold-4-twitter-ws-128-dino-emoberta-time2vec --group $GROUP --dataset twitter --fold 4 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256

python evaluate.py  --config_file configs/combos/dino_emoberta.yaml --name fold-0-twitter-ws-128-dino-emoberta-time2vec --group $GROUP --dataset twitter --fold 0 --window_size 128  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/dino_emoberta.yaml --name fold-1-twitter-ws-128-dino-emoberta-time2vec --group $GROUP --dataset twitter --fold 1 --window_size 128  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/dino_emoberta.yaml --name fold-2-twitter-ws-128-dino-emoberta-time2vec --group $GROUP --dataset twitter --fold 2 --window_size 128  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/dino_emoberta.yaml --name fold-3-twitter-ws-128-dino-emoberta-time2vec --group $GROUP --dataset twitter --fold 3 --window_size 128  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/dino_emoberta.yaml --name fold-4-twitter-ws-128-dino-emoberta-time2vec --group $GROUP --dataset twitter --fold 4 --window_size 128  --position_embeddings time2vec --output_dir $GROUP

#################################
#################################
#################################
#################################

python main.py  --config_file configs/combos/dino_roberta.yaml --name fold-0-twitter-ws-128-dino-roberta-time2vec --group $GROUP --dataset twitter --fold 0 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/dino_roberta.yaml --name fold-1-twitter-ws-128-dino-roberta-time2vec --group $GROUP --dataset twitter --fold 1 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/dino_roberta.yaml --name fold-2-twitter-ws-128-dino-roberta-time2vec --group $GROUP --dataset twitter --fold 2 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/dino_roberta.yaml --name fold-3-twitter-ws-128-dino-roberta-time2vec --group $GROUP --dataset twitter --fold 3 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/dino_roberta.yaml --name fold-4-twitter-ws-128-dino-roberta-time2vec --group $GROUP --dataset twitter --fold 4 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256

python evaluate.py  --config_file configs/combos/dino_roberta.yaml --name fold-0-twitter-ws-128-dino-roberta-time2vec --group $GROUP --dataset twitter --fold 0 --window_size 128  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/dino_roberta.yaml --name fold-1-twitter-ws-128-dino-roberta-time2vec --group $GROUP --dataset twitter --fold 1 --window_size 128  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/dino_roberta.yaml --name fold-2-twitter-ws-128-dino-roberta-time2vec --group $GROUP --dataset twitter --fold 2 --window_size 128  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/dino_roberta.yaml --name fold-3-twitter-ws-128-dino-roberta-time2vec --group $GROUP --dataset twitter --fold 3 --window_size 128  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/dino_roberta.yaml --name fold-4-twitter-ws-128-dino-roberta-time2vec --group $GROUP --dataset twitter --fold 4 --window_size 128  --position_embeddings time2vec --output_dir $GROUP

# #################################
# #################################
# #################################

python main.py  --config_file configs/combos/dino_minilm.yaml --name fold-0-twitter-ws-128-dino-minilm-time2vec --group $GROUP --dataset twitter --fold 0 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/dino_minilm.yaml --name fold-1-twitter-ws-128-dino-minilm-time2vec --group $GROUP --dataset twitter --fold 1 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/dino_minilm.yaml --name fold-2-twitter-ws-128-dino-minilm-time2vec --group $GROUP --dataset twitter --fold 2 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/dino_minilm.yaml --name fold-3-twitter-ws-128-dino-minilm-time2vec --group $GROUP --dataset twitter --fold 3 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/dino_minilm.yaml --name fold-4-twitter-ws-128-dino-minilm-time2vec --group $GROUP --dataset twitter --fold 4 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256

python evaluate.py  --config_file configs/combos/dino_minilm.yaml --name fold-0-twitter-ws-128-dino-minilm-time2vec --group $GROUP --dataset twitter --fold 0 --window_size 128  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/dino_minilm.yaml --name fold-1-twitter-ws-128-dino-minilm-time2vec --group $GROUP --dataset twitter --fold 1 --window_size 128  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/dino_minilm.yaml --name fold-2-twitter-ws-128-dino-minilm-time2vec --group $GROUP --dataset twitter --fold 2 --window_size 128  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/dino_minilm.yaml --name fold-3-twitter-ws-128-dino-minilm-time2vec --group $GROUP --dataset twitter --fold 3 --window_size 128  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/dino_minilm.yaml --name fold-4-twitter-ws-128-dino-minilm-time2vec --group $GROUP --dataset twitter --fold 4 --window_size 128  --position_embeddings time2vec --output_dir $GROUP


# #################################
# #################################
# #################################

python main.py  --config_file configs/combos/clip_roberta.yaml --name fold-0-twitter-ws-128-clip-roberta-time2vec --group $GROUP --dataset twitter --fold 0 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_roberta.yaml --name fold-1-twitter-ws-128-clip-roberta-time2vec --group $GROUP --dataset twitter --fold 1 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_roberta.yaml --name fold-2-twitter-ws-128-clip-roberta-time2vec --group $GROUP --dataset twitter --fold 2 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_roberta.yaml --name fold-3-twitter-ws-128-clip-roberta-time2vec --group $GROUP --dataset twitter --fold 3 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_roberta.yaml --name fold-4-twitter-ws-128-clip-roberta-time2vec --group $GROUP --dataset twitter --fold 4 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256

python evaluate.py  --config_file configs/combos/clip_roberta.yaml --name fold-0-twitter-ws-128-clip-roberta-time2vec --group $GROUP --dataset twitter --fold 0 --window_size 128  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_roberta.yaml --name fold-1-twitter-ws-128-clip-roberta-time2vec --group $GROUP --dataset twitter --fold 1 --window_size 128  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_roberta.yaml --name fold-2-twitter-ws-128-clip-roberta-time2vec --group $GROUP --dataset twitter --fold 2 --window_size 128  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_roberta.yaml --name fold-3-twitter-ws-128-clip-roberta-time2vec --group $GROUP --dataset twitter --fold 3 --window_size 128  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_roberta.yaml --name fold-4-twitter-ws-128-clip-roberta-time2vec --group $GROUP --dataset twitter --fold 4 --window_size 128  --position_embeddings time2vec --output_dir $GROUP


# #################################
# #################################
# #################################

python main.py  --config_file configs/combos/clip_minilm.yaml --name fold-0-twitter-ws-128-clip-minilm-time2vec --group $GROUP --dataset twitter --fold 0 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_minilm.yaml --name fold-1-twitter-ws-128-clip-minilm-time2vec --group $GROUP --dataset twitter --fold 1 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_minilm.yaml --name fold-2-twitter-ws-128-clip-minilm-time2vec --group $GROUP --dataset twitter --fold 2 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_minilm.yaml --name fold-3-twitter-ws-128-clip-minilm-time2vec --group $GROUP --dataset twitter --fold 3 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_minilm.yaml --name fold-4-twitter-ws-128-clip-minilm-time2vec --group $GROUP --dataset twitter --fold 4 --window_size 128  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256

python evaluate.py  --config_file configs/combos/clip_minilm.yaml --name fold-0-twitter-ws-128-clip-minilm-time2vec --group $GROUP --dataset twitter --fold 0 --window_size 128  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_minilm.yaml --name fold-1-twitter-ws-128-clip-minilm-time2vec --group $GROUP --dataset twitter --fold 1 --window_size 128  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_minilm.yaml --name fold-2-twitter-ws-128-clip-minilm-time2vec --group $GROUP --dataset twitter --fold 2 --window_size 128  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_minilm.yaml --name fold-3-twitter-ws-128-clip-minilm-time2vec --group $GROUP --dataset twitter --fold 3 --window_size 128  --position_embeddings time2vec --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_minilm.yaml --name fold-4-twitter-ws-128-clip-minilm-time2vec --group $GROUP --dataset twitter --fold 4 --window_size 128  --position_embeddings time2vec --output_dir $GROUP


################################################
################### Reddit Time2Vec ####################
################################################

python main.py  --config_file configs/combos/clip_emoberta.yaml --name reddit-ws-32-clip-emoberta-time2vec --group $GROUP --dataset reddit --window_size 32  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_emoberta.yaml --name reddit-ws-64-clip-emoberta-time2vec   --group $GROUP --dataset reddit --window_size 64  --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_emoberta.yaml --name reddit-ws-128-clip-emoberta-time2vec  --group $GROUP --dataset reddit --window_size 128 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_emoberta.yaml --name reddit-ws-256-clip-emoberta-time2vec  --group $GROUP --dataset reddit --window_size 256 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 64 --accumulation_steps 4
python main.py  --config_file configs/combos/clip_emoberta.yaml --name reddit-ws-512-clip-emoberta-time2vec  --group $GROUP --dataset reddit --window_size 512 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 32 --accumulation_steps 8

python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name reddit-ws-32-clip-emoberta-time2vec --group $GROUP --dataset reddit --window_size 32  --position_embeddings time2vec  --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name reddit-ws-64-clip-emoberta-time2vec   --group $GROUP --dataset reddit --window_size 64  --position_embeddings time2vec  --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name reddit-ws-128-clip-emoberta-time2vec  --group $GROUP --dataset reddit --window_size 128 --position_embeddings time2vec  --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name reddit-ws-256-clip-emoberta-time2vec  --group $GROUP --dataset reddit --window_size 256 --position_embeddings time2vec  --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name reddit-ws-512-clip-emoberta-time2vec  --group $GROUP --dataset reddit --window_size 512 --position_embeddings time2vec --output_dir $GROUP


################################################
################### Reddit LEARNED ####################
################################################
python main.py  --config_file configs/combos/clip_emoberta.yaml --name reddit-ws-32-clip-emoberta-learned --group $GROUP --dataset reddit --window_size 32  --position_embeddings learned --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_emoberta.yaml --name reddit-ws-64-clip-emoberta-learned   --group $GROUP --dataset reddit --window_size 64  --position_embeddings learned --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_emoberta.yaml --name reddit-ws-128-clip-emoberta-learned  --group $GROUP --dataset reddit --window_size 128 --position_embeddings learned --mode run --epochs $EPOCHS --batch_size 256
python main.py  --config_file configs/combos/clip_emoberta.yaml --name reddit-ws-256-clip-emoberta-learned  --group $GROUP --dataset reddit --window_size 256 --position_embeddings learned --mode run --epochs $EPOCHS --batch_size 64 --accumulation_steps 4
python main.py  --config_file configs/combos/clip_emoberta.yaml --name reddit-ws-512-clip-emoberta-learned  --group $GROUP --dataset reddit --window_size 512 --position_embeddings learned --mode run --epochs $EPOCHS --batch_size 32 --accumulation_steps 8

python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name reddit-ws-32-clip-emoberta-learned --group $GROUP --dataset reddit --window_size 32  --position_embeddings learned  --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name reddit-ws-64-clip-emoberta-learned   --group $GROUP --dataset reddit --window_size 64  --position_embeddings learned  --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name reddit-ws-128-clip-emoberta-learned  --group $GROUP --dataset reddit --window_size 128 --position_embeddings learned  --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name reddit-ws-256-clip-emoberta-learned  --group $GROUP --dataset reddit --window_size 256 --position_embeddings learned  --output_dir $GROUP
python evaluate.py  --config_file configs/combos/clip_emoberta.yaml --name reddit-ws-512-clip-emoberta-learned  --group $GROUP --dataset reddit --window_size 512 --position_embeddings learned --output_dir $GROUP

################ REDDIT DINO + EMOBERTA
python main.py  --config_file configs/combos/dino_emoberta.yaml --name reddit-ws-128-dino-emoberta-time2vec  --group $GROUP --dataset reddit --window_size 128 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python evaluate.py  --config_file configs/combos/dino_emoberta.yaml --name reddit-ws-128-dino-emoberta-time2vec  --group $GROUP --dataset reddit --window_size 128 --position_embeddings time2vec  --output_dir $GROUP
#########################################

################ REDDIT DINO + ROBERTA
python main.py  --config_file configs/combos/dino_roberta.yaml --name reddit-ws-128-dino-roberta-time2vec  --group $GROUP --dataset reddit --window_size 128 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python evaluate.py  --config_file configs/combos/dino_roberta.yaml --name reddit-ws-128-dino-roberta-time2vec  --group $GROUP --dataset reddit --window_size 128 --position_embeddings time2vec  --output_dir $GROUP
#########################################

################ REDDIT DINO + MINILM
python main.py  --config_file configs/combos/dino_minilm.yaml --name reddit-ws-128-dino-minilm-time2vec  --group $GROUP --dataset reddit --window_size 128 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python evaluate.py  --config_file configs/combos/dino_minilm.yaml --name reddit-ws-128-dino-minilm-time2vec  --group $GROUP --dataset reddit --window_size 128 --position_embeddings time2vec  --output_dir $GROUP
#########################################

################ REDDIT CLIP + ROBERTA
python main.py  --config_file configs/combos/clip_roberta.yaml --name reddit-ws-128-clip-roberta-time2vec  --group $GROUP --dataset reddit --window_size 128 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python evaluate.py  --config_file configs/combos/clip_roberta.yaml --name reddit-ws-128-clip-roberta-time2vec  --group $GROUP --dataset reddit --window_size 128 --position_embeddings time2vec  --output_dir $GROUP
#########################################

################ REDDIT CLIP + MINILM
python main.py  --config_file configs/combos/clip_minilm.yaml --name reddit-ws-128-clip-minilm-time2vec  --group $GROUP --dataset reddit --window_size 128 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python evaluate.py  --config_file configs/combos/clip_minilm.yaml --name reddit-ws-128-clip-minilm-time2vec  --group $GROUP --dataset reddit --window_size 128 --position_embeddings time2vec  --output_dir $GROUP
#########################################


###########################################
######### TWITTER ABLATION ################
###########################################

python main.py --config_file configs/combos/clip_roberta.yaml --name fold-2-twitter-ws-128-clip-roberta --group twitter-ablation --dataset twitter --fold 2 --window_size 128 --position_embeddings time2vec --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/clip_minilm.yaml --name fold-2-twitter-ws-128-clip-minilm --group twitter-ablation --dataset twitter --fold 2 --window_size 128 --position_embeddings time2vec --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/dino_emoberta.yaml --name fold-2-twitter-ws-128-dino-emoberta --group twitter-ablation --dataset twitter --fold 2 --window_size 128 --position_embeddings time2vec --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/dino_roberta.yaml --name fold-2-twitter-ws-128-dino-roberta --group twitter-ablation --dataset twitter --fold 2 --window_size 128 --position_embeddings time2vec --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/dino_minilm.yaml --name fold-2-twitter-ws-128-dino-minilm --group twitter-ablation --dataset twitter --fold 2 --window_size 128 --position_embeddings time2vec --mode run --epochs 200 --batch_size 256

python main.py --config_file configs/combos/clip_roberta.yaml --name fold-2-twitter-ws-256-clip-roberta --group twitter-ablation --dataset twitter --fold 2 --window_size 256 --position_embeddings time2vec --mode run --epochs 200 --batch_size 128 --accumulation_steps 2
python main.py --config_file configs/combos/clip_minilm.yaml --name fold-2-twitter-ws-256-clip-minilm --group twitter-ablation --dataset twitter --fold 2 --window_size 256 --position_embeddings time2vec --mode run --epochs 200 --batch_size 128 --accumulation_steps 2
python main.py --config_file configs/combos/dino_emoberta.yaml --name fold-2-twitter-ws-256-dino-emoberta --group twitter-ablation --dataset twitter --fold 2 --window_size 256 --position_embeddings time2vec --mode run --epochs 200 --batch_size 128 --accumulation_steps 2
python main.py --config_file configs/combos/dino_roberta.yaml --name fold-2-twitter-ws-256-dino-roberta --group twitter-ablation --dataset twitter --fold 2 --window_size 256 --position_embeddings time2vec --mode run --epochs 200 --batch_size 128 --accumulation_steps 2
python main.py --config_file configs/combos/dino_minilm.yaml --name fold-2-twitter-ws-256-dino-minilm --group twitter-ablation --dataset twitter --fold 2 --window_size 256 --position_embeddings time2vec --mode run --epochs 200 --batch_size 128 --accumulation_steps 2

python main.py --config_file configs/combos/clip_roberta.yaml --name fold-3-twitter-ws-128-clip-roberta --group twitter-ablation --dataset twitter --fold 3 --window_size 128 --position_embeddings time2vec --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/clip_minilm.yaml --name fold-3-twitter-ws-128-clip-minilm --group twitter-ablation --dataset twitter --fold 3 --window_size 128 --position_embeddings time2vec --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/dino_emoberta.yaml --name fold-3-twitter-ws-128-dino-emoberta --group twitter-ablation --dataset twitter --fold 3 --window_size 128 --position_embeddings time2vec --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/dino_roberta.yaml --name fold-3-twitter-ws-128-dino-roberta --group twitter-ablation --dataset twitter --fold 3 --window_size 128 --position_embeddings time2vec --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/dino_minilm.yaml --name fold-3-twitter-ws-128-dino-minilm --group twitter-ablation --dataset twitter --fold 3 --window_size 128 --position_embeddings time2vec --mode run --epochs 200 --batch_size 256

python main.py --config_file configs/combos/clip_roberta.yaml --name fold-3-twitter-ws-256-clip-roberta --group twitter-ablation --dataset twitter --fold 3 --window_size 256 --position_embeddings time2vec --mode run --epochs 200 --batch_size 128 --accumulation_steps 2
python main.py --config_file configs/combos/clip_minilm.yaml --name fold-3-twitter-ws-256-clip-minilm --group twitter-ablation --dataset twitter --fold 3 --window_size 256 --position_embeddings time2vec --mode run --epochs 200 --batch_size 128 --accumulation_steps 2
python main.py --config_file configs/combos/dino_emoberta.yaml --name fold-3-twitter-ws-256-dino-emoberta --group twitter-ablation --dataset twitter --fold 3 --window_size 256 --position_embeddings time2vec --mode run --epochs 200 --batch_size 128 --accumulation_steps 2
python main.py --config_file configs/combos/dino_roberta.yaml --name fold-3-twitter-ws-256-dino-roberta --group twitter-ablation --dataset twitter --fold 3 --window_size 256 --position_embeddings time2vec --mode run --epochs 200 --batch_size 128 --accumulation_steps 2
python main.py --config_file configs/combos/dino_minilm.yaml --name fold-3-twitter-ws-256-dino-minilm --group twitter-ablation --dataset twitter --fold 3 --window_size 256 --position_embeddings time2vec --mode run --epochs 200 --batch_size 128 --accumulation_steps 2

python main.py --config_file configs/combos/clip_roberta.yaml --name fold-4-twitter-ws-128-clip-roberta --group twitter-ablation --dataset twitter --fold 4 --window_size 128 --position_embeddings time2vec --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/clip_minilm.yaml --name fold-4-twitter-ws-128-clip-minilm --group twitter-ablation --dataset twitter --fold 4 --window_size 128 --position_embeddings time2vec --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/dino_emoberta.yaml --name fold-4-twitter-ws-128-dino-emoberta --group twitter-ablation --dataset twitter --fold 4 --window_size 128 --position_embeddings time2vec --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/dino_roberta.yaml --name fold-4-twitter-ws-128-dino-roberta --group twitter-ablation --dataset twitter --fold 4 --window_size 128 --position_embeddings time2vec --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/dino_minilm.yaml --name fold-4-twitter-ws-128-dino-minilm --group twitter-ablation --dataset twitter --fold 4 --window_size 128 --position_embeddings time2vec --mode run --epochs 200 --batch_size 256

python main.py --config_file configs/combos/clip_roberta.yaml --name fold-4-twitter-ws-256-clip-roberta --group twitter-ablation --dataset twitter --fold 4 --window_size 256 --position_embeddings time2vec --mode run --epochs 200 --batch_size 128 --accumulation_steps 2
python main.py --config_file configs/combos/clip_minilm.yaml --name fold-4-twitter-ws-256-clip-minilm --group twitter-ablation --dataset twitter --fold 4 --window_size 256 --position_embeddings time2vec --mode run --epochs 200 --batch_size 128 --accumulation_steps 2
python main.py --config_file configs/combos/dino_emoberta.yaml --name fold-4-twitter-ws-256-dino-emoberta --group twitter-ablation --dataset twitter --fold 4 --window_size 256 --position_embeddings time2vec --mode run --epochs 200 --batch_size 128 --accumulation_steps 2
python main.py --config_file configs/combos/dino_roberta.yaml --name fold-4-twitter-ws-256-dino-roberta --group twitter-ablation --dataset twitter --fold 4 --window_size 256 --position_embeddings time2vec --mode run --epochs 200 --batch_size 128 --accumulation_steps 2
python main.py --config_file configs/combos/dino_minilm.yaml --name fold-4-twitter-ws-256-dino-minilm --group twitter-ablation --dataset twitter --fold 4 --window_size 256 --position_embeddings time2vec --mode run --epochs 200 --batch_size 128 --accumulation_steps 2



## EMOBERTA
python main.py --config_file configs/combos/clip_emoberta.yaml --name twitter-ws-512-clip-emoberta-learned --group twitter-ablation --dataset twitter --fold 0 --window_size 512 --position_embeddings learned --mode run --epochs 200 --batch_size 32 --accumulation_steps 8
python main.py --config_file configs/combos/clip_emoberta.yaml --name twitter-ws-512-clip-emoberta-zero --group twitter-ablation --dataset twitter --fold 0 --window_size 512 --position_embeddings zero --mode run --epochs 200 --batch_size 32 --accumulation_steps 8

python main.py --config_file configs/combos/clip_emoberta.yaml --name twitter-ws-256-clip-emoberta-learned --group twitter-ablation --dataset twitter --fold 0 --window_size 256 --position_embeddings learned --mode run --epochs 200 --batch_size 128 --accumulation_steps 2
python main.py --config_file configs/combos/clip_emoberta.yaml --name twitter-ws-256-clip-emoberta-zero --group twitter-ablation --dataset twitter --fold 0 --window_size 256 --position_embeddings zero --mode run --epochs 200 --batch_size 128 --accumulation_steps 2

python main.py --config_file configs/combos/clip_emoberta.yaml --name twitter-ws-128-clip-emoberta-learned --group twitter-ablation --dataset twitter --fold 0 --window_size 128 --position_embeddings learned --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/clip_emoberta.yaml --name twitter-ws-128-clip-emoberta-zero --group twitter-ablation --dataset twitter --fold 0 --window_size 128 --position_embeddings zero --mode run --epochs 200 --batch_size 256

python main.py --config_file configs/combos/clip_emoberta.yaml --name twitter-ws-64-clip-emoberta-learned --group twitter-ablation --dataset twitter --fold 0 --window_size 64 --position_embeddings learned --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/clip_emoberta.yaml --name twitter-ws-64-clip-emoberta-zero --group twitter-ablation --dataset twitter --fold 0 --window_size 64 --position_embeddings zero --mode run --epochs 200 --batch_size 256

python main.py --config_file configs/combos/clip_emoberta.yaml --name twitter-ws-32-clip-emoberta-learned --group twitter-ablation --dataset twitter --fold 0 --window_size 32 --position_embeddings learned --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/clip_emoberta.yaml --name twitter-ws-32-clip-emoberta-zero --group twitter-ablation --dataset twitter --fold 0 --window_size 32 --position_embeddings zero --mode run --epochs 200 --batch_size 256

## ROBERTA
python main.py --config_file configs/combos/clip_roberta.yaml --name twitter-ws-512-clip-roberta-learned --group twitter-ablation --dataset twitter --fold 0 --window_size 512 --position_embeddings learned --mode run --epochs 200 --batch_size 32 --accumulation_steps 8
python main.py --config_file configs/combos/clip_roberta.yaml --name twitter-ws-512-clip-roberta-zero --group twitter-ablation --dataset twitter --fold 0 --window_size 512 --position_embeddings zero --mode run --epochs 200 --batch_size 32 --accumulation_steps 8

python main.py --config_file configs/combos/clip_roberta.yaml --name twitter-ws-256-clip-roberta-learned --group twitter-ablation --dataset twitter --fold 0 --window_size 256 --position_embeddings learned --mode run --epochs 200 --batch_size 128 --accumulation_steps 2
python main.py --config_file configs/combos/clip_roberta.yaml --name twitter-ws-256-clip-roberta-zero --group twitter-ablation --dataset twitter --fold 0 --window_size 256 --position_embeddings zero --mode run --epochs 200 --batch_size 128 --accumulation_steps 2

python main.py --config_file configs/combos/clip_roberta.yaml --name twitter-ws-128-clip-roberta-learned --group twitter-ablation --dataset twitter --fold 0 --window_size 128 --position_embeddings learned --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/clip_roberta.yaml --name twitter-ws-128-clip-roberta-zero --group twitter-ablation --dataset twitter --fold 0 --window_size 128 --position_embeddings zero --mode run --epochs 200 --batch_size 256

python main.py --config_file configs/combos/clip_roberta.yaml --name twitter-ws-64-clip-roberta-learned --group twitter-ablation --dataset twitter --fold 0 --window_size 64 --position_embeddings learned --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/clip_roberta.yaml --name twitter-ws-64-clip-roberta-zero --group twitter-ablation --dataset twitter --fold 0 --window_size 64 --position_embeddings zero --mode run --epochs 200 --batch_size 256

python main.py --config_file configs/combos/clip_roberta.yaml --name twitter-ws-32-clip-roberta-learned --group twitter-ablation --dataset twitter --fold 0 --window_size 32 --position_embeddings learned --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/clip_roberta.yaml --name twitter-ws-32-clip-roberta-zero --group twitter-ablation --dataset twitter --fold 0 --window_size 32 --position_embeddings zero --mode run --epochs 200 --batch_size 256

############################ FOLD 1
#### EMOBERTA
python main.py --config_file configs/combos/clip_emoberta.yaml --name fold-1-twitter-ws-128-clip-emoberta-learned --group twitter-ablation --dataset twitter --fold 1 --window_size 128 --position_embeddings learned --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/clip_emoberta.yaml --name fold-1-twitter-ws-128-clip-emoberta-zero --group twitter-ablation --dataset twitter --fold 1 --window_size 128 --position_embeddings zero --mode run --epochs 200 --batch_size 256

# #### ROBERTA
python main.py --config_file configs/combos/clip_roberta.yaml --name fold-1-twitter-ws-128-clip-roberta-learned --group twitter-ablation --dataset twitter --fold 1 --window_size 128 --position_embeddings learned --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/clip_roberta.yaml --name fold-1-twitter-ws-128-clip-roberta-zero --group twitter-ablation --dataset twitter --fold 1 --window_size 128 --position_embeddings zero --mode run --epochs 200 --batch_size 256

# ############################ FOLD 2
# #### EMOBERTA
python main.py --config_file configs/combos/clip_emoberta.yaml --name fold-2-twitter-ws-128-clip-emoberta-learned --group twitter-ablation --dataset twitter --fold 2 --window_size 128 --position_embeddings learned --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/clip_emoberta.yaml --name fold-2-twitter-ws-128-clip-emoberta-zero --group twitter-ablation --dataset twitter --fold 2 --window_size 128 --position_embeddings zero --mode run --epochs 200 --batch_size 256

# #### ROBERTA
python main.py --config_file configs/combos/clip_roberta.yaml --name fold-2-twitter-ws-128-clip-roberta-learned --group twitter-ablation --dataset twitter --fold 2 --window_size 128 --position_embeddings learned --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/clip_roberta.yaml --name fold-2-twitter-ws-128-clip-roberta-zero --group twitter-ablation --dataset twitter --fold 2 --window_size 128 --position_embeddings zero --mode run --epochs 200 --batch_size 256

############################ FOLD 3
#### EMOBERTA
python main.py --config_file configs/combos/clip_emoberta.yaml --name fold-3-twitter-ws-128-clip-emoberta-learned --group twitter-ablation --dataset twitter --fold 3 --window_size 128 --position_embeddings learned --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/clip_emoberta.yaml --name fold-3-twitter-ws-128-clip-emoberta-zero --group twitter-ablation --dataset twitter --fold 3 --window_size 128 --position_embeddings zero --mode run --epochs 200 --batch_size 256

#### ROBERTA
python main.py --config_file configs/combos/clip_roberta.yaml --name fold-3-twitter-ws-128-clip-roberta-learned --group twitter-ablation --dataset twitter --fold 3 --window_size 128 --position_embeddings learned --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/clip_roberta.yaml --name fold-3-twitter-ws-128-clip-roberta-zero --group twitter-ablation --dataset twitter --fold 3 --window_size 128 --position_embeddings zero --mode run --epochs 200 --batch_size 256

############################ FOLD 4
#### EMOBERTA
python main.py --config_file configs/combos/clip_emoberta.yaml --name fold-4-twitter-ws-128-clip-emoberta-learned --group twitter-ablation --dataset twitter --fold 4 --window_size 128 --position_embeddings learned --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/clip_emoberta.yaml --name fold-4-twitter-ws-128-clip-emoberta-zero --group twitter-ablation --dataset twitter --fold 4 --window_size 128 --position_embeddings zero --mode run --epochs 200 --batch_size 256

#### ROBERTA
python main.py --config_file configs/combos/clip_roberta.yaml --name fold-4-twitter-ws-128-clip-roberta-learned --group twitter-ablation --dataset twitter --fold 4 --window_size 128 --position_embeddings learned --mode run --epochs 200 --batch_size 256
python main.py --config_file configs/combos/clip_roberta.yaml --name fold-4-twitter-ws-128-clip-roberta-zero --group twitter-ablation --dataset twitter --fold 4 --window_size 128 --position_embeddings zero --mode run --epochs 200 --batch_size 256


################# REDDIT DINO + EMOBERTA
python main.py  --config_file configs/combos/dino_emoberta.yaml --name reddit-ws-128-dino-emoberta-learned  --group $GROUP --dataset reddit --window_size 128 --position_embeddings learned --mode run --epochs $EPOCHS --batch_size 256
python evaluate.py  --config_file configs/combos/dino_emoberta.yaml --name reddit-ws-128-dino-emoberta-learned  --group $GROUP --dataset reddit --window_size 128 --position_embeddings learned  --output_dir $GROUP

python main.py  --config_file configs/combos/dino_emoberta.yaml --name reddit-ws-128-dino-emoberta-zero  --group $GROUP --dataset reddit --window_size 128 --position_embeddings zero --mode run --epochs $EPOCHS --batch_size 256
python evaluate.py  --config_file configs/combos/dino_emoberta.yaml --name reddit-ws-128-dino-emoberta-zero  --group $GROUP --dataset reddit --window_size 128 --position_embeddings zero  --output_dir $GROUP

python main.py  --config_file configs/combos/dino_emoberta.yaml --name reddit-ws-128-dino-emoberta-time2vec  --group $GROUP --dataset reddit --window_size 128 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python evaluate.py  --config_file configs/combos/dino_emoberta.yaml --name reddit-ws-128-dino-emoberta-time2vec  --group $GROUP --dataset reddit --window_size 128 --position_embeddings time2vec  --output_dir $GROUP
##########################################

################# REDDIT DINO + ROBERTA
python main.py  --config_file configs/combos/dino_roberta.yaml --name reddit-ws-128-dino-roberta-learned  --group $GROUP --dataset reddit --window_size 128 --position_embeddings learned --mode run --epochs $EPOCHS --batch_size 256
python evaluate.py  --config_file configs/combos/dino_roberta.yaml --name reddit-ws-128-dino-roberta-learned  --group $GROUP --dataset reddit --window_size 128 --position_embeddings learned  --output_dir $GROUP

python main.py  --config_file configs/combos/dino_roberta.yaml --name reddit-ws-128-dino-roberta-zero  --group $GROUP --dataset reddit --window_size 128 --position_embeddings zero --mode run --epochs $EPOCHS --batch_size 256
python evaluate.py  --config_file configs/combos/dino_roberta.yaml --name reddit-ws-128-dino-roberta-zero  --group $GROUP --dataset reddit --window_size 128 --position_embeddings zero  --output_dir $GROUP

python main.py  --config_file configs/combos/dino_roberta.yaml --name reddit-ws-128-dino-roberta-time2vec  --group $GROUP --dataset reddit --window_size 128 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python evaluate.py  --config_file configs/combos/dino_roberta.yaml --name reddit-ws-128-dino-roberta-time2vec  --group $GROUP --dataset reddit --window_size 128 --position_embeddings time2vec  --output_dir $GROUP
##########################################

################# REDDIT DINO + MINILM
python main.py  --config_file configs/combos/dino_minilm.yaml --name reddit-ws-128-dino-minilm-learned  --group $GROUP --dataset reddit --window_size 128 --position_embeddings learned --mode run --epochs $EPOCHS --batch_size 256
python evaluate.py  --config_file configs/combos/dino_minilm.yaml --name reddit-ws-128-dino-minilm-learned  --group $GROUP --dataset reddit --window_size 128 --position_embeddings learned  --output_dir $GROUP

python main.py  --config_file configs/combos/dino_minilm.yaml --name reddit-ws-128-dino-minilm-zero  --group $GROUP --dataset reddit --window_size 128 --position_embeddings zero --mode run --epochs $EPOCHS --batch_size 256
python evaluate.py  --config_file configs/combos/dino_minilm.yaml --name reddit-ws-128-dino-minilm-zero  --group $GROUP --dataset reddit --window_size 128 --position_embeddings zero  --output_dir $GROUP

python main.py  --config_file configs/combos/dino_minilm.yaml --name reddit-ws-128-dino-minilm-time2vec  --group $GROUP --dataset reddit --window_size 128 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python evaluate.py  --config_file configs/combos/dino_minilm.yaml --name reddit-ws-128-dino-minilm-time2vec  --group $GROUP --dataset reddit --window_size 128 --position_embeddings time2vec  --output_dir $GROUP
##########################################

################# REDDIT CLIP + ROBERTA
python main.py  --config_file configs/combos/clip_roberta.yaml --name reddit-ws-128-clip-roberta-learned  --group $GROUP --dataset reddit --window_size 128 --position_embeddings learned --mode run --epochs $EPOCHS --batch_size 256
python evaluate.py  --config_file configs/combos/clip_roberta.yaml --name reddit-ws-128-clip-roberta-learned  --group $GROUP --dataset reddit --window_size 128 --position_embeddings learned  --output_dir $GROUP

python main.py  --config_file configs/combos/clip_roberta.yaml --name reddit-ws-128-clip-roberta-zero  --group $GROUP --dataset reddit --window_size 128 --position_embeddings zero --mode run --epochs $EPOCHS --batch_size 256
python evaluate.py  --config_file configs/combos/clip_roberta.yaml --name reddit-ws-128-clip-roberta-zero  --group $GROUP --dataset reddit --window_size 128 --position_embeddings zero  --output_dir $GROUP

python main.py  --config_file configs/combos/clip_roberta.yaml --name reddit-ws-128-clip-roberta-time2vec  --group $GROUP --dataset reddit --window_size 128 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python evaluate.py  --config_file configs/combos/clip_roberta.yaml --name reddit-ws-128-clip-roberta-time2vec  --group $GROUP --dataset reddit --window_size 128 --position_embeddings time2vec  --output_dir $GROUP
##########################################

################# REDDIT CLIP + MINILM
python main.py  --config_file configs/combos/clip_minilm.yaml --name reddit-ws-128-clip-minilm-learned  --group $GROUP --dataset reddit --window_size 128 --position_embeddings learned --mode run --epochs $EPOCHS --batch_size 256
python evaluate.py  --config_file configs/combos/clip_minilm.yaml --name reddit-ws-128-clip-minilm-learned  --group $GROUP --dataset reddit --window_size 128 --position_embeddings learned  --output_dir $GROUP

python main.py  --config_file configs/combos/clip_minilm.yaml --name reddit-ws-128-clip-minilm-zero  --group $GROUP --dataset reddit --window_size 128 --position_embeddings zero --mode run --epochs $EPOCHS --batch_size 256
python evaluate.py  --config_file configs/combos/clip_minilm.yaml --name reddit-ws-128-clip-minilm-zero  --group $GROUP --dataset reddit --window_size 128 --position_embeddings zero  --output_dir $GROUP

python main.py  --config_file configs/combos/clip_minilm.yaml --name reddit-ws-128-clip-minilm-time2vec  --group $GROUP --dataset reddit --window_size 128 --position_embeddings time2vec --mode run --epochs $EPOCHS --batch_size 256
python evaluate.py  --config_file configs/combos/clip_minilm.yaml --name reddit-ws-128-clip-minilm-time2vec  --group $GROUP --dataset reddit --window_size 128 --position_embeddings time2vec  --output_dir $GROUP
##########################################
