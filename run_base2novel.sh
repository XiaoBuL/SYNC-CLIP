TRAIN=1
DATASET=$1
NUM_SHOTS=16
DEVICE=$2
SEED=$3
SYN_SHOTS=16
TRAINER=SynCLIP
CFG=vit_b16_c2_ep20_batch4_4+4ctx
if [ $TRAIN -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python train.py \
    --root /data/weijie/dataset \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ./output/base2new/train_base/${DATASET}/shots_${NUM_SHOTS}/${TRAINER}/${CFG}/seed${SEED} \
    TRAINER.SynCLIP.SYN_SHOTS ${SYN_SHOTS} \
    DATASET.NUM_SHOTS ${NUM_SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
else
    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python train.py \
    --root /data/weijie/dataset \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ./output/base2new/test_new/${DATASET}/shots_${NUM_SHOTS}/${TRAINER}/${CFG}/seed${SEED} \
    --model-dir ./output/base2new/train_base/${DATASET}/shots_${NUM_SHOTS}/${TRAINER}/${CFG}/seed${SEED} \
    --load-epoch 10 \
    --eval-only \
    TRAINER.SynCLIP.SYN_SHOTS ${SYN_SHOTS} \
    DATASET.NUM_SHOTS ${NUM_SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi
