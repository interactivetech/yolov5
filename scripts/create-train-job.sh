ROOT_DIR=/run/determined/workdir/yolov5
JOB_DIR=/run/determined/workdir/jobs
EXP_NAME=test-train-200-epochs
PROJ_DIR=/run/determined/workdir/exps
DATA_YAML=/run/determined/workdir/virat-aerial-156-frames-v2-coco-yolov5/data.yml
MODEL_CFG=${ROOT_DIR}/models/yolov5x.yaml
BATCH_SIZE=8
TRAIN_IMSIZE=640
HYP_YAML=${ROOT_DIR}/data/hyps/hyp.scratch-low.yaml
INIT_WEIGHTS=yolov5x6.pt
EPOCHS=200
if [ ! -f "${JOB_DIR}" ]; then
    echo "${JOB_DIR} does not exist."
    mkdir ${JOB_DIR}
    echo "created ${JOB_DIR}"
fi
if [ ! -f "${JOB_DIR}/${EXP_NAME}" ]; then
    echo "${JOB_DIR}/${EXP_NAME} does not exist."
    mkdir ${JOB_DIR}/${EXP_NAME}
    echo "created ${JOB_DIR}/${EXP_NAME}"
fi

if [ ! -f "${PROJ_DIR}" ]; then
    echo "${PROJ_DIR} does not exist."
    mkdir ${PROJ_DIR}
    echo "created ${PROJ_DIR}"
fi

if [ ! -f "${PROJ_DIR}/${EXP_NAME}" ]; then
    echo "${PROJ_DIR}/${EXP_NAME} does not exist."
    mkdir ${PROJ_DIR}/${EXP_NAME}
    echo "created ${PROJ_DIR}/${EXP_NAME}"
fi

if [ ! -f "${PROJ_DIR}/${EXP_NAME}/info/" ]; then
    echo "${PROJ_DIR}/${EXP_NAME}/info/ does not exist."
    mkdir ${PROJ_DIR}/${EXP_NAME}/info/
    echo "created ${PROJ_DIR}/${EXP_NAME}/info/"
fi
sed -e "s|\${ROOT_DIR}|${ROOT_DIR}|" \
    -e "s|\${EXP_NAME}|${EXP_NAME}|" \
    -e "s|\${PROJ_DIR}|${PROJ_DIR}|" \
    -e "s|\${DATA_YAML}|${DATA_YAML}|" \
    -e "s|\${MODEL_CFG}|${MODEL_CFG}|" \
    -e "s|\${BATCH_SIZE}|${BATCH_SIZE}|" \
    -e "s|\${TRAIN_IMSIZE}|${TRAIN_IMSIZE}|" \
    -e "s|\${HYP_YAML}|${HYP_YAML}|" \
    -e "s|\${INIT_WEIGHTS}|${INIT_WEIGHTS}|" \
    -e "s|\${EPOCHS}|${EPOCHS}|" \
   train-template.sh > ${JOB_DIR}/${EXP_NAME}/train.sh