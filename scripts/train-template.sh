
python ${ROOT_DIR}/train.py --data ${DATA_YAML} \
                --cfg ${MODEL_CFG} \
                --batch-size ${BATCH_SIZE} \
                --imgsz ${TRAIN_IMSIZE} \
                --hyp ${HYP_YAML} \
                --project ${PROJ_DIR} \
                --name ${EXP_NAME} \
                --weights ${PROJ_DIR}/${EXP_NAME}/${INIT_WEIGHTS}  \
                --epochs ${EPOCHS}  2>&1 | tee ${PROJ_DIR}/${EXP_NAME}/info/out.log