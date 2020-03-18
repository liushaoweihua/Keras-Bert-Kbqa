MODEL_DIR="../models"
DATA_DIR="../data"

python run_deploy.py \
    -model_configs ${MODEL_DIR}/model_configs.json \
    -log_path deploy_log/ \
    -prior_checks ${DATA_DIR}/data/prior_check.txt \
    -database ${DATA_DIR}/data/database.txt \
    -utter_search ${DATA_DIR}/templates/utter_search.txt \
    -device_map "cpu"
