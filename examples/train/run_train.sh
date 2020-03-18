CONFIG_FILE="train_config.json"
SAVE_PATH="../models"

python run_train.py \
    -config ${CONFIG_FILE} \
    -save_path "../models" \
    -device_map "2"
