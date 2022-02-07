import transformers
import torch

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 16
EPOCHS = 5
BASE_MODEL_PATH = "./base_model/bert-base-uncased"
SAVE_MODEL_PATH = "pytorch_model.bin"
TRAIN_DATA_PATH = "./dataset/ner_dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH, do_lower_case=True if BASE_MODEL_PATH.endswith("uncased") else False
)
LABEL_ENCODER_PATH = "label_encoder.joblib"
DEVICE = 'cpu'

