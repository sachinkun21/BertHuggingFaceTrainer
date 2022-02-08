import transformers
import torch

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 16
EPOCHS = 5
BASE_MODEL_PATH = "dvcfiles/base_model/bert-base-uncased"
SAVE_MODEL_PATH = "dvcfiles/finetuned_model/pytorch_model.bin"
TRAIN_DATA_PATH = "dvcfiles/dataset/ner_dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH, do_lower_case=True if BASE_MODEL_PATH.endswith("uncased") else False
)
LABEL_ENCODER_PATH = "label_encoder.joblib"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

