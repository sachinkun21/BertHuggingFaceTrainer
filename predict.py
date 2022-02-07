import numpy as np

import joblib
import torch

import config
import dataset
import engine
from model import NERModel


if __name__=='__main__':
    sentence = 'My name is Real Slim Shady.'
    meta_data = joblib.load(config.LABEL_ENCODER_PATH)
    enc_pos = meta_data["enc_pos"]
    enc_tag = meta_data["enc_tag"]

    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    tokenized_sentence = config.TOKENIZER.encode(sentence)

    sentence = sentence.split()
    print(sentence)
    print(tokenized_sentence)

    test_dataset = dataset.EntityDataset(
        texts=[sentence],
        pos=[[0] * len(sentence)],
        tag=[[0] * len(sentence)]
    )

    device = torch.device(config.DEVICE)
    model = NERModel(num_tag=num_tag, num_pos=num_pos)
    model.load_state_dict(torch.load(config.BASE_MODEL_PATH))
    model.to(device)

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag, pos, _ = model(**data)

        print(
            enc_tag.inverse_transform(
                tag.argmax(2).cpu().numpy().reshape(-1)
            )[:len(tokenized_sentence)]
        )
        print(
            enc_pos.inverse_transform(

                pos.argmax(2).cpu().numpy().reshape(-1)
            )[:len(tokenized_sentence)]
        )
