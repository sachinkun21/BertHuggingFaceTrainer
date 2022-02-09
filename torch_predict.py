import joblib
import torch

import config
import dataset
from model import NERModel


# loading labelEncoders
meta_data = joblib.load(config.LABEL_ENCODER_PATH)
enc_pos = meta_data["enc_pos"]
enc_tag = meta_data["enc_tag"]
num_pos = len(list(enc_pos.classes_))
num_tag = len(list(enc_tag.classes_))

# loading model
device = torch.device(config.DEVICE)
model = NERModel(num_tag=num_tag, num_pos=num_pos)
model.load_state_dict(torch.load(config.SAVE_MODEL_PATH, map_location=device))
model.to(device)


def predict(sentence):
    tokenized_sentence = config.TOKENIZER.encode(sentence)
    sentence = sentence.split()

    test_dataset = dataset.EntityDataset(
        texts=[sentence],
        pos=[[0] * len(sentence)],
        tag=[[0] * len(sentence)]
    )

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)

        # infenence from model
        pos, tag, _ = model(**data)

        # decoding predicted tags and predicted POS
        pred_tags = enc_tag.inverse_transform(
            tag.argmax(2).cpu().numpy().reshape(-1)
        )[:len(tokenized_sentence)]

        pred_pos = enc_pos.inverse_transform(
                pos.argmax(2).cpu().numpy().reshape(-1)
            )[:len(tokenized_sentence)]

        # creating final output by mapping tags and pos to words
        dict_output = {}
        index = 0
        for word in sentence:
            ids = (config.TOKENIZER(word, add_special_tokens=False))
            dict_output[word] = [pred_tags[index+1], pred_pos[index+1]]
            index += len(ids['input_ids'])

        return dict_output


if __name__ == '__main__':
    sentence = 'I was in New York last night with jason to attend ICML conference.'
    output = predict(sentence)
    print(output)
