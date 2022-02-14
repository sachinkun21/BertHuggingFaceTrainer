import joblib
import pandas as pd
import numpy as np
import warnings

import torch

from sklearn import model_selection
from sklearn import preprocessing

import config
import engine
import dataset
from model import NERModel


from transformers import logging, AdamW
from transformers import get_linear_schedule_with_warmup

# removing warning messages
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

import wandb

# setting up wandb for tracking logs

def process_data(path):
    print(f"--> Reading dataset from {path} for training")
    df = pd.read_csv(path, encoding='latin-1')
    df['Sentence #'].fillna(method='ffill', inplace=True)

    print(f"--> Fitting LabelEncoder on entities and pos")
    enc_pos = preprocessing.LabelEncoder()
    enc_tag = preprocessing.LabelEncoder()

    # label encoding pos and tags
    df.loc[:,'POS'] = enc_pos.fit_transform(df['POS'])
    df.loc[:, 'Tag'] = enc_tag.fit_transform(df["Tag"])

    # collating sentence, tag and pos
    sent_list = df.groupby('Sentence #')['Word'].apply(np.array).values
    pos_list = df.groupby('Sentence #')['POS'].apply(np.array).values
    tag_list = df.groupby('Sentence #')['Tag'].apply(np.array).values

    return sent_list, pos_list, tag_list, enc_pos, enc_tag


if __name__ == '__main__':
    wandb.init(project="BERTNerHF", entity="sachinkun21")

    path = config.TRAIN_DATA_PATH
    sent_list, pos_list, tag_list, enc_pos, enc_tag = process_data(path)

    # saving label-encoder for inference-time
    print(f"--> Saving the labelEncoder at {config.LABEL_ENCODER_PATH} for inference.")
    label_encoders = {'enc_pos': enc_pos, "enc_tag": enc_tag}
    joblib.dump(label_encoders, config.LABEL_ENCODER_PATH)

    num_pos = len(enc_pos.classes_)
    num_tag = len(enc_tag.classes_)
    print(f"--> Number of NER Classes {num_tag}\t\tNumber of POS Classes {num_pos}")

    (train_sent,
     test_sent,
     train_pos,
     test_pos,
     train_tag,
     test_tag
     ) = model_selection.train_test_split(sent_list, pos_list, tag_list,
                                          test_size=0.2, random_state=42)

    train_dataset = dataset.EntityDataset(train_sent, train_pos, train_tag)
    valid_dataset = dataset.EntityDataset(test_sent, test_pos, test_tag)

    print(f"--> Creating Dataloaders with TRAIN_BATCH_SIZE: {config.TRAIN_BATCH_SIZE} and VAL_BATCH_SIZE: {config.VAL_BATCH_SIZE}")
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=config.TRAIN_BATCH_SIZE,
                                                   num_workers=1)

    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=config.VAL_BATCH_SIZE,
                                                   num_workers=1)

    # configuring model
    device = torch.device(config.DEVICE)
    print(f"--> Using {device} for training")
    print("--> Initializing the NERModel and setting optimizer")
    model = NERModel(num_tag, num_pos)
    model.to(device)

    # configuring optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_params = [
        {"params": [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01, },
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.00, }
        ]

    num_train_steps = int(len(train_sent)/config.TRAIN_BATCH_SIZE*config.EPOCHS)
    learning_rate = 3e-5
    optimizer = AdamW(optimizer_params, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    # starting the trainer
    print(f"--> Starting training for {config.EPOCHS} Epochs: ")
    best_loss = np.inf

    wandb.config = {
        "model":config.BASE_MODEL_PATH,
        "learning_rate": learning_rate,
        "epochs": config.EPOCHS,
        "train_batch_size": config.TRAIN_BATCH_SIZE,
        "test_batch_size": config.VAL_BATCH_SIZE
    }

    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(train_dataloader, model, optimizer, device, scheduler, epoch, wandb)
        val_loss = engine.eval_fn(valid_dataloader, model,  device,  epoch, wandb)
        print(f'Epoch: {epoch}       TrainLoss: {train_loss}       ValLoss: {val_loss}')

        # logging loss to wandb
        wandb.log({"Epoch": epoch, "Epoch_Train_Loss": train_loss})
        wandb.log({"Epoch": epoch, "Epoch_Val_Loss": val_loss})

        if val_loss < best_loss:
            print(f"--> Saving Model at {config.SAVE_MODEL_PATH} with loss value as {val_loss}")
            torch.save(model.state_dict(), config.SAVE_MODEL_PATH)
            best_loss = val_loss

#{Word:tags[i] for i iqan range(len(tokenize(word))) for word in sentence.split(" ")]}