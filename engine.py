import torch
from tqdm import tqdm

import config

train_batch = 0
def train_fn(data_loader, model, optimizer, device, scheduler, epoch, wandb):
    # setting model to train mode
    global train_batch
    model.train()
    final_loss = 0
    # starting the training process
    for data in tqdm(data_loader, total=step_size):
        # setting data to device
        for k, v in data.items():
            data[k] = v.to(device)

        optimizer.zero_grad()
        _, _, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()

        # logging step level loss info
        wandb.log({"Train Step": epoch, "TrainSteploss": final_loss}, step=train_batch)
        train_batch += 1

    return final_loss/len(data_loader)


eval_batch = 0
def eval_fn(data_loader, model, device, epoch, wandb):
    # setting model to eval mode
    global eval_batch
    model.eval()
    eval_loss = 0
    for data in tqdm(data_loader, total=step_size):
        for k, v in data.items():
            data[k] = v.to(device)
        _, _, loss = model(**data)
        eval_loss += loss.item()

        # logging step level loss info
        wandb.log({"Train Step": epoch, "TrainSteploss": eval_loss}, step=eval_batch)
        eval_batch += 1
    return eval_loss/len(data_loader)

