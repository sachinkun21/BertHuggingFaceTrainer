import torch
from tqdm import tqdm


def train_fn(data_loader, model, optimizer, device, scheduler):
    # setting model to train mode
    model.train()
    final_loss = 0

    # starting the training process
    for data in tqdm(data_loader, total=len(data_loader)):
        # setting data to device
        for k, v in data.items():
            data[k] = v.to(device)

        optimizer.zero_grad()
        _, _, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()

    return final_loss/len(data_loader)


def eval_fn(data_loader, model, device):
    # setting model to eval mode
    model.eval()
    eval_loss = 0

    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
            _, _, loss = model(**data)
        eval_loss += loss.item()

    return eval_loss/len(data_loader)



