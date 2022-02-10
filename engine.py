from tqdm import tqdm


def train_fn(data_loader, model, optimizer, device, scheduler, epoch, wandb):
    # setting model to train mode
    init_step = 1
    model.train()
    final_loss = 0
    step_size = len(data_loader)
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

        # logging batch level loss info
        wandb.log({"Average Train Batch loss": final_loss/init_step})
        init_step += 1

    return final_loss/len(data_loader)


def eval_fn(data_loader, model, device, epoch, wandb):
    # setting model to eval mode
    init_step = 1
    model.eval()
    eval_loss = 0
    step_size = len(data_loader)
    for data in tqdm(data_loader, total=step_size):
        for k, v in data.items():
            data[k] = v.to(device)
        _, _, loss = model(**data)
        eval_loss += loss.item()

        # logging batch level loss info
        wandb.log({"Average Val Batch loss": eval_loss / init_step})
        init_step += 1

    return eval_loss/len(data_loader)

