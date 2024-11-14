from tqdm import tqdm
import torch

import config

def train_fn(model, data_loader, optimizer):
    model.train()
    fin_loss = 0
    tk0 = tqdm(data_loader, total=len(data_loader))
    for data in tk0:
        for k, v in data.items():
            data[k] = v.to(config.DEVICE)

        targets = data['targets']
        # Used in the model to find the length of each target
        targets_mask = targets != -1

        optimizer.zero_grad()

        _, loss = model(images=data['images'], labels=targets, labels_mask=targets_mask)

        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
    return fin_loss / len(data_loader)


def eval_fn(model, data_loader):
    model.eval()
    fin_loss = 0
    fin_preds = []
    tk0 = tqdm(data_loader, total=len(data_loader))
    with torch.no_grad():
        for data in tk0:
            for k, v in data.items():
                data[k] = v.to(config.DEVICE)

            targets = data['targets']
            # Used in the model to find the length of each target
            
            targets_mask = targets != -1
            batch_preds, loss = model(images=data['images'], labels=targets, labels_mask=targets_mask)
            fin_loss += loss.item()
            fin_preds.append(batch_preds)
    return fin_preds, fin_loss / len(data_loader)