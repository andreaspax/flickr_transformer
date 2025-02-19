import torch
# import wandb
import tqdm
import decoder
import utils
import dataset
import os
import torch.multiprocessing as mp


batch_size = 126
epochs = 1
initial_lr = 0.001
d_model = 512
dff = 2048
device = utils.get_device()
vocab_size = len(utils.get_vocab())
dropout = 0.1
seed = 2

torch.manual_seed(seed)

print("Initialising model...")
model = decoder.Decoder(di_initial=512, d_model=d_model, dff=dff, vocab_size=vocab_size, dropout=dropout)
model.to(device)

params = sum(p.numel() for p in model.parameters())
print("param count:", params)

# wandb.init(
#     project="mlx6-mnist-transformer",
#     config={
#         "learning_rate": 'scheduler',
#         "epochs": epochs,
#         "params": params,
#         "encoder_layers": 3,
#         "decoder_layers": 3,
#         "d_model": d_model,
#         "dff": dff,
#         "dropout": dropout,
#     },
# )

print("Loading dataset and dataloader...")
train_dataloader = torch.utils.data.DataLoader(
    dataset.FlickrClipDataset(dataset.train_ds),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=dataset._collate_fn
)
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size/10, num_workers=0)


# Simplified loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
optimiser = torch.optim.Adam(model.parameters(), lr=initial_lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=2)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1} started")
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    model.train()
    prgs = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False)
    
    for batch in prgs:
        optimiser.zero_grad()
        img_emb, caption_in_emb, _, caption_out = [t.to(device) for t in batch]

        # Forward pass
        logits = model(caption_in_emb, img_emb)
        logits = logits.view(-1, vocab_size)
        caption_out = caption_out.reshape(-1)

        # Mask and loss calculation
        mask = (caption_out != 0)
        loss = loss_fn(logits[mask], caption_out[mask])
        loss.backward()
        
        # Update weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        # Metrics
        with torch.no_grad():
            _, predicted = torch.max(logits, dim=-1)
            correct = (predicted[mask] == caption_out[mask]).sum().item()
            total_correct += correct
            total_samples += mask.sum().item()
            total_loss += loss.item()

        prgs.set_postfix({'loss': loss.item(), 'acc': correct/mask.sum().item()})

    avg_loss = total_loss / len(train_dataloader)
    epoch_accuracy = total_correct / total_samples
    
    scheduler.step(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Acc: {epoch_accuracy:.2%}")
    print(f"Current LR: {optimiser.param_groups[0]['lr']}")

print("Saving model...")
torch.save(model.state_dict(), "weights/decoder.pt")
