import torch
# import wandb
import tqdm
import decoder
import utils
import dataset


batch_size = 64
gradient_accumulation_steps = 8
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

print("Loading dataset...")
train_dataset = dataset.FlickrClipDataset(dataset.train_ds)
# val_dataset = dataset.Flickr30kDataset(seed=seed, split='val')

print("Loading dataloader...")
train_dataloader = torch.utils.data.DataLoader(train_dataset
                                , batch_size=batch_size
                                , shuffle=True
                                , collate_fn=dataset._collate_fn)
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size/10, num_workers=0)


# Replace the loss_fn definition
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

optimiser = torch.optim.Adam(model.parameters(), lr=initial_lr)

# ReduceLROnPlateau - reduces LR when validation loss stops improving
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser,
    mode='min',
    factor=0.5,  # multiply LR by this factor when reducing
    patience=2,   # number of epochs to wait before reducing LR
)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1} started")
    total_loss = 0
    num_batches = 0
    total_correct = 0
    total_samples = 0
    
    model.train()
    optimiser.zero_grad()  # Zero gradients at start of epoch
    
    prgs = tqdm.tqdm((train_dataloader), desc=f"Epoch {epoch+1}", leave=False)
    for batch_idx, batch in enumerate(prgs):
        img_emb = batch[0].to(device)
        caption_in_emb = batch[1].to(device)
        caption_out = batch[3].to(device)

        # Forward pass
        logits = model(caption_in_emb, img_emb)
        logits = logits.view(-1, vocab_size)
        caption_out = caption_out.reshape(-1)
        # caption_out = caption_out.view(-1)
        
        # Create mask for non-padding tokens
        mask = (caption_out != 0)
        logits = logits[mask]
        caption_out = caption_out[mask]
        
        # Calculate loss
        loss = loss_fn(logits, caption_out) / gradient_accumulation_steps  # Scale loss
        loss.backward()
        
        # Calculate accuracy
        with torch.no_grad():
            _, predicted = torch.max(logits, dim=-1)
            correct = (predicted == caption_out).sum().item()
            total_correct += correct
            total_samples += mask.sum().item()
            total_loss += loss.item() * gradient_accumulation_steps  # Unscale loss for logging
        
        # Update weights every gradient_accumulation_steps batches
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            optimiser.zero_grad()
            num_batches += 1
        
        # Update progress bar
        prgs.set_postfix({
            'loss': loss.item() * gradient_accumulation_steps,
            'acc': correct / mask.sum().item()
        })

    # Handle any remaining gradients at end of epoch
    if (batch_idx + 1) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        optimiser.zero_grad()
        num_batches += 1

    avg_loss = total_loss / num_batches
    epoch_accuracy = total_correct / total_samples

    # Step the scheduler
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(avg_loss)
        print(f"Current LR: {scheduler.optimizer.param_groups[0]['lr']}")

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Acc: {epoch_accuracy:.2%}")

print("Saving model...")
torch.save(model.state_dict(), "weights/decoder.pt")
