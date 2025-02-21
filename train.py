import torch
import wandb
import tqdm
import decoder
import utils
import dataset
import torch.multiprocessing as mp
import datetime

ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')


def validate(model, val_dataloader, loss_fn, device):
    model.eval()
    total_val_loss = 0
    total_val_correct = 0
    total_val_samples = 0
    
    with torch.no_grad():
        for batch in tqdm.tqdm(val_dataloader, desc="Validating", leave=False):
            combined_outputs, attention_mask, caption_out = [t.to(device) for t in batch]
            
            # Forward pass
            logits = model(combined_outputs)
            
            # Mask and loss calculation
            mask = (attention_mask != 0)
            loss = loss_fn(logits[mask], caption_out[mask])
            
            # Metrics
            _, predicted = torch.max(logits, dim=-1)
            correct = (predicted[mask] == caption_out[mask]).sum().item()
            total_val_correct += correct
            total_val_samples += mask.sum().item()
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_dataloader)
    val_accuracy = total_val_correct / total_val_samples
    
    return avg_val_loss, val_accuracy


def train(load_state_dict=None):
    batch_size = 64
    epochs = 20
    initial_lr = 0.0005
    d_model = 512
    dff = 2048
    device = utils.get_device()
    vocab_size = len(utils.get_vocab())
    dropout = 0.1
    seed = 2

    # Add these parameters
    gradient_accumulation_steps = 4
    
    torch.manual_seed(seed)
    print("Device:", device)

    print("Initialising model...")
    model = decoder.Decoder(di_initial=512, d_model=d_model, dff=dff, vocab_size=vocab_size, dropout=dropout)

    # Load model state dict if provided
    if load_state_dict:
        model.load_state_dict(torch.load(load_state_dict))

    model.to(device)

    params = sum(p.numel() for p in model.parameters())
    print("param count:", params)

    wandb.init(
        project="mlx6-flicker-captioning",
        config={
            "learning_rate": 'scheduler',
            "epochs": epochs,
            "params": params,
            "decoder_layers": 12,
            "d_model": d_model,
            "dff": dff,
            "dropout": dropout,
        },
    )

    print("Loading dataset and dataloader...")
    train_dataloader = torch.utils.data.DataLoader(
        dataset.FlickrClipDataset(dataset.train_ds),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset._collate_fn,
        num_workers=4,
        persistent_workers=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset.FlickrClipDataset(dataset.val_ds),
        batch_size=batch_size//2,
        shuffle=True,
        collate_fn=dataset._collate_fn,
        num_workers=4,
        persistent_workers=True
    )

    # Simplified loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} started")
        # Training phase
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        prgs = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False)
        
        for i, batch in enumerate(prgs):
            optimiser.zero_grad()
            combined_outputs, attention_mask, caption_out = [t.to(device) for t in batch]

            # Forward pass
            logits = model(combined_outputs)

            mask = (attention_mask != 0)

            loss = loss_fn(logits[mask], caption_out[mask]) / gradient_accumulation_steps
            loss.backward()
            
            if (i + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                optimiser.zero_grad()

            # Metrics
            with torch.no_grad():
                _, predicted = torch.max(logits, dim=-1)
                correct = (predicted[mask] == caption_out[mask]).sum().item()
                total_correct += correct
                total_samples += mask.sum().item()
                total_loss += loss.item()

            prgs.set_postfix({'loss': loss.item(), 'acc': correct/mask.sum().item()})
            wandb.log({
                "batch_train_loss": loss.item(), 
                "batch_train_accuracy": correct / mask.sum().item()
            })

        avg_train_loss = total_loss / len(train_dataloader)
        train_accuracy = total_correct / total_samples
        
        # Validation phase
        avg_val_loss, val_accuracy = validate(model, val_dataloader, loss_fn, device)
        
        # Update learning rate scheduler with validation loss
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best validation loss: {best_val_loss:.4f}")
            
            # Save model locally
            best_model_path = f"weights/flicker-captioning-best_{epoch+1}_{ts}.pt"
            torch.save(model.state_dict(), best_model_path)
            
            # Create NEW artifact for each best model
            best_artifact = wandb.Artifact(f"best-flicker-captioning-{epoch+1}", type="decoder-model")
            best_artifact.add_file(best_model_path)
            wandb.log_artifact(best_artifact)

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2%}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2%}")
        print(f"Current LR: {optimiser.param_groups[0]['lr']}")

        # Log metrics
        wandb.log({
            "learning_rate": optimiser.param_groups[0]['lr'],
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "epoch": epoch
        })

        # Create NEW artifact for each epoch
        model_checkpoint_path = f"weights/flicker-captioning-best_{epoch+1}_{ts}.pt"
        torch.save(model.state_dict(), model_checkpoint_path)
        model_checkpoint_artifact = wandb.Artifact(f"checkpoint-flicker-captioning-{epoch+1}", type="decoder-model")
        model_checkpoint_artifact.add_file(model_checkpoint_path)
        wandb.log_artifact(model_checkpoint_artifact)

    # After training completes, log final model separately
    print("Saving final model...")
    final_model_path = f"weights/flicker-captioning-final_{ts}.pt"
    torch.save(model.state_dict(), final_model_path)
    final_artifact = wandb.Artifact("final-flicker-captioning", type="decoder-model")
    final_artifact.add_file(final_model_path)
    wandb.log_artifact(final_artifact)
    
    print("Done!")
    wandb.finish()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    train(load_state_dict="weights/flicker-captioning-final.pt")
