import random

import torch
import torch.nn.functional as F
import decoder
import dataset
import utils



d_model = 512
dff = 2048
device = utils.get_device()
vocab = utils.get_vocab()
vocab_size = len(vocab)
dropout = 0.1
seed = 2

model = decoder.Decoder(di_initial=512, d_model=d_model, dff=dff, vocab_size=vocab_size, dropout=dropout)

print("Loading model...")
model.load_state_dict(torch.load("weights/flicker-captioning-best.pt", map_location=device, weights_only=True))

# count number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

model.to(device)
model.eval()

with torch.no_grad():
    val_ds = dataset.FlickrClipDataset(dataset.val_ds)
    # Test DataLoader with custom collate function
    dataloader = torch.utils.data.DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False,
        collate_fn=dataset._collate_fn  # Re-enable the custom collate_fn
    )
    batch = next(iter(dataloader))
    _, caption = val_ds.__getitem__(0)

    print("Caption: ",caption)
    max_tokens = 77
    prediction_sequence = []

    # Create start token (usually 10 for start token)
    token_seq = batch[0]  # Shape: [1, 1]
    
    with torch.no_grad():
        for n in range(1):
            print(f"\nCurrent sequence: {token_seq}")
            
            # Make prediction
            logits = model(token_seq)
            
            # Get prediction for next token only
            prediction = torch.argmax(logits[0, n]).item()
            print(prediction)
            
            word = vocab[prediction]
            print(word)

            prediction_sequence.append(word)
            print(f"Predicted so far: {prediction_sequence}")

            if prediction == 49407: # End token
                break
            
            prediction_emb = dataset.get_token_embedding(prediction)
            # Add prediction to token sequence
            token_seq = torch.cat([token_seq, torch.tensor([[prediction_emb]], device=device)], dim=1)
            
            if token_seq.size(1) >= max_tokens:
                break
    
    print("Prediction: ",prediction_sequence)