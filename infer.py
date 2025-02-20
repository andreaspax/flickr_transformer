import random
import matplotlib.pyplot as plt
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
model.load_state_dict(torch.load("weights/flicker-captioning-final.pt", map_location=device, weights_only=True))

# count number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

model.to(device)
model.eval()

with torch.no_grad():
    train_ds = dataset.FlickrClipDataset(dataset.train_ds)
    # Test DataLoader with custom collate function
    photo, caption = train_ds.__getitem__(12)

    print("Caption: ",caption)
    max_length = 77
    prediction_sequence = []

    # Create start token (usually 10 for start token)
    tgt_seq = dataset.get_initial_img_embedding(photo).unsqueeze(0)  # Shape: [1, 1]

    
    for _ in range(max_length):
        with torch.no_grad():
            output = model(tgt_seq)  # Predict next token
            next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(0)  # Get most probable token
            next_emb = dataset.get_token_embedding(next_token).unsqueeze(0)

        # Append next_token to prediction sequence
        prediction_sequence.append(next_token.item())

        # Append next token to sequence
        tgt_seq = torch.cat([tgt_seq, next_emb], dim=1)
        
        # Stop if <EOS> token is generated
        if next_token.item() == 49407:
            break

    # Decode token IDs into words
    print(prediction_sequence)

    print(dataset.decode_tokens(prediction_sequence))

    plt.figure(figsize=(11, 6))
    
    # Create two subplots side by side
    plt.subplot(1, 2, 1)
    plt.imshow(photo)
    
    plt.axis('off')
    plt.text(0, -0.3, dataset.decode_tokens(prediction_sequence), 
             va='top', ha='center', fontsize=8,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
    
    plt.tight_layout()
    plt.show()