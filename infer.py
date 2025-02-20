import matplotlib.pyplot as plt
import torch
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
model.load_state_dict(torch.load("weights/flicker-captioning-final_2025_02_20__16_41_54.pt", map_location=device, weights_only=True))

# count number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

model.to(device)
model.eval()

def generate_caption(model, image, temperature=0.7, max_length=77):
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Get image embedding using CLIP
        processor_output = dataset.clip_processor(images=image)
        img_tensor = torch.tensor(processor_output.pixel_values).to(device)
        img_emb = dataset.clip_model.get_image_features(img_tensor)
        img_emb = img_emb.unsqueeze(1)  # Add sequence dimension
        
        # Initialize with just the image embedding
        current_output = img_emb
        generated_tokens = []
        
        # Generate tokens one at a time
        for _ in range(max_length):
            # Forward pass through decoder
            logits = model(current_output)[:, -1, :]  # Get predictions for next token
            
            # Apply temperature scaling
            scaled_logits = logits / temperature
            
            # Convert to probabilities
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to generated sequence
            generated_tokens.append(next_token.item())
            
            # Stop if we generate the end token
            if next_token.item() == dataset.clip_processor.tokenizer.eos_token_id:
                break
                
            # Add token embedding for next iteration
            next_token_emb = dataset.clip_model.text_model.embeddings.token_embedding(next_token)
            current_output = torch.cat([current_output, next_token_emb], dim=1)
        
        # Decode the generated tokens
        caption = dataset.clip_processor.decode(generated_tokens, skip_special_tokens=True)
        return caption

def main():
    # Create dataset wrapper
    val_dataset = dataset.FlickrClipDataset(dataset.val_ds)
    
    # Test on a few images from validation set
    test_indices = [0, 10, 20, 30]  # Test first few images
    temperatures = [0.7, 0.8, 0.9]  # Try different temperatures
    
    for idx in test_indices:
        # Get item through the wrapper
        photo, caption = val_dataset[idx]
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(photo)
        plt.axis('off')
        plt.title('Input Image')
        
        print(f"\nImage {idx}:")
        print(f"True caption: {caption}")
        print("\nGenerated captions at different temperatures:")
        
        for temp in temperatures:
            generated = generate_caption(model, photo, temperature=temp)
            print(f"T={temp:.1f}: {generated}")
        
        plt.show()

if __name__ == "__main__":
    main()