import matplotlib.pyplot as plt
import torch
import decoder
import transformers
import utils
import dataset
from PIL import Image

torch.manual_seed(2)

def models():
    device = utils.get_device()
    model = decoder.Decoder(di_initial=512, d_model=512, dff=2048, vocab_size=49408)
    model.load_state_dict(torch.load("weights/flicker-captioning-best_19.pt", map_location=device))
    model.to(device)
    model.eval()
    pretrained_model = "openai/clip-vit-base-patch32"
    clip_model = transformers.CLIPModel.from_pretrained(pretrained_model).to(device)
    clip_processor = transformers.CLIPProcessor.from_pretrained(pretrained_model)
    clip_model.eval()
    return model, clip_model, clip_processor

def generate_caption(image, temperature=0.8, max_length=76):
    model, clip_model, clip_processor = models()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Get image embedding using CLIP
        processor_output = clip_processor(images=image, return_tensors="pt", padding=True, do_rescale=True)
        img_tensor = torch.tensor(processor_output.pixel_values).to(device)
        img_emb = clip_model.get_image_features(img_tensor)
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
            
            # Stop if we generate the end token or period token
            if next_token.item() == clip_processor.tokenizer.eos_token_id or next_token.item() == 269:
                break
                
            # Add token embedding for next iteration
            next_token_emb = clip_model.text_model.embeddings.token_embedding(next_token)
            current_output = torch.cat([current_output, next_token_emb], dim=1)
        
        # Decode the generated tokens
        caption = clip_processor.decode(generated_tokens, skip_special_tokens=True)
        return caption

    # Format text: capitalize first word of sentences and fix spaces before periods
def format_caption(caption):
        formatted = '. '.join([s.strip().capitalize() for s in caption.strip().split('.') if s.strip()])
        formatted = formatted.replace(' .', '.')  # Remove space before period
        if caption.endswith('.'):  # Maintain trailing period if originally present
            formatted += '.'
        return formatted

def main():
    # Create dataset wrapper
    val_dataset = dataset.FlickrClipDataset(dataset.val_ds)
    
    # Test on a few images from validation set
    test_indices = [876]  # Test first few images
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
            generated = generate_caption(photo, temperature=temp)
            formatted = format_caption(generated)
            print(f"T={temp:.1f} - Formatted caption: {formatted}")
        plt.show()

if __name__ == "__main__":
    # Load image
    device = utils.get_device()
    image = Image.open("/Users/andreas.paxinos/Downloads/img_outdoor-random-story-generator-scaled-1.jpg")
    predictions = ['car','people','grass','cat']
    for k in range(5):
        caption = generate_caption(image)
        print(f"Caption {k}: {caption}")
        predictions.append(caption)

    _, clip_model, clip_processor = models()
    processor_output = clip_processor(
        images=image, 
        text=predictions, 
        return_tensors="pt", 
        padding=True, 
        do_rescale=True
    ).to(device)
    model_output = clip_model(**processor_output)
    # print(model_output.shape)
    print(model_output.logits_per_image)
    print("Argmax: ", torch.argmax(model_output.logits_per_image))
    # Get index of maximum logit value
    best_idx = torch.argmax(model_output.logits_per_image).item()
    # Find corresponding prediction
    best_prediction = predictions[best_idx]
    print(f"Best prediction: {best_prediction} (index {best_idx})")
    
    # main()