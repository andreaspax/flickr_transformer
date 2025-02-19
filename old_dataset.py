import torch
import matplotlib.pyplot as plt
import datasets
import transformers
import PIL
import io



class Flickr30kDataset(torch.utils.data.Dataset):
  def __init__(self, seed=2, split='train'):
        print("Init Flickr30kDataset")
        torch.manual_seed(seed)
        
        # Initialize CLIP processor and model once
        self.processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        print("Loading dataset...")
        ds = datasets.load_dataset(
            "nlphuji/flickr30k", 
            split=f"test[:10000]",
            trust_remote_code=True
        ).filter(
            lambda x: x['split'] == split
        ).select_columns(['image', 'caption'])

        # Unpack all images and captions first
        print("Unpacking data...")
        images = []
        captions = []
        for item in ds:
            for caption in item['caption']:
                images.append(item['image'])
                captions.append(caption)

        # Process all data at once
        print("Processing images and captions in batch...")
        with torch.no_grad():
            # Process all images in one go
            img_inputs = self.processor(images=images, return_tensors="pt", padding=True)
            img_embs = self.model.get_image_features(**img_inputs)
            
            # Process all captions in one go
            cap_inputs = self.processor(
                text=captions, 
                return_tensors="pt", 
                padding=True
            )

        # Store processed data
        self.processed_data = []
        for img, img_emb, caption, tokens in zip(images, img_embs, captions, cap_inputs.input_ids):
            self.processed_data.append({
                'img': img,
                'img_emb': img_emb,
                'caption': caption,
                'tokenised_caption_in': tokens[1:-1],  # Remove first and last tokens
                'tokenised_caption_out': tokens[1:]    # Remove first token
            })

        # Clean up
        del self.model
        del self.processor
        torch.cuda.empty_cache()

  def __len__(self):
    return len(self.processed_data)

  def __getitem__(self, idx):
    """Return a single item from the dataset"""
    return self.processed_data[idx]

def collate_fn(batch):
    return {
        'img_emb': torch.stack([item['img_emb'] for item in batch]),
        'tokenised_caption_in': torch.stack([item['tokenised_caption_in'] for item in batch]),
        'tokenised_caption_out': torch.stack([item['tokenised_caption_out'] for item in batch]),
        'caption': [item['caption'] for item in batch]
    }





if __name__ == '__main__':
    # Test the dataset
    test_ds = Flickr30kDataset()
    print(f"Dataset size: {len(test_ds)}")
    
    # Test DataLoader with custom collate function
    dataloader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=2, 
        shuffle=True,
        collate_fn=collate_fn  # Re-enable the custom collate_fn
    )
    
    # Get first batch
    batch = next(iter(dataloader))
    print("\nFirst batch:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: tensor shape {v.shape}")
        else:
            print(f"{k}: {type(v)}, len: {len(v)}")
    
    print(batch['caption'][1])
    print(batch['tokenised_caption_in'][1])
    print(batch['tokenised_caption_out'][1])
    print(batch['img_emb'][1])

    # Find matching caption
    target_caption = batch['caption'][1]
    matching_items = [item for item in test_ds.processed_data if item['caption'] == target_caption]
    
    if matching_items:
        img_value = matching_items[0]['img']
        img_emb_value = matching_items[0]['img_emb']
        print("Image embedding for caption:", img_emb_value)
        
        # Plot the image
        plt.imshow(img_value)
        plt.title(f"Image for caption: {target_caption}")
        plt.axis('off')
        plt.show()
    else:
        print(f"No image found with caption: {target_caption}")