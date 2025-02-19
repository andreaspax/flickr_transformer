import os
import torch
import datasets
import transformers 
import utils

script_dir = os.path.dirname(os.path.abspath(__file__))
device = utils.get_device()

ds = datasets.load_dataset("nlphuji/flickr30k", split="test", trust_remote_code=True)

pretrained_model = "openai/clip-vit-base-patch32"

clip_model = transformers.CLIPModel.from_pretrained(pretrained_model).to(device)
clip_processor = transformers.CLIPProcessor.from_pretrained(pretrained_model)


for param in clip_model.parameters():
    param.requires_grad = False

clip_model.eval()


length = len(ds)
train_length = int(length * 0.99) 
val_length = int(length * 0.999)

train_ds = ds.select(range(train_length))
val_ds = ds.select(range(train_length, val_length))
test_ds = ds.select(range(val_length, length))


class FlickrClipDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset) * 5

    def __getitem__(self, idx):
        photo_id = idx // 5
        caption_id = idx % 5

        photo = self.dataset[photo_id]["image"]
        caption = self.dataset[photo_id]["caption"][caption_id]

        return photo, caption

def get_token_embedding(token_id):
        return clip_model.get_text_features(token_id)

def _collate_fn(batch):
    photos = [item[0] for item in batch]
    captions = [item[1] for item in batch]

    with torch.no_grad():

        processor_outputs = clip_processor(
            images=photos,
            text=captions, 
            return_tensors="pt", 
            padding='max_length',
            truncation=True,
        ).to(device)

        input_tokens = processor_outputs.input_ids
        input_attention_mask = processor_outputs.attention_mask

        output_tokens = input_tokens[:, 1:]
        input_tokens = input_tokens[:, 1:-1]
        input_attention_mask = input_attention_mask[:, 1:-1]

        clip_image_outputs = clip_model.get_image_features(processor_outputs.pixel_values).unsqueeze(1)
        clip_text_outputs = clip_model.text_model(input_tokens, attention_mask=input_attention_mask)
        combined_outputs = torch.cat((clip_image_outputs, clip_text_outputs.last_hidden_state), dim=1)
        attention_mask = torch.cat((input_attention_mask, torch.zeros(input_attention_mask.size(0), 1).to(device)), dim=1)


        return (
            combined_outputs,
            attention_mask,
            output_tokens,
        )
    


if __name__ == "__main__":

    val_ds = FlickrClipDataset(val_ds)
    print(f"Dataset size: {len(val_ds)}")

    # Test DataLoader with custom collate function
    dataloader = torch.utils.data.DataLoader(
        val_ds, 
        batch_size=2, 
        shuffle=False,
        collate_fn=_collate_fn  # Re-enable the custom collate_fn
    )

    # Test
    batch = next(iter(dataloader))
    photo, caption = val_ds.__getitem__(1)

    import matplotlib.pyplot as plt

    print(caption)
    print(batch[0].shape)
    print(batch[1].shape)
    print(batch[1])
    print(batch[2].shape)
    print(batch[2])
    plt.imshow(photo)
    plt.show()
