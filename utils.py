import transformers
import torch


def get_vocab():
    tokenizer = transformers.CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    vocab = tokenizer.get_vocab()
    vocab = {v: k for k, v in vocab.items()}
    return vocab


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
if __name__ == "__main__":
    vocab = get_vocab()
    print(vocab[49407])
    device = get_device()
    print(device)