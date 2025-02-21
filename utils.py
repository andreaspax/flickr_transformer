import transformers
import torch
import decoder

def get_vocab():
    tokenizer = transformers.CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    vocab = tokenizer.get_vocab()
    vocab = {v: k for k, v in vocab.items()}
    return vocab

def load_model():
    device = get_device()
    model = decoder.Decoder(di_initial=512, d_model=512, dff=2048, vocab_size=49408)
    model.load_state_dict(torch.load("weights/flicker-captioning-best.pt", map_location=device))
    model.to(device)
    model.eval()
    return model

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