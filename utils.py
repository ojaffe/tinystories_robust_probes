from transformers import AutoTokenizer


# https://colab.research.google.com/drive/1jCkpikz0J2o20FBQmYmAGdiKmJGOMo-o?usp=sharing
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable: {100 * trainable_params / all_param:.2f}%"
    )

def create_llama_pause_tokenizer():
    enc = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-955k-token-2T")
    enc.add_tokens(["<PAUSE>"])
    pause_token_id = enc.encode("<PAUSE>")[-1]  # Don't get BOS token

    return enc, pause_token_id
