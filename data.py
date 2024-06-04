from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler
import pandas as pd
import glob
import json
from tqdm import tqdm


class TinyStoriesDataset(Dataset):
    def __init__(self, all_examples):
        self.data = pd.DataFrame.from_dict({"text": all_examples})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data.iloc[idx]
        text = x["text"]  # "content" key for refined web
        return text


class PadCollate():
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        text = batch
        out = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_seq_len + 1, return_tensors="pt")  # +1 since we remove 1 for X, Y
        input_ids = out["input_ids"]
        attention_mask = out["attention_mask"] # TODO add EOS token

        return input_ids, attention_mask


def create_dataloaders(
    tokenizer,
    batch_size: int,
    max_seq_len: int,
):
    files = glob.glob("data/TinyStories_all_data/*.json")[:10]
    all_examples = []
    for i in tqdm(files):
        with open(i, "r") as f:
            json_data = json.load(f)
            json_examples = [i["summary"] for i in json_data]
            all_examples += json_examples

    data = TinyStoriesDataset(all_examples)

    train_prop = 0.9

    indices = range(len(data))
    train_indices = indices[:round(train_prop*len(data))]
    val_indices = indices[round(train_prop*len(data)):]

    train_loader = DataLoader(data, batch_size=batch_size, 
                              sampler=SubsetRandomSampler(train_indices), collate_fn=PadCollate(tokenizer, max_seq_len))
    val_loader = DataLoader(data, batch_size=batch_size, 
                            sampler=SequentialSampler(val_indices), collate_fn=PadCollate(tokenizer, max_seq_len))

    return train_loader, val_loader
