import torch

class IMDBReviewDataset(torch.utils.data.Dataset):

    def __init__(self, reviews, labels, tokenizer, max_len):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)
        

    def __getitem__(self, idx):
        review = self.reviews[idx]

        encoding = self.tokenizer.encode_plus(
            review,
            max_length = self.max_len,
            add_special_tokens = True,
            padding='max_length',
            return_attention_mask = True,
            return_token_type_ids = False,
            return_tensors = 'pt'
        )

        return {
            'review_text': review,
            'input_ids' : encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype = torch.long)
        }

def createDataLoader(reviews, labels, tokenizer, max_len, batch_size):
    dataset = IMDBReviewDataset(
        reviews = reviews,
        labels = labels,
        tokenizer = tokenizer,
        max_len = max_len
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers = 2)
  