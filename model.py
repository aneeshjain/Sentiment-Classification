from torch import nn
from transformers import BertModel

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.dropout = nn.Dropout(p = 0.2)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.softmax = nn.Softmax(dim = 1)

  def forward(self, input_ids, attention_mask):
    bert_output = self.bert(
        input_ids = input_ids,
        attention_mask = attention_mask
        )
    output = self.dropout(bert_output['pooler_output'])
    output = self.out(output)
    return self.softmax(output)