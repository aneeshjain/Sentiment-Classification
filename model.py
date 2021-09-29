from torch import nn
from transformers import BertModel
from absl import flags

FLAGS = flags.FLAGS


class BERTSentimentAnalyser(nn.Module):

  def __init__(self):
    super(BERTSentimentAnalyser, self).__init__()
    self.bert = BertModel.from_pretrained(FLAGS.pre_trained_model_name)
    self.dropout = nn.Dropout(p = 0.2)
    self.out = nn.Linear(self.bert.config.hidden_size, 2)
    self.softmax = nn.Softmax(dim = 1)

  def forward(self, input_ids, attention_mask):
    bert_output = self.bert(
        input_ids = input_ids,
        attention_mask = attention_mask
        )
    output = self.dropout(bert_output['pooler_output'])
    output = self.out(output)
    return self.softmax(output)