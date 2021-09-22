import csv
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

def read_imdb(data_path):
  texts = []
  labels = []
  with open(data_path, 'r') as read_obj:           
    csv_reader = csv.reader(read_obj)
    header = next(csv_reader)
    # Check file as empty
    if header != None:
        # Iterate over each row after the header in the csv
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            texts.append(row[0])
            labels.append(int(row[1]))
    return texts, labels


def check_token_dist(review_list):
    token_len = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    for sent in review_list:
        tokens = tokenizer.encode(sent, max_length = 512)
        token_len.append(len(tokens))
    
    ax = sns.distplot(token_len)
    plt.xlabel("Token Count")
    plt.show()