from absl import app
from absl import flags
from model import SentimentClassifier
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import numpy as np
import torch
import transformers
from transformers import BertModel
from torch import nn, optim
from torch.utils import data
from collections import defaultdict
import os


FLAGS = flags.FLAGS

flags.DEFINE_string('pre_trained_model_name', 'bert-base-uncased', 'Name of the pre-trained model to use')


def train_epoch(
    model, 
    data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    n_examples,
    epoch_num, batch_size):
  
  model = model.train()
  losses = []
  correct_predictions = 0
  with tqdm(data_loader, unit="batch") as tepoch:
    for batch in tepoch:
      
      tepoch.set_description(f"Epoch {epoch_num}")

      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['label'].to(device)

      # print(input_ids.shape)
      # print(attention_mask.shape)
      # print(labels.shape)
      outputs = model(input_ids = input_ids, attention_mask = attention_mask)

      _, preds = torch.max(outputs, dim = 1)

      loss = loss_fn(outputs, labels)

      batch_correct = torch.sum(preds == labels)
      correct_predictions += batch_correct 
      losses.append(loss.item())

      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
      optimizer.step()
      scheduler.step()
      optimizer.zero_grad()

      

      tepoch.set_postfix(loss=loss.item(), accuracy=100. * (batch_correct.item()/batch_size))
  
  mean_loss = np.mean(losses)
  accuracy = correct_predictions.double()/n_examples
  return accuracy, mean_loss


def eval(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = []

  with torch.no_grad():
    with tqdm(data_loader, unit="batch") as tepoch:
      for batch in tepoch:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids = input_ids, attention_mask = attention_mask)

        _, preds = torch.max(outputs, dim = 1)

        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
  return correct_predictions.double()/n_examples, np.mean(losses)
  



def main(argv):

    model = SentimentClassifier(2)

if __name__ == '__main__':
  app.run(main)