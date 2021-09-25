from absl import app
from absl import flags
from model import SentimentClassifier
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import numpy as np
import torch
import transformers
from torch import nn, optim
from torch.utils import data
from collections import defaultdict
import os
from model import SentimentClassifier
from dataset import createDataLoader
from utils import read_imdb


FLAGS = flags.FLAGS

flags.DEFINE_string('pre_trained_model_name', 'bert-base-uncased', 'Name of the pre-trained model to use')
flags.DEFINE_string('data_path', '/Users/aneesh/Desktop/Grad School/VIRGINIA TECH DOCS/Courses/NLP/HW1/imdb dataset', 'Path to dataset')
flags.DEFINE_float('lr', 2e-5, 'Learning Rate')
flags.DEFINE_integer('epochs', 3, 'Number of training epochs')
flags.DEFINE_integer('batch_size', 8, 'Train/eval batch size')
flags.DEFINE_integer('max_len', 512, 'Maximum input sequence length')
flags.DEFINE_string('outptu_path', '', 'Output path for model and logs')



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
  correct_predictions = 0

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


  train_path = os.path.join(FLAGS.data_path, "Train.csv")
  test_path = os.path.join(FLAGS.data_path, "Test.csv")
  val_path = os.path.join(FLAGS.data_path, "Valid.csv")

  train_texts, train_labels = read_imdb(train_path)
  test_texts, test_labels = read_imdb(test_path)
  val_texts, val_labels = read_imdb(val_path)


  tokenizer = transformers.BertTokenizer.from_pretrained(FLAGS.pre_trained_model_name)

  train_DataLoader = createDataLoader(train_texts, train_labels, tokenizer, FLAGS.max_len, FLAGS.batch_size)
  test_DataLoader = createDataLoader(test_texts, test_labels, tokenizer, FLAGS.max_len, FLAGS.batch_size)
  val_DataLoader = createDataLoader(val_texts, val_labels, tokenizer, FLAGS.max_len, FLAGS.batch_size)

  model = SentimentClassifier(n_classes = 2)

  optimizer = transformers.AdamW(model.parameters(), lr = FLAGS.lr, correct_bias = False)

  total_steps = len(train_DataLoader)*FLAGS.epochs
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps = 0,
      num_training_steps = total_steps
  )

  loss_fn = nn.CrossEntropyLoss().to(device)

  history = defaultdict(list)
  best_acc = 0
  model = model.to(device)
  #scheduler = scheduler.to(device)
  #optimizer = optimizer.to(device)
  for epoch in range(FLAGS.epochs):
    train_epoch_acc, train_epoch_loss = train_epoch(
        model, 
        train_DataLoader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(train_texts),
        epoch+1,
        FLAGS.batch_size
    )

    print(f'Train loss: {train_epoch_loss} accuracy: {train_epoch_acc}')

    val_epoch_acc, val_epoch_loss = eval(
        model, 
        val_DataLoader,
        loss_fn,
        device,
        len(val_texts)
    )
    print(f'Validation loss: {val_epoch_loss} accuracy: {val_epoch_acc}')

    history['train_acc'].append(train_epoch_acc)
    history['train_loss'].append(train_epoch_loss)
    history['val_acc'].append(val_epoch_acc)
    history['val_loss'].append(val_epoch_loss)

    if(val_epoch_acc>best_acc):
      torch.save(model, 'model.pth')
      best_acc = val_epoch_acc


if __name__ == '__main__':
  app.run(main)