from absl import app
from absl import flags
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import numpy as np
import torch
import transformers
from torch import nn, optim
from torch.utils import data
from collections import defaultdict
import os
from model import BERTSentimentAnalyser
from dataset import createDataLoader
from utils import read_imdb
from sklearn.metrics import f1_score
import logging
import time


logger = logging.getLogger("Assignment-1")
logger.handlers = []

FLAGS = flags.FLAGS

flags.DEFINE_string('pre_trained_model_name', 'bert-base-uncased', 'Name of the pre-trained model to use')
flags.DEFINE_string('data_path', './imdb dataset', 'Path to dataset')
flags.DEFINE_float('lr', 2e-5, 'Learning Rate')
flags.DEFINE_integer('epochs', 5, 'Number of training epochs')
flags.DEFINE_integer('batch_size', 8, 'Train/eval batch size')
flags.DEFINE_integer('max_len', 512, 'Maximum input sequence length')
flags.DEFINE_string('output_path', './model_logs', 'Output path for model and logs')
flags.DEFINE_bool('do_train', True, 'Set flag for training')
flags.DEFINE_bool('do_eval', True, 'Set flag for evaluation')
flags.DEFINE_bool('do_test', False, 'Set flag for testing')
flags.DEFINE_string('load_model_path', '', 'Path to load saved model for testing')



def train_loop(
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

  f1 = 0
  batch_num = 0
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
      
      f1 += f1_score(labels.cpu(), preds.cpu())
      batch_num += 1 


      

      tepoch.set_postfix(loss=loss.item(), accuracy=100. * (batch_correct.item()/batch_size))
  
  mean_loss = np.mean(losses)
  accuracy = correct_predictions.double()/n_examples
  return f1/batch_num, accuracy, mean_loss


def eval(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  f1 = 0
  batch_num = 0

  with torch.no_grad():
    with tqdm(data_loader, unit="batch") as tepoch:
      for batch in tepoch:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids = input_ids, attention_mask = attention_mask)

        _, preds = torch.max(outputs, dim = 1)

        loss = loss_fn(outputs, labels)
        
        f1 += f1_score(labels.cpu(), preds.cpu())
        batch_num += 1 

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
  return f1/batch_num, correct_predictions.double()/n_examples, np.mean(losses)
  



def main(argv):

  if not os.path.exists(FLAGS.output_path):
    os.makedirs(FLAGS.output_path)

  timestr = time.strftime("%Y%m%d-%H%M%S")
  fh = logging.FileHandler(os.path.join(FLAGS.output_path, timestr+".log"))
  fh.setLevel(logging.INFO)
  logger.addHandler(fh)


  

  train_path = os.path.join(FLAGS.data_path, "Train.csv")
  test_path = os.path.join(FLAGS.data_path, "Test.csv")
  val_path = os.path.join(FLAGS.data_path, "Valid.csv")

  train_texts, train_labels = read_imdb(train_path)
  test_texts, test_labels = read_imdb(test_path)
  val_texts, val_labels = read_imdb(val_path)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


  tokenizer = transformers.BertTokenizer.from_pretrained(FLAGS.pre_trained_model_name)
  loss_fn = nn.CrossEntropyLoss().to(device)

  if(FLAGS.do_train):
    train_DataLoader = createDataLoader(train_texts, train_labels, tokenizer, FLAGS.max_len, FLAGS.batch_size)
  if(FLAGS.do_test):
    test_DataLoader = createDataLoader(test_texts, test_labels, tokenizer, FLAGS.max_len, FLAGS.batch_size)
  if(FLAGS.do_eval):
    val_DataLoader = createDataLoader(val_texts, val_labels, tokenizer, FLAGS.max_len, FLAGS.batch_size)

  if(FLAGS.do_train):
    model = BERTSentimentAnalyser()

    optimizer = transformers.AdamW(model.parameters(), lr = FLAGS.lr, correct_bias = False)

    total_steps = len(train_DataLoader)*FLAGS.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = total_steps
    )

    

    best_f1 = 0
    model = model.to(device)
    #scheduler = scheduler.to(device)
    #optimizer = optimizer.to(device)
    for epoch in range(FLAGS.epochs):
      train_epoch_f1, train_epoch_acc, train_epoch_loss = train_loop(
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

      print(f'Train loss: {train_epoch_loss} accuracy: {train_epoch_acc} F1: {train_epoch_f1}')
      logger.info(f'Train loss: {train_epoch_loss} accuracy: {train_epoch_acc} F1: {train_epoch_f1}')

      if(FLAGS.do_eval):
        val_epoch_f1, val_epoch_acc, val_epoch_loss = eval(
            model, 
            val_DataLoader,
            loss_fn,
            device,
            len(val_texts)
        )
        print(f'Validation loss: {val_epoch_loss} accuracy: {val_epoch_acc} F1: {val_epoch_f1}')
        logger.info(f'Validation loss: {val_epoch_loss} accuracy: {val_epoch_acc} F1: {val_epoch_f1}')

        if(val_epoch_f1>best_f1):
          torch.save(model.state_dict(), os.path.join(FLAGS.output_path, 'best_model_state.bin'))
          best_f1 = val_epoch_f1
      
      else:
        if(train_epoch_f1>best_f1):
          torch.save(model.state_dict(), os.path.join(FLAGS.output_path, 'best_model_state.bin'))
          best_f1 = train_epoch_f1

        
      

  if(FLAGS.do_test):
    
    model = BERTSentimentAnalyser()
    model = model.to(device)

    if not FLAGS.load_model_path:
      model.load_state_dict(torch.load(os.path.join(FLAGS.output_path,'best_model_state.bin')))
    else:
      model.load_state_dict(torch.load(os.path.join(FLAGS.load_model_path,'best_model_state.bin')))

    test_f1, test_acc, test_loss = eval(
            model, 
            test_DataLoader,
            loss_fn,
            device,
            len(test_texts)
        )
    print(f'Test loss: {test_loss} accuracy: {test_acc} F1: {test_f1}')
    logger.info(f'Test loss: {test_loss} accuracy: {test_acc} F1: {test_f1}')



if __name__ == '__main__':
  app.run(main)