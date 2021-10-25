## IMDB Review Semantic Analysis with BERT

#### This folder contains the following files and directories and files:

* imdb dataset - contains the Train/Test/Valid data
* main.py - contains the main code to run train, test and evaluation jobs
* dataset.py - contains code for the dataset class and a function that creates the dataloader for PyTorch
* model.py - contains the model definiton
* utils.py - contains some utility functions, eg. read_imdb that reads data from the csv files

### Packages:
* torch
* transformers
* sklearn (for f1 score)
* absl (for flags)
* tqdm (for progress bar tracking)

### Usage:

If you type the following command in your terminal you will see all the arguments that can be passed to run various kinds of jobs:

```
python3 main.py --help


flags:

main.py:
  --batch_size: Train/eval batch size
    (default: '8')
    (an integer)
  --data_path: Path to dataset
    (default: './imdb dataset')
  --[no]do_eval: Set flag for evaluation
    (default: 'true')
  --[no]do_test: Set flag for testing
    (default: 'false')
  --[no]do_train: Set flag for training
    (default: 'true')
  --epochs: Number of training epochs
    (default: '3')
    (an integer)
  --load_model_path: Path to load saved model
    (default: '')
  --lr: Learning Rate
    (default: '2e-05')
    (a number)
  --max_len: Maximum input sequence length
    (default: '512')
    (an integer)
  --output_path: Output path for model and logs
    (default: './model_logs')
  --pre_trained_model_name: Name of the pre-trained model to use
    (default: 'bert-base-uncased')
```
The most important among these are:
* --data_path: Although the default is set to the correct directory, the full path may need to be passed as an argument depending on your system

* --do_train/--do_eval/--do_test: These are boolean flags. The default for --do_train and --do_eval is True. Depending the flags chosen, the best model based on training F1 or validation F1 will be saved. --do_test enables testing on the test set. Add prefix 'no' to the flag to set it as False.

* --output_path: The path given to this flag is created and the best model and training logs are saved here. A default value "./model_logs" is given, and if unchanged the final artifacts will be found here. The log files will report epoch accuracy, loss and F1 scores for training, testing and validation.

* --load_model_path: This is the path from where the saved model will be loaded at test time. If during training you change the output_path, be sure to pass the same path here. If no change is made to --output_path, the saved model will be automatically loaded, no need to pass the path here again.

* The rest of the flags are common model hyperparameters. The default values are set to give the best performing model


### Example Commands:

#### Training with validation and testing

```
python3 main.py --do_train --do_eval --do_test --data_path /home/aneesh/SentimentClassification/HW1/imdb\ dataset --epochs 1
```

#### Only testing
```
python3 main.py --nodo_train --nodo_eval --do_test --data_path /home/aneesh/SentimentClassification/HW1/imdb\ dataset --load_model_path ./model_logs
```

