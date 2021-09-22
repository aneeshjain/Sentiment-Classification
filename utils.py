import csv


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