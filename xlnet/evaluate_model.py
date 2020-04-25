import pathlib
import csv
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from absl import flags

flags.DEFINE_string("predict_file_path", default=None,
      help="Path of predictions tsv file.")
flags.DEFINE_string("test_file_path", default=None,
      help="Path of test data file.")

labels_dict = {
    "neg": 0,
    "pos": 1,
    "neu": 2
}

def get_labels(test_file: pathlib.Path):
    labels = []
    with test_file.open(mode= 'r') as f:
        lines = f.readlines()
        for line in lines:
            label, _ = line.split(',', 1)
            labels.append(labels_dict[label])

    return labels
            
def get_predictions(predict_file: pathlib.Path):
    predictions = []
    with predict_file.open(mode= 'r') as f:
        reader = csv.DictReader(f, delimiter= '\t')
        for row in reader:
            predictions.append(labels_dict[row['prediction']])

    return predictions

if __name__ == "__main__":
    if not flags.predict_path:
        raise ValueError("Predictions file path MUST be specified")
    predict_file = pathlib.Path(flags.predict_file_path)
    if not flags.test_path:
        raise ValueError("Test data file path MUST be specified")
    test_file= pathlib.Path(flags.test_file_path)
    labels = get_labels(test_file)
    predictions = get_predictions(predict_file)
    print(classification_report(labels, predictions))
    print(accuracy_score(labels, predictions))
    print(confusion_matrix(labels, predictions))