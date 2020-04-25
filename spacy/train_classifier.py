# sentiment analysis for Connext Blog Posts
from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import thinc.extra.datasets
import spacy
from spacy.util import minibatch, compounding
from typing import NamedTuple
import pathlib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import unicodedata
from bs4 import BeautifulSoup
from config.constants import SpacyVars as sv

CONNEXT_TRAIN_DATA_FILE_PATH = sv.CONNEXT_TRAIN_DATA_FILE_PATH
CONNEXT_TEXT_LIMIT = sv.CONNEXT_TEXT_LIMIT
CONNEXT_SPLIT = sv.CONNEXT_SPLIT
STANFORD_LABELS_FILE_PATH = sv.STANFORD_LABELS_FILE_PATH
STANFORD_PHRASES_FILE_PATH = sv.STANFORD_PHRASES_FILE_PATH
STANFORD_TEXT_LIMIT = sv.STANFORD_TEXT_LIMIT
STANFORD_SPLIT = sv.STANFORD_SPLIT
IMDB_TEXT_LIMIT = sv.IMDB_TEXT_LIMIT
IMDB_SPLIT = sv.IMDB_SPLIT
REPORT_FILE_PATH = sv.REPORT_FILE_PATH

class StanfordConfig(NamedTuple):
    labels_file: str = STANFORD_LABELS_FILE_PATH
    phrases_file: str = STANFORD_PHRASES_FILE_PATH
    limit: int = STANFORD_TEXT_LIMIT
    split: float = STANFORD_SPLIT

class IMDBConfig(NamedTuple):
    limit: int = IMDB_TEXT_LIMIT
    split: float = IMDB_SPLIT

class ConnextConfig(NamedTuple):
    train_file: str = CONNEXT_TRAIN_DATA_FILE_PATH
    limit: int = CONNEXT_TEXT_LIMIT
    split: float = CONNEXT_SPLIT

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_texts=("Number of texts to train from", "option", "t", int),
    n_iter=("Number of training iterations", "option", "n", int),
    init_tok2vec=("Pretrained tok2vec weights", "option", "t2v", Path)
)


def main(model=None, output_dir=None, n_iter=20, n_texts=2000, init_tok2vec=None, data_loader= None):
    if data_loader is None:
        raise ValueError("Data Loader is required")

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
    # enable working with GPU
    spacy.require_gpu()
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat",
            config={
                "exclusive_classes": True,
                "architecture": "simple_cnn",
            }
        )
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe("textcat")

    # add label to text classifier
    textcat.add_label("POSITIVE")
    textcat.add_label("NEGATIVE")
    textcat.add_label("NEUTRAL")

    # load the IMDB dataset
    print("Loading data...")
    (train_texts, train_cats), (dev_texts, dev_cats) = data_loader()
    train_texts = train_texts[:n_texts] if n_texts is not None else train_texts
    train_cats = train_cats[:n_texts] if n_texts is not None else train_cats
    print(
        "Using {} examples ({} training, {} evaluation)".format(
            n_texts, len(train_texts), len(dev_texts)
        )
    )
    print("text {}".format(dev_texts[0]))
    print("Cat {}".format(dev_cats[0]))
    train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        if init_tok2vec is not None:
            with init_tok2vec.open("rb") as file_:
                textcat.model.tok2vec.from_bytes(file_.read())
        print("Training the model...")
        print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
        batch_sizes = compounding(4.0, 32.0, 1.001)
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            random.shuffle(train_data)
            batches = minibatch(train_data, size=batch_sizes)
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print(
                "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
                    losses["textcat"],
                    scores["textcat_p"],
                    scores["textcat_r"],
                    scores["textcat_f"],
                )
            )

    # test the trained model
    test_text = "This movie sucked"
    test_text2 = "In the summer time we can have some good time. In the summer time we can do SKR"
    test_text3 = "I know that girl. She is very superficial. She is all about looks and money. She wants to do SKR"
    test_text4 = "A brand new gucci pouch. If you don't invest then you're losing out. All of that assets."
    test_text5 = "There were good moments in my high school: Koforidua Secondary Technical School."
    test_text6 = "Ex President Obama has swag."
    test_text7 = "Robert Freeman is a pathological liar."
    test_text8 = "Dear all, find attached the document"
    test_text9 = "What happens when technology meets science?"
    doc = nlp(test_text)
    print(test_text, doc.cats)

    if output_dir is not None:
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        print(test_text, doc2.cats)
        doc2 = nlp2(test_text3)
        print(test_text3, doc2.cats)
        doc2 = nlp2(test_text4)
        print(test_text4, doc2.cats)
        doc2 = nlp2(test_text5)
        print(test_text5, doc2.cats)
        doc2 = nlp2(test_text6)
        print(test_text6, doc2.cats)
        doc2 = nlp2(test_text)
        print(test_text, doc2.cats)
        doc2 = nlp2(test_text7)
        print(test_text7, doc2.cats)
        doc2 = nlp2(test_text8)
        print(test_text8, doc2.cats)
        doc2 = nlp2(test_text9)
        print(test_text9, doc2.cats)

    do_report(dev_texts, dev_cats, output_dir)

def do_report(texts, cats, model_dir) -> None:
    labels_dict = {
        "NEGATIVE": 0,
        "POSITIVE": 1,
        "NEUTRAL": 2
    }
    def get_labels(cats):
        labels = []
        for cat in cats:
            label = None
            for label, bool_value in cat.items():
                if bool_value:
                    label = labels_dict[label]
                    labels.append(label)
                    break

        return labels
    def get_predictions(texts, model_dir):
        nlp = spacy.load(model_dir)
        predictions = []
        for text in texts:
            doc = nlp(text)
            largest = 0.0
            predicted_label = 'POSITIVE'
            for label, value_float in doc.cats.items():
                if value_float > largest:
                    largest = value_float
                    predicted_label = label
            predictions.append(labels_dict[predicted_label])

        return predictions

    # todo drop None values if any
    labels = get_labels(cats)
    predictions = get_predictions(texts, model_dir)
    print("lengths equal? {} {}".format(len(labels), len(predictions)))
    print("Report : ")
    print(classification_report(labels, predictions))
    print(accuracy_score(labels, predictions))
    print(confusion_matrix(labels, predictions))
    text_to_write = ""
    text_to_write += classification_report(labels, predictions)
    text_to_write += "\n\n"
    text_to_write += str(accuracy_score(labels, predictions))
    text_to_write += "\n"
    text_to_write += confusion_matrix(labels, predictions)
    report_file = pathlib.Path(REPORT_FILE_PATH)
    report_file.write_text(text_to_write)

    return
        

def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if label == "NEGATIVE":
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}

def load_imdb_data(config: IMDBConfig = IMDBConfig()):
    """Load data from the IMDB dataset."""
    # Partition off part of the train data for evaluation
    train_data, _ = thinc.extra.datasets.imdb()
    random.shuffle(train_data)
    train_data = train_data[-config.limit:]
    texts, labels = zip(*train_data)
    cats = [ {"POSITIVE": bool(y), "NEGATIVE": not bool(y), "NEUTRAL": False} for y in labels ]

    return texts, cats

def escape_newline(s):
    return s.replace('\n', '')

def get_phrases_ids_and_scores(file_path, size= 20000):
    with open(file_path) as f:
        lines = f.readlines()
        lines = lines[1:]
        random.shuffle(lines)
        lines = lines[:size] if size != 0 else lines
        ids_scores = [ line.split('|') for line in lines ]
        d = { arr[0]: float(escape_newline(arr[1])) for arr in ids_scores }
        return d

def get_phrases_and_scores(ids_scores, file_path):
    print("inside get phrases and scores")
    with open(file_path) as f:
        lines = f.readlines()
        ids_phrases = { escape_newline(line.split("|")[1]): line.split("|")[0] for line in lines }
        d = {}
        print("")
        for id, score in ids_scores.items():
            if id in ids_phrases:
                phrase = ids_phrases[id]
                d[phrase] = score
    
        return d

def get_label(score):
    label = { "POSITIVE": False, "NEGATIVE": False, "NEUTRAL": False }
    if score < 0.4:
        label["NEGATIVE"] = True
    elif score > 0.6:
        label["POSITIVE"] = True
    else:
        label["NEUTRAL"] = True

    return label

def load_stanford_data(config: StanfordConfig = StanfordConfig()):
    ids_scores = get_phrases_ids_and_scores(config.labels_file, config.limit)
    phrases_scores = get_phrases_and_scores(ids_scores, config.phrases_file)
    train_data = list(phrases_scores.keys())
    cats = [ get_label(score) for score in phrases_scores.values() ]

    return (train_data, cats)

def load_connext_data(config: ConnextConfig = ConnextConfig()):
    def get_label_connext(cat):
        label = { "POSITIVE": False, "NEGATIVE": False, "NEUTRAL": False }
        if cat == "neu":
            label["NEUTRAL"] = True
        elif cat == "pos":
            label["POSITIVE"] = True
        else:
            label["NEGATIVE"] = True
        return label
    path = pathlib.Path(config.train_file)
    texts = []
    cats = []
    with path.open(mode= 'r') as f:
        for line in f:
            parts = line.split(",")
            cats.append(get_label_connext(parts[0]))
            texts.append(parts[1])
    
    return texts, cats

def load_stanford_imdb_connext(
    imdb_config: IMDBConfig = IMDBConfig(),
    stanford_config: StanfordConfig = StanfordConfig(),
    connext_config: ConnextConfig = ConnextConfig()
    ):
    train_data = []
    train_dev = []
    cats = []
    cats_dev = []

    #stanford
    stanford_train_data, stanford_cats = load_stanford_data()
    split = int(len(stanford_train_data) * stanford_config.split)
    train_data.extend(stanford_train_data[:split])
    train_dev.extend(stanford_train_data[split:])
    cats.extend(stanford_cats[:split])
    cats_dev.extend(stanford_cats[split:])

    #IMDB
    texts, categories = load_imdb_data()
    split = int(len(texts) * imdb_config.split)
    train_data.extend(texts[:split])
    train_dev.extend(texts[split:])
    cats.extend(categories[:split])
    cats_dev.extend(categories[split:])

    # Connext
    texts, categories = load_connext_data()
    # no splitting, only relatively few connext posts
    train_data.extend(texts)
    cats.extend(categories)

    return (train_data, cats), (train_dev, cats_dev)

def get_label_xlnet(label_obj):
    label = ""
    for key, label_bool in label_obj.items():
        if label_bool:
            if key == "NEGATIVE":
                label = "neg"
            elif key == "POSITIVE":
                label = "pos"
            else:
                label = "neu"
            break
    return label

def remove_control_characters(s):
        return "".join(ch if unicodedata.category(ch)[0]!="C" else ' ' for ch in s )

def clean_text(s):
    raw = BeautifulSoup(s, 'html.parser').get_text()
    raw = remove_control_characters(raw)
    lines = [ line.strip() for line in raw.splitlines() ]
    text = "".join(lines)

    return text

def prepare_xlnet_data():
    (train_texts, train_cats), (dev_texts, dev_cats) = load_stanford_imdb_connext()
    train_file = pathlib.Path('train_all.txt')
    with train_file.open(mode= 'w') as f:
        for i, text in enumerate(train_texts):
            line = get_label_xlnet(train_cats[i]) + ',' + clean_text(text) + '\n'
            f.write(line)
    test_file = pathlib.Path('test_all.txt')
    with test_file.open(mode= 'w') as f:
        for i, text in enumerate(dev_texts):
            line = get_label_xlnet(dev_cats[i]) + ',' + clean_text(text) + '\n'
            f.write(line)