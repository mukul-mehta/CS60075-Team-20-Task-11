import argparse
from random import choice
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel
from dataloader import prepare_data, create_dataloader
from config import read_config
from training import *
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef

parser = argparse.ArgumentParser()
parser.add_argument('--config', default="config.ini",
                    help="INI file for model configuration")
parser.add_argument('--model', help="Which model to run",
                    choices=["baseline", "vanilla-bert", "bert-linear", "bert-bilstm"], default="bert-linear")


PATHS = read_config(filename="config.ini", section="DATASET")
HYPERPARAMS = read_config(filename="config.ini", section="HYPERPARAMS")


def predict_labels_baseline(classifier, test_dataloader):
    model = AutoModel.from_pretrained(
        "allenai/scibert_scivocab_uncased", output_hidden_states=True)

    model.to(device)
    test_embeddings = []
    test_labels = []

    for idx, batch in enumerate(test_dataloader):
        if idx % 20 == 0:
            print(f"Done with {idx} of {len(test_dataloader)} batches")
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_lengths = batch[2].to(device)
        b_labels = batch[3].to(device)

        with torch.no_grad():
            output = model(b_input_ids, attention_mask=b_input_mask,
                           token_type_ids=None, return_dict=False)[2][-1]
            output = output[:, 0, :].view(-1, 768)
            test_embeddings.append(output.cpu().numpy())
            test_labels.append(b_labels.cpu().numpy())

    flat_test_embeddings = np.concatenate(test_embeddings, axis=0)
    flat_test_labels = np.concatenate(test_labels, axis=0)
    y_pred = classifier.predict(flat_test_embeddings)

    mcc = matthews_corrcoef(flat_test_labels, y_pred)
    report = classification_report(flat_test_labels, y_pred)
    return mcc, report


def predict_labels_scibert_linear(model, test_dataloader):
    print(
        f'Predicting labels for {len(test_dataloader) * batch_size} test sentences with model SciBERT-Linear')

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

    for idx, batch in enumerate(test_dataloader):
        if idx % 20 == 0:
            print(f"Done with {idx} of {len(test_dataloader)}")

        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_lengths, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            outputs = model(b_input_ids,
                            b_input_mask,
                            None)
            _, labels = torch.max(outputs, dim=1)

        logits = labels.cpu().numpy()
        label_ids = b_labels.cpu().numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    print('DONE.')
    flat_predictions = np.concatenate(predictions, axis=0)

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
    report = classification_report(flat_true_labels, flat_predictions)

    return mcc, report


def predict_labels_scibert_bilstm(model, test_dataloader):
    print(
        f"Predicting labels for {len(test_dataloader) * batch_size} test sentences for SciBERT-BiLSTM")
    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for idx, batch in enumerate(test_dataloader):
        if idx % 10 == 0:
            print(f"Done with {idx} of {len(test_dataloader)}")

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_lengths, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            outputs = model(b_input_ids,
                            b_input_mask,
                            None,
                            b_lengths)
            _, labels = torch.max(outputs, dim=1)

        logits = labels.cpu().numpy()
        label_ids = b_labels.cpu().numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    print('DONE.')

    flat_predictions = np.concatenate(predictions, axis=0)

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
    report = classification_report(flat_true_labels, flat_predictions)

    return mcc, report


if __name__ == "__main__":
    args = parser.parse_args()
    print(args.model)

    print('Loading SciBERT tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(
        'allenai/scibert_scivocab_uncased')

    batch_size = int(HYPERPARAMS["BATCH_SIZE"])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_sentences, train_labels = prepare_data(
        input_dir=PATHS["TRAIN_DATA_PATH"], oversample=True)

    trial_sentences, trial_labels = prepare_data(
        input_dir=PATHS["VALIDATION_DATA_PATH"], oversample=True)

    train_sentences = train_sentences + trial_sentences
    train_labels = train_labels + trial_labels

    train_dataloader = create_dataloader(
        train_sentences, train_labels, tokenizer)

    test_sentences, test_labels = prepare_data(
        input_dir=PATHS["TEST_DATA_PATH"], oversample=False)

    test_dataloader = create_dataloader(test_sentences, test_labels, tokenizer)

    if args.model == "baseline":
        classifier, time_taken = train_baseline(train_dataloader)
        mcc, report = predict_labels_baseline(classifier, test_dataloader)

        print(f"Time Taken is {time_taken} seconds")
        print(f"MCC Score is {mcc}")
        print(report)

    if args.model == "bert-linear":
        model, time_taken = train_scibert_linear(
            train_dataloader, train_sentences, train_labels, True)
        mcc, report = predict_labels_scibert_linear(model, test_dataloader)

        print(f"Time Taken is {time_taken} seconds")
        print(f"MCC Score is {mcc}")
        print(report)

    if args.model == "bert-bilstm":
        model, time_taken = train_scibert_bilstm(
            train_dataloader, train_sentences, train_labels, True)
        mcc, report = predict_labels_scibert_bilstm(model, test_dataloader)

        print(f"Time Taken is {time_taken} seconds")
        print(f"MCC Score is {mcc}")
        print(report)
