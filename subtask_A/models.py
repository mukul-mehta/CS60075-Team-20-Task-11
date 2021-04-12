from transformers import AutoModelForSequenceClassification
from subtask_A.dataloader import HYPERPARAMS
import torch
from transformers import AutoModel
from sklearn.linear_model import LogisticRegression
from config import read_config

HYPERPARAMS = read_config(filename="config.ini", section="HYPERPARAMS")


class BaselineLogisticRegression:
    def __init__(self, train_embeddings, train_labels):
        self.classifier = LogisticRegression()
        self.train_embeddings = train_embeddings
        self.train_labels = train_labels

    def fit(self):
        self.classifier = self.classifier.fit(
            self.train_embeddings, self.train_labels)
        return self.classifier

    def predict(self, test_embeddings):
        predictions = self.classifier.predict(test_embeddings)
        return predictions


VanillaSciBERT = AutoModelForSequenceClassification.from_pretrained(
    'allenai/scibert_scivocab_uncased',
    output_attentions=False,
    output_hidden_states=False,
    num_labels=2)


class BERT_Linear(torch.nn.Module):
    def __init__(self):
        super(BERT_Linear, self).__init__()
        self.bert = AutoModel.from_pretrained(
            'allenai/scibert_scivocab_uncased',  output_hidden_states=True)

        self.classification_head = torch.nn.Sequential(
            torch.nn.Dropout(float(HYPERPARAMS["DROPOUT"])),
            torch.nn.Linear(768, 400),
            torch.nn.Dropout(float(HYPERPARAMS["DROPOUT"])),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 2))

    def forward(self, ids, mask, token_type_ids):
        bert_output = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)[2]
        bert_output = bert_output[12]
        bert_output = bert_output[:, 0, :].view(-1, 768)

        output = self.classification_head(bert_output)
        return output


class BERT_BiLSTM(torch.nn.Module):
    def __init__(self):
        super(BERT_BiLSTM, self).__init__()
        self.bert = AutoModel.from_pretrained(
            'allenai/scibert_scivocab_uncased',  output_hidden_states=True)
        self.bilstm = torch.nn.LSTM(768, 400,
                                    num_layers=2, batch_first=True, bidirectional=True)

        self.classification_head = torch.nn.Sequential(
            torch.nn.Dropout(float(HYPERPARAMS["DROPOUT"])),
            torch.nn.Linear(800, 200),
            torch.nn.Dropout(float(HYPERPARAMS["DROPOUT"])),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 2),
        )

    def forward(self, ids, mask, token_type_ids, lengths):
        bert_output = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)[2]
        bert_output = bert_output[12]

        _, (lstm_hidden_out, _) = self.lstm(
            torch.nn.utils.rnn.pack_padded_sequence(bert_output,
                                                    lengths.cpu(), batch_first=True, enforce_sorted=False))

        bilstm_output = torch.cat(
            (lstm_hidden_out[0], lstm_hidden_out[1]), dim=1)

        output = self.classification_head(bilstm_output)
        return output


MODELS = {
    "BaselineLogisticRegression": BaselineLogisticRegression,
    "VanillaSciBERT": VanillaSciBERT,
    "BERT_Linear": BERT_Linear,
    "BERT_BiLSTM": BERT_BiLSTM
}
