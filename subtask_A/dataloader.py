import os
import torch
from config import read_config

from torch.utils.data import TensorDataset, dataloader
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


PATHS = read_config(filename="config.ini", section="DATASET")
HYPERPARAMS = read_config(filename="config.ini", section="HYPERPARAMS")


def prepare_data(input_dir, oversample=False):
    subfolders = [f.path.split("/")[-1]
                  for f in os.scandir(input_dir) if f.is_dir()]

    stanza_list = []
    stanza_len = []
    sent_num_list = []
    name_list = []

    for file in subfolders:
        for i in os.listdir(input_dir + file + '/'):
            for files in os.listdir(input_dir + file + '/' + str(i)):
                if files.endswith("Stanza-out.txt"):
                    stanza_file = open(input_dir + file + '/' +
                                       str(i) + '/' + files, "r")
                    stanza_lines = stanza_file.read()

                    stanza_lines_list = list(
                        filter(None, map(lambda x: x.lower(), stanza_lines.splitlines())))

                    stanza_len.append(len(stanza_lines_list))
                    stanza_list.append(stanza_lines_list)

                if files.endswith("sentences.txt"):
                    sentence_file = open(
                        input_dir + file + '/' + str(i) + '/' + files, "r")

                    sentence_num_list = list(
                        filter(None, sentence_file.read().splitlines()))
                    sent_num_list.append(sentence_num_list)
            name_list.append(file + '/' + str(i))

    sent_num_list = [[int(s) for s in sublist]
                     for sublist in sent_num_list]

    sent_num_list = [list(set(x)) for x in sent_num_list]

    multihot_sent = []

    for i in range(len(stanza_list)):
        temp = [0] * stanza_len[i]
        for j in range(len(sent_num_list[i])):
            t1 = sent_num_list[i][j] - 1
            temp[t1] = 1
        multihot_sent.append(temp)

    # return stanza_list, multihot_sent
    sentences = [item for sublist in stanza_list for item in sublist]
    labels = [item for sublist in multihot_sent for item in sublist]

    sent_tuple = list(set((zip(sentences, labels))))
    sentences = []
    labels = []
    for sentence, label in sent_tuple:
        if len(sentence) > 4:
            sentences.append(sentence)
            labels.append(label)

    if oversample:
        label1_sentences = int(
            PATHS["OVERSAMPLING"]) * [sentence for sentence, label in sent_tuple if label == 1]
        label0_sentences = [sentence for sentence,
                            label in sent_tuple if label == 0]

        sentences = label1_sentences + label0_sentences
        labels = len(label1_sentences) * [1] + len(label0_sentences) * [0]

    return sentences, labels


def create_dataloader(sentences, labels, tokenizer):
    input_ids = []
    attention_masks = []
    lengths = []

    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,                      # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            # Pad & truncate all sentences.
            max_length=int(HYPERPARAMS["MAX_TOKEN_LENGTH"]),
            truncation=True,
            return_length=True,
            padding="max_length",
            return_attention_mask=True,   # Construct attn. masks.
            return_tensors='pt',     # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
        lengths.append(encoded_dict['length'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)

    dataset = TensorDataset(input_ids, attention_masks, lengths, labels)

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    dataloader = DataLoader(
        dataset,  # The training samples.
        sampler=RandomSampler(dataset),  # Select batches randomly
        # Trains with this batch size.
        batch_size=int(HYPERPARAMS["BATCH_SIZE"])
    )

    return dataloader


if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Load the BERT tokenizer.
    print('Loading SciBERT tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(
        'allenai/scibert_scivocab_uncased')
    sentences, labels = prepare_data(
        input_dir=PATHS["TRAIN_DATA_PATH"], oversample=True)

    train_dataloader = create_dataloader(sentences, labels, tokenizer)

    assert (len(sentences) == 61598)
    assert(len(sentences) // 32 + 1 == len(train_dataloader))
