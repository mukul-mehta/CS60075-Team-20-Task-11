
import os
import copy
import random
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import time
import utils

from transformers import AutoTokenizer, AutoModel
from models import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'


val = 42
random.seed(val)
np.random.seed(val)
torch.manual_seed(val)
torch.cuda.manual_seed_all(val)

# Training set

inp_dir = "../datasets/train/"
folders = ["query_wellformedness", "passage_re-ranking", "part-of-speech_tagging",
           "sentence_compression", "sentiment_analysis", "temporal_information_extraction",
           "phrase_grounding", "text_generation", "text-to-speech_synthesis",
           "smile_recognition", "topic_models", "question_generation",
           "relation_extraction", "paraphrase_generation", "question_similarity",
           "question_answering", "sentence_classification", "prosody_prediction",
           "semantic_role_labeling", "text_summarization", "semantic_parsing",
           "sarcasm_detection", "natural_language_inference", "negation_scope_resolution"]

in_stanza_list = []
in_sent_numbers = []
in_entities = []
file_name_list = []
total_phrases_truth = 0

for folder in folders:
    cnt = 0
    for i in os.listdir(inp_dir + folder + '/'):
        cnt = cnt+1
        for files in os.listdir(inp_dir + folder + '/' + str(i)):
            if files.endswith("Stanza-out.txt"):
                stanza_file = open(inp_dir + folder + '/' +
                                   str(i) + '/' + files, "r")
                stanza_lines = (stanza_file.read()).lower()
                stanza_lines_list = list(
                    filter(None, stanza_lines.splitlines()))
                in_stanza_list.append(stanza_lines_list)
            if files.endswith("sentences.txt"):
                sentence_file = open(inp_dir + folder +
                                     '/' + str(i) + '/' + files, "r")
                sentence_num_list = list(
                    filter(None, (sentence_file.read().lower()).splitlines()))
                in_sent_numbers.append(list(map(int, sentence_num_list)))
            if files.endswith("entities.txt"):
                entities_file = open(inp_dir + folder +
                                     '/' + str(i) + '/' + files, "r")
                entities_list = list(
                    filter(None, (entities_file.read().lower()).splitlines()))
                in_entities.append(entities_list)
                total_phrases_truth = total_phrases_truth + len(entities_list)
        file_name_list.append(folder + '/' + str(i))

# Validation Set

valid_inp_dir = "../datasets/validation/"
valid_list_of_folders = ["machine-translation", "named-entity-recognition",
                         "question-answering", "relation-classification", "text-classification"]
valid_in_stanza_list = []
valid_in_sent_numbers = []
valid_in_entities = []
valid_file_name_list = []
valid_total_phrases_truth = 0
for folder in valid_list_of_folders:
    cnt = 0
    for i in os.listdir(valid_inp_dir + folder + '/'):
        cnt = cnt+1
        for files in os.listdir(valid_inp_dir + folder + '/' + str(i)):
            if files.endswith("Stanza-out.txt"):
                stanza_file = open(valid_inp_dir + folder +
                                   '/' + str(i) + '/' + files, "r")
                stanza_lines = stanza_file.read().lower()
                stanza_lines_list = list(
                    filter(None, stanza_lines.splitlines()))
                valid_in_stanza_list.append(stanza_lines_list)
            if files.endswith("sentences.txt"):
                sentence_file = open(
                    valid_inp_dir + folder + '/' + str(i) + '/' + files, "r")
                sentence_num_list = list(
                    filter(None, (sentence_file.read().lower()).splitlines()))
                valid_in_sent_numbers.append(list(map(int, sentence_num_list)))
            if files.endswith("entities.txt"):
                entities_file = open(
                    valid_inp_dir + folder + '/' + str(i) + '/' + files, "r")
                entities_list = list(
                    filter(None, (entities_file.read().lower()).splitlines()))
                valid_in_entities.append(entities_list)
                valid_total_phrases_truth = valid_total_phrases_truth + \
                    len(entities_list)
        valid_file_name_list.append(folder + '/' + str(i))

# Training set loading

taskB_in = []
taskB_label = []

for i in range(len(in_stanza_list)):
    try:
        entity_list = [j.split('\t') for j in in_entities[i]]
        entity_list.sort(key=lambda x: (int(x[0]), int(x[1])))
        sent_num_list = copy.deepcopy(in_sent_numbers[i])
        sent_num_list.sort()
        sent_list = []

        for x in sent_num_list:
            sent_list.append(in_stanza_list[i][x-1])

        sent_dict_list = dict(zip(sent_num_list, sent_list))
        for n, ind_s, ind_e, ph in entity_list:
            if int(n) in sent_num_list:
                sent_dict_list[int(n)] = sent_dict_list[int(n)].replace(
                    ph, utils.BILOU_substring(len(ph.split())), 1)

        sent_label_list = list(sent_dict_list.values())
        taskB_in.append(sent_list)
        taskB_label.append(sent_label_list)
    except:
        pass

# Validation set loading

valid_taskB_in = []
valid_taskB_label = []
for i in range(len(valid_in_stanza_list)):
    valid_entity_list = [j.split('\t') for j in valid_in_entities[i]]
    valid_entity_list.sort(key=lambda x: (int(x[0]), int(x[1])))
    valid_sent_num_list = copy.deepcopy(valid_in_sent_numbers[i])
    valid_sent_num_list.sort()
    valid_sent_list = []

    for x in valid_sent_num_list:
        valid_sent_list.append(valid_in_stanza_list[i][x-1])

    valid_sent_dict_list = dict(zip(valid_sent_num_list, valid_sent_list))
    for n, ind_s, ind_e, ph in valid_entity_list:
        if int(n) in valid_sent_num_list:
            valid_sent_dict_list[int(n)] = valid_sent_dict_list[int(n)].replace(
                ph, utils.BILOU_substring(len(ph.split())), 1)

    valid_sent_label_list = list(valid_sent_dict_list.values())
    valid_taskB_in.append(valid_sent_list)
    valid_taskB_label.append(valid_sent_label_list)


for i, out in enumerate(taskB_label):
    for j, line in enumerate(out):
        for k, tok in enumerate(line.split()):
            if tok not in ['B', 'I', 'L', 'U']:
                taskB_label[i][j] = taskB_label[i][j].replace(tok, 'O', 1)


for i, out in enumerate(valid_taskB_label):
    for j, line in enumerate(out):
        for k, tok in enumerate(line.split()):
            if tok not in ['B', 'I', 'L', 'U']:
                valid_taskB_label[i][j] = valid_taskB_label[i][j].replace(
                    tok, 'O', 1)

# TRAINING MODEL


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 768
HIDDEN_DIM = 200


biluo_code = {"B": 0, "I": 1, "L": 2, "U": 3,
              "O": 4, START_TAG: 5, STOP_TAG: 6}
biluo_decode = {v: k for k, v in biluo_code.items()}


tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")


model = BiLSTM_CRF(biluo_code, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
if (device == 'cuda'):
    model.cuda()

# Evaluation functions


def eval_func(listp, listg):
    correct = 0
    total_p = 0
    for i in zip(listp, listg):
        flag = 0
        for j in zip(i[0], i[1]):
            if j[0] == 'U' or j[0] == 'B':
                total_p += 1
            if flag == 0:
                if j[0] == 'U' and j[1] == 'U':
                    correct += 1
                if j[0] == 'B' and j[1] == 'B':
                    flag = 1
            else:
                if j[0] == 'L' and j[1] == 'L':
                    correct += 1
                    flag = 0
                elif j[0] == j[1]:
                    continue
                else:
                    flag = 0
    return correct, total_p


def training_loop():

    taskB_out_label = []
    for i, file in enumerate(taskB_in):
        output = []
        for j, sent in enumerate(file):
            with torch.no_grad():
                precheck_sent = utils.create_input(
                    taskB_in[i][j].split(), tokenizer).to(device)
                sent_out_label = model(precheck_sent)[1]
                sent_str_label = [biluo_decode[t] for t in sent_out_label]
                output.append(sent_str_label)
        taskB_out_label.append(output)

    true_pos = 0
    total_ph_pred = 0
    for i, file in enumerate(taskB_label):
        file_true_pos, file_total_ph_pred = eval_func(
            taskB_out_label[i], [["O"] + s.split() + ["O"] for s in taskB_label[i]])
        true_pos = true_pos + file_true_pos
        total_ph_pred = total_ph_pred + file_total_ph_pred

    precision = 0
    recall = 0
    F1score = 0
    if(total_ph_pred != 0):
        precision = true_pos/total_ph_pred
    if(total_phrases_truth != 0):
        recall = true_pos/total_phrases_truth
    if((precision + recall) != 0):
        F1score = 2 * precision*recall/(precision+recall)
    print("Precision : {} | Recall : {} | F1 Score : {}".format(
        precision, recall, F1score))


def validation_loop():

    valid_taskB_out_label = []
    for i, file in enumerate(valid_taskB_in):
        output = []
        for j, sent in enumerate(file):
            with torch.no_grad():
                precheck_sent = utils.create_input(
                    valid_taskB_in[i][j].split(), tokenizer).to(device)
                sent_out_label = model(precheck_sent)[1]
                sent_str_label = [biluo_decode[t] for t in sent_out_label]
                output.append(sent_str_label)
        valid_taskB_out_label.append(output)

    valid_true_pos = 0
    valid_total_ph_pred = 0
    for i, file in enumerate(valid_taskB_label):
        valid_file_true_pos, valid_file_total_ph_pred = eval_func(
            valid_taskB_out_label[i], [["O"] + s.split() + ["O"] for s in valid_taskB_label[i]])
        valid_true_pos = valid_true_pos + valid_file_true_pos
        valid_total_ph_pred = valid_total_ph_pred + valid_file_total_ph_pred

    valid_precision = 0
    valid_recall = 0
    valid_F1score = 0
    if(valid_total_ph_pred != 0):
        valid_precision = valid_true_pos/valid_total_ph_pred
    if(valid_total_phrases_truth != 0):
        valid_recall = valid_true_pos/valid_total_phrases_truth
    if((valid_precision + valid_recall) != 0):
        valid_F1score = 2 * valid_precision * \
            valid_recall/(valid_precision+valid_recall)
    print("Precision : {} | Recall : {} | F1 Score : {}".format(
        valid_precision, valid_recall, valid_F1score))


# Main Training Loop

with torch.no_grad():

    precheck_sent = utils.create_input(
        taskB_in[0][0].split(), tokenizer).to(device)
    precheck_tags = torch.tensor(
        [4] + [biluo_code[t] for t in taskB_label[0][0].split()] + [4], dtype=torch.long).to(device)
    print("Checkpoint reached! Starting model training......")


for epoch in range(4):
    start = time.time()
    model.train()

    for i, file in enumerate(taskB_in):
        try:
            if (i % 10 == 0):
                print(f"done with {i} of {len(taskB_in)}")
            for j, sent in enumerate(file):

                model.zero_grad()

                sentence_in = utils.create_input(
                    sent.split(), tokenizer).to(device)

                targets = torch.tensor(
                    [4] + [biluo_code[t] for t in taskB_label[i][j].split()] + [4], dtype=torch.long).to(device)

                loss = model.neg_log_likelihood(sentence_in, targets)

                loss.backward()

                optimizer.step()
        except:
            pass

    end = time.time()

    torch.save(model, "./model" + str(epoch) + ".pt")

    print("Epoch ", epoch, " completed. Time taken : ", end-start)

    model.eval()
    training_loop()
    validation_loop()
    valid_total_loss = 0.
    with torch.no_grad():
        for k, valid_file in enumerate(valid_taskB_in):
            for l, valid_sent in enumerate(valid_file):
                valid_sentence_in = utils.create_input(
                    valid_sent.split(), tokenizer).to(device)
                valid_targets = torch.tensor(
                    [4] + [biluo_code[t] for t in valid_taskB_label[k][l].split()] + [4], dtype=torch.long).to(device)
                loss = model.neg_log_likelihood(
                    valid_sentence_in, valid_targets)
                valid_total_loss += loss.item()

    print("Validation loss after epoch", epoch, " = ", valid_total_loss)


with torch.no_grad():
    precheck_sent = utils.create_input(
        taskB_in[0][0].split(), tokenizer).to(device)
    print(model(precheck_sent))

# Loading trained model

model = torch.load("./model0.pt")
model.eval()


# Test set

test_inp_dir = "../datasets/test/"

test_list_of_folders = ["constituency_parsing", "coreference_resolution",
                        "data-to-text_generation", "dependency_parsing",
                        "document_classification", "entity_linking",
                        "face_alignment", "face_detection", "hypernym_discovery",
                        "natural_language_inference"]

stanza_list_test = []
stanza_sent_numbers = []
entities_test = []
filename_list_entities = []
test_total_phrases_truth = 0
Capital_stanza_list_test = []

for folder in test_list_of_folders:
    cnt = 0
    for i in os.listdir(test_inp_dir + folder + '/'):
        cnt = cnt+1
        for files in os.listdir(test_inp_dir + folder + '/' + str(i)):
            if files.endswith("Stanza-out.txt"):
                stanza_file = open(test_inp_dir + folder +
                                   '/' + str(i) + '/' + files, "r")
                print(test_inp_dir + folder + '/' + str(i))
                Capital_stanza_lines = stanza_file.read()
                Capital_stanza_lines_list = list(
                    filter(None, Capital_stanza_lines.splitlines()))
                Capital_stanza_list_test.append(Capital_stanza_lines_list)

                stanza_lines = Capital_stanza_lines.lower()
                stanza_lines_list = list(
                    filter(None, stanza_lines.splitlines()))
                stanza_list_test.append(stanza_lines_list)
            if files.endswith("sentences.txt"):
                sentence_file = open(
                    test_inp_dir + folder + '/' + str(i) + '/' + 'sentences.txt', "r")
                sentence_num_list = list(
                    filter(None, (sentence_file.read().lower()).splitlines()))
                stanza_sent_numbers.append(list(map(int, sentence_num_list)))

        filename_list_entities.append(folder + '/' + str(i))


test_taskB_in = []
Capital_test_taskB_in = []

for i in range(len(stanza_sent_numbers)):

    test_sent_num_list = copy.deepcopy(stanza_sent_numbers[i])
    test_sent_num_list.sort()
    test_sent_list = []
    Capital_test_sent_list = []

    for x in test_sent_num_list:
        test_sent_list.append(stanza_list_test[i][x-1])
        Capital_test_sent_list.append(Capital_stanza_list_test[i][x-1])

    test_taskB_in.append(test_sent_list)
    Capital_test_taskB_in.append(Capital_test_sent_list)


list_of_dict_for_number_to_sentence = []
for i in range(len(stanza_list_test)):

    test_sent_num_list = copy.deepcopy(stanza_sent_numbers[i])
    test_sent_num_list.sort()
    test_sent_list = []

    for x in test_sent_num_list:
        test_sent_list.append(stanza_list_test[i][x-1])
    test_sent_dict_list = dict(zip(test_sent_num_list, test_sent_list))
    list_of_dict_for_number_to_sentence.append(test_sent_dict_list)


list_of_dict_for_sentence_to_number = [
    dict((v, k) for k, v in a.items()) for a in list_of_dict_for_number_to_sentence]


test_taskB_out_label = []

for i, file in enumerate(test_taskB_in):
    output = []
    for j, sent in enumerate(file):
        with torch.no_grad():
            precheck_sent = utils.create_input(
                test_taskB_in[i][j].split(), tokenizer).to(device)
            sent_out_label = model(precheck_sent)[1]
            sent_str_label = [biluo_decode[t] for t in sent_out_label]
            output.append(sent_str_label)
    test_taskB_out_label.append(output)

# Writing predictions in file

for i, file in enumerate(test_taskB_in):

    print(filename_list_entities[i])

    f1 = open(test_inp_dir + filename_list_entities[i] + "/entities.txt", "w")
    f1.seek(0)
    f1.truncate()

    for j, sent in enumerate(file):

        biluo_list = (test_taskB_out_label[i][j])[1:-1]
        respective_sentence = Capital_test_taskB_in[i][j].split()
        sentence_number = (list_of_dict_for_sentence_to_number[i])[
            test_taskB_in[i][j]]

        if(len(respective_sentence) != len(biluo_list)):
            print("Length mismatch in the sentence and BILUO sequence")
            continue

        temp_phrase_storer = []
        temp_phrase = []
        cnt_of_words_in_sentence = 0

        for k in zip(biluo_list, respective_sentence):

            if (k[0] == "U"):
                temp_phrase_storer = temp_phrase_storer + [k[1]]

                start_of_word = 0
                if(cnt_of_words_in_sentence == 0):
                    start_of_word = 0
                else:
                    start_of_word = len(
                        (" ".join(respective_sentence[0:cnt_of_words_in_sentence])).strip() + " ")

                end_of_word = start_of_word + len(k[1].strip())

                f1.write(str(sentence_number) + "\t" + str(start_of_word) +
                         "\t" + str(end_of_word) + "\t" + k[1].strip() + "\n")

            elif (k[0] == "B"):
                temp_phrase = temp_phrase + [k[1]]

            elif (k[0] == "I"):
                temp_phrase = temp_phrase + [" ", k[1]]

            elif (k[0] == "L"):
                temp_phrase = temp_phrase + [" ", k[1]]

                end_of_words = len((" ".join(respective_sentence[0:cnt_of_words_in_sentence])).strip(
                ) + " ") + len(respective_sentence[cnt_of_words_in_sentence].strip())
                start_of_words = end_of_words - \
                    len(("".join(temp_phrase)).strip())

                f1.write(str(sentence_number) + "\t" + str(start_of_words) + "\t" +
                         str(end_of_words) + "\t" + ("".join(temp_phrase)).strip() + "\n")
                temp_phrase_storer = temp_phrase_storer + \
                    copy.deepcopy(["".join(temp_phrase)])
                temp_phrase = []

            cnt_of_words_in_sentence += 1

    f1.close()
print("Program complete!")
