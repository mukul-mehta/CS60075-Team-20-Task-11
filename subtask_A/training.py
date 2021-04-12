import datetime
import torch
import time
import numpy as np
import random

from transformers import AutoModel, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from models import BaselineLogisticRegression, VanillaSciBERT, BERT_BiLSTM, BERT_Linear
from config import read_config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
HYPERPARAMS = read_config(filename="config.ini", section="HYPERPARAMS")


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train_baseline(train_dataloader):
    start = time.time()
    train_embeddings = []
    train_labels = []

    model = AutoModel.from_pretrained(
        "allenai/scibert_scivocab_uncased", output_hidden_states=True)

    model.to(device)

    for idx, batch in enumerate(train_dataloader):
        if idx % 10 == 0:
            print(f"Done with {idx} of {len(train_dataloader)} batches")
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_lengths = batch[2].to(device)
        b_labels = batch[3].to(device)

        with torch.no_grad():
            output = model(b_input_ids, attention_mask=b_input_mask,
                           token_type_ids=None, return_dict=False)[2][-1]
            output = output[:, 0, :].view(-1, 768)
            train_embeddings.append(output.cpu().numpy())
            train_labels.append(b_labels.cpu().numpy())

    flat_embeddings = np.concatenate(train_embeddings, axis=0)
    flat_labels = np.concatenate(train_labels, axis=0)

    baseline_model = BaselineLogisticRegression(
        train_embeddings=flat_embeddings, train_labels=flat_labels)

    classifier = baseline_model.fit()

    end = time.time()
    time_taken = end - start

    return classifier, time_taken


def train_vanilla_scibert(train_dataloader):
    model = AutoModelForSequenceClassification.from_pretrained(
        'allenai/scibert_scivocab_uncased',
        output_attentions=False,
        output_hidden_states=False,
        num_labels=2)

    optimizer = AdamW(model.parameters(),
                      lr=2e-5,
                      eps=1e-8
                      )

    epochs = int(HYPERPARAMS["EPOCHS"])

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    seed_val = int(HYPERPARAMS["SEED_VALUE"])

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    total_t0 = time.time()

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    loss_values = []

    for epoch in range(epochs):
        print(f"============= Epoch {epoch + 1} / {epochs} =============")
        print(f"============= Training =============")

        start_time = time.time()
        total_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            if step % 100 == 0 and not step == 0:
                elapsed = format_time(time.time() - start_time)
                print(
                    f"\nBatch {step} of {len(train_dataloader)}. Elapsed: {elapsed}")

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[3].to(device)

            model.zero_grad()

            output = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels)

            loss = output[0]

            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)

        loss_values.append(avg_train_loss)
        print("")
        print(f"==== Average Training Loss: {avg_train_loss} ====")
        print(f"==== Training Epoch Time: {time.time() - start_time} ====")

    end = time.time()
    print("\n")
    print("Training Completed!")
    time_taken = end - total_t0
    return model, time_taken


def train_scibert_linear(train_dataloader, train_sentences, train_labels):
    label0_sent = [i for i, j in zip(train_sentences, train_labels) if j == 0]
    label1_sent = [i for i, j in zip(train_sentences, train_labels) if j == 1]
    c_weights = [len(label1_sent), len(label0_sent)]
    c_weights = [i/sum(c_weights) for i in c_weights]

    c_weights = torch.tensor(c_weights)

    CELoss = torch.nn.CrossEntropyLoss(weight=c_weights.to(device))

    model = BERT_Linear()
    model.to(device)

    optimizer = AdamW(model.parameters(),
                      lr=2e-5,
                      eps=1e-8
                      )

    epochs = int(HYPERPARAMS["EPOCHS"])

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    seed_val = int(HYPERPARAMS["SEED_VALUE"])

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    total_t0 = time.time()

    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 50 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                    step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            _ = batch[2].to(device)
            b_labels = batch[3].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # In PyTorch, calling `model` will in turn call the model's `forward`
            # function and pass down the arguments. The `forward` function is
            # documented here:
            # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
            # The results are returned in a results object, documented here:
            # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
            # Specifically, we'll get the loss (because we provided labels) and the
            # "logits"--the model outputs prior to activation.
            outputs = model(b_input_ids,
                            b_input_mask,
                            None)

            loss = CELoss(outputs, b_labels)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

    end = time.time()
    time_taken = end - total_t0

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(time_taken))
    return model, time_taken


def train_scibert_bilstm(train_dataloader, train_sentences, train_labels):
    label0_sent = [i for i, j in zip(train_sentences, train_labels) if j == 0]
    label1_sent = [i for i, j in zip(train_sentences, train_labels) if j == 1]
    c_weights = [len(label1_sent), len(label0_sent)]
    c_weights = [i/sum(c_weights) for i in c_weights]

    c_weights = torch.tensor(c_weights)

    CELoss = torch.nn.CrossEntropyLoss(weight=c_weights.to(device))

    model = BERT_BiLSTM()
    model.to(device)

    optimizer = AdamW(model.parameters(),
                      lr=2e-5,
                      eps=1e-8
                      )

    epochs = int(HYPERPARAMS["EPOCHS"])

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    seed_val = int(HYPERPARAMS["SEED_VALUE"])

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    total_t0 = time.time()

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 50 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                    step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: lengths
            #   [3]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_lengths = batch[2].to(device)
            b_labels = batch[3].to(device)

            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # In PyTorch, calling `model` will in turn call the model's `forward`
            # function and pass down the arguments. The `forward` function is
            # documented here:
            # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
            # The results are returned in a results object, documented here:
            # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
            # Specifically, we'll get the loss (because we provided labels) and the
            # "logits"--the model outputs prior to activation.
            outputs = model(b_input_ids,
                            b_input_mask,
                            None,
                            b_lengths)

            loss = CELoss(outputs, b_labels)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

    end = time.time()
    time_taken = end - total_t0

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(time_taken))
    return model, time_taken
