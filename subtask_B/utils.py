import torch


def create_input(seq, tokenizer):
    for count, i in enumerate(seq):
        temp = tokenizer.tokenize(i)
        if(len(temp) > 1):
            seq[count] = temp[0]
    sentences = " ".join(seq)
    inputs = tokenizer(sentences, return_tensors="pt")
    return inputs


def log_func(vec):
    max_score = vec[0, maxval(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def maxval(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def BILOU_substring(n):
    if n == 0:
        return ''
    elif n == 1:
        return 'U'
    elif n == 2:
        return 'B L'
    else:
        t1 = 'I '*(n-2)
        return 'B '+t1+'L'
