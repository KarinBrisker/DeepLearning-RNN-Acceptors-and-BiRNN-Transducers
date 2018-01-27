import random
import dynet as dy
import numpy as np

from collections import defaultdict
from itertools import count
import sys

global LAYERS, INPUT_DIM, HIDDEN_DIM, vocab, VOCAB_SIZE, OUTPUT_DIM, num_epochs

LAYERS = 2
INPUT_DIM = 30
HIDDEN_DIM = 35
vocab = ['a', 'b', 'c', 'd', 'start', 'end', '1', '2', '3', '4', '5', '6', '7', '8', '9']
VOCAB_SIZE = len(vocab)
OUTPUT_DIM = 2
num_epochs = 20

model = dy.Model()

lstm = dy.LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)

lookup = model.add_lookup_parameters((VOCAB_SIZE, INPUT_DIM))

# MLP after LSTM outputs
W_mlp = model.add_parameters((OUTPUT_DIM, HIDDEN_DIM))
b_mlp = model.add_parameters(OUTPUT_DIM)

def read_data(file_name):
    data = []
    row = []
    for line in file(file_name):
        text = line.strip()
        data.append(text)
    return data

def make_data_context(data, w2i):
    contexts = []
    for row in data:
        curr_tag, curr_word = row.split(' ')
        context = []
        if curr_tag == 'good':
            curr_tag = 1
        elif curr_tag == 'bad':
            curr_tag = 0
        else:
            print('error')
        context.append(curr_tag)
        context.append(w2i['start'])
        for i in range(len(curr_word)):
            context.append(w2i[curr_word[i]])
        context.append(w2i['end'])
        contexts.append(context)
    return contexts


def make_indexes_to_data(data):
    # strings to IDs
    L2I = {l: i for i, l in enumerate(data)}
    I2L = {i: l for l, i in L2I.iteritems()}
    return L2I, I2L


# return loss for one word
def do_one_word(model, word, tag):
    # setup the sentence
    dy.renew_cg()
    s0 = lstm.initial_state()

    s = s0
    
    for char in word:
        s = s.add_input(lookup[char])

    # MLP 
    W = dy.parameter(W_mlp)
    b = dy.parameter(b_mlp)

    probs = dy.softmax(W * s.output() + b)

    # loss of the model
    loss = -dy.log(dy.pick(probs, tag))
    loss.backward()
    trainer.update()
    return loss

# make prediction
def make_prediction(model, word, tag):
    # setup 
    dy.renew_cg()
    s0 = lstm.initial_state()

    W = dy.parameter(W_mlp)
    bias = dy.parameter(b_mlp)
    s = s0
    for char in word:
        s = s.add_input(lookup[char])

    # MLP after LSTM outputs
    probs = dy.softmax(W * s.output() + bias)

    prediction =  np.argmax(probs.npvalue())
    return prediction

if __name__ == '__main__':
    
    train = read_data('train')
    test = read_data('test')
    c2i, i2c = make_indexes_to_data(vocab)

    contexts = make_data_context(train, c2i)
    contexts_test = make_data_context(test, c2i)


    trainer = dy.AdagradTrainer(model)

    for epoch in range(num_epochs):
        print 1
        for line in contexts:
            tag, word = line[0], line[1:]
            loss = do_one_word(model, word, tag)
            loss_value = loss.value()
            loss.backward()
            trainer.update()
        print 2
        correct_answers = 0.0  
        for line in contexts_test:
            tag, word = line[0], line[1:]
            # setup the sentence
            prediction = make_prediction(model, word, tag)
            if tag == prediction:
                correct_answers += 1
        print(correct_answers/len(contexts_test))
        print(correct_answers)
        print(len(contexts_test))