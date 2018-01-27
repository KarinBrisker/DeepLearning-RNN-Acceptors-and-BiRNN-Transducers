
import dynet as dy
import sys
import numpy as np
import random
from collections import Counter, defaultdict
import sys
import argparse

# reads train file. adds start*2, end*2 for each sentence for appropriate windows
# split for words and tags
def read_data(file_name, is_ner):    
    counter = 0
    sent = []
    sent.append(tuple(('start','start')))
    for line in file(file_name):
        counter += 1
        if (counter%5000 == 0):
            print counter
        if len(line.strip()) == 0:
            sent.append(tuple(('end','end')))
            yield sent
            sent = []
            sent.append(tuple(('start','start')))
        elif len(line.strip()) == 1:
            continue
        else:
            if(is_ner):
                word_and_tag = line.strip().split("\t")
            else:
                word_and_tag = line.strip().split(" ")
            word = word_and_tag[0]
            tag = word_and_tag[1]
            sent.append(tuple((word,tag)))      

# indexes to data
def make_indexes_to_data(data):
    # strings to IDs
    L2I = {l:i for i,l in enumerate(data)}
    I2L = {i:l for l,i in L2I.iteritems()}
    return L2I,I2L


def build_tagging_graph(words):
    if option == 'a' or option == 'c':
        res = build_tagging_graph1(words)
        return res
    if option == 'b' or option == 'd':
        res = build_tagging_graph2(words)
        return res


def build_tagging_graph1(words):
    dy.renew_cg()
    # parameters -> expressions
    H = dy.parameter(pH)
    O = dy.parameter(pO)

    # initialize the RNNs
    f_init = fwdRNN.initial_state()
    b_init = bwdRNN.initial_state()

    # get the word vectors. word_rep(...) returns a 128-dim vector expression for each word.
    wembs = []
    if option == 'a':
        for i, w in enumerate(words):
            wembs.append(word_rep_1(w))
    if option == 'c':
        for i, w in enumerate(words):
            word,pre,suff = word_rep_3(w)
            wembs.append(word + pre + suff)
    # feed word vectors into biLSTM
    fw_exps = f_init.transduce(wembs)
    bw_exps = b_init.transduce(reversed(wembs))

    # biLSTM states
    bi_exps = [dy.concatenate([f,b]) for f,b in zip(fw_exps, reversed(bw_exps))]

    # feed each biLSTM state to an MLP
    exps = []
    for x in bi_exps:
        r_t = O*(dy.tanh(H * x))
        exps.append(r_t)

    return exps

def build_tagging_graph2(words):
    dy.renew_cg()
    # parameters -> expressions
    H = dy.parameter(pH)
    O = dy.parameter(pO)

    # initialize the RNNs
    f_init = fwdRNN.initial_state()
    b_init = bwdRNN.initial_state()

    cf_init = cFwdRNN.initial_state()
    cb_init = cBwdRNN.initial_state()

    if option == 'b':
    # get the word vectors. word_rep(...) returns a 128-dim vector expression for each word.
        wembs = [word_rep_2(w, cf_init, cb_init) for w in words]
    if option == 'd':
        wembs = [word_rep_4(w, cf_init, cb_init) for w in words]

    # feed word vectors into biLSTM
    fw_exps = f_init.transduce(wembs)
    bw_exps = b_init.transduce(reversed(wembs))

    # biLSTM states
    bi_exps = [dy.concatenate([f,b]) for f,b in zip(fw_exps, reversed(bw_exps))]

    # feed each biLSTM state to an MLP
    exps = []
    for x in bi_exps:
        r_t = O*(dy.tanh(H * x))
        exps.append(r_t)

    return exps

def word_rep_1(w):
    widx = vw[w] if wc[w] > 5 else UNK
    return WORDS_LOOKUP[widx]

def word_rep_2(w, cf_init, cb_init):
    if wc[w] > 0:
        w_index = vw[w]
        return WORDS_LOOKUP[w_index]
    else:
        pad_char = vc["<*>"]
        char_ids = [pad_char] + [vc.get(c,CUNK) for c in w] + [pad_char]
        char_embs = [CHARS_LOOKUP[cid] for cid in char_ids]
        fw_exps = cf_init.transduce(char_embs)
        bw_exps = cb_init.transduce(reversed(char_embs))
        return dy.concatenate([ fw_exps[-1], bw_exps[-1] ])

def word_rep_3(w):
    unk_prefix ="unk-prefix"
    unk_suffix = "unk-suffix"
    if len(w) >= 3:
        pref = '*prefix*' + w[:3]
        suff = '*suffix*' + w[-3:]
    else:
        pref = unk_prefix
        suff = unk_suffix
    widx = vw[w] if wc[w] > 5 else UNK
    preidx = vw[pref] if wc[pref] > 5 else vw[unk_prefix]
    suffidx = vw[suff] if wc[suff] > 5 else vw[unk_suffix]
    return [WORDS_LOOKUP[widx], WORDS_LOOKUP[preidx], WORDS_LOOKUP[suffidx]]


def word_rep_4(w, cf_init, cb_init):
    first = word_rep_1(w)
    second = word_rep_2(w, cf_init, cb_init)
    return  dy.concatenate([first, second])


def sent_loss_precalc(words, tags, vecs):
    errs = []
    for v,t in zip(vecs,tags):
        tid = vt[t]
        err = dy.pickneglogsoftmax(v, tid)
        errs.append(err)
    return dy.esum(errs)

def sent_loss(words, tags):
    return sent_loss_precalc(words, tags, build_tagging_graph(words))

def tag_sent_precalc(words, vecs):
    log_probs = [v.npvalue() for v in vecs]
    tags = []
    for prb in log_probs:
        tag = np.argmax(prb)
        tags.append(ix_to_tag[tag])
    return zip(words, tags)

def tag_sent(words):
    return tag_sent_precalc(words, build_tagging_graph(words))

def read_data_test(file_name):
    counter = 0
    sent = []
    sent.append('start')
    for line in file(file_name):
        counter += 1
        if (counter%5000 == 0):
            print counter
        if len(line.strip()) == 0:
            sent.append('end')
            yield sent
            sent = []
            sent.append('start')
        else:
            word = line.strip()
            sent.append(word)

if __name__ == '__main__':
    is_ner = True
    option =  sys.argv[1]
    train = list(read_data('copy.txt', is_ner))

    if option == 'a':
        words=[]
        tags=[]
        wc=Counter()
        for sent in train:
            for w,p in sent:
                words.append(w)
                tags.append(p)
                wc[w]+=1
        words.append("_UNK_")

    if option == 'b' or option == 'd':
        chars=set()
        words=[]
        tags=[]
        wc=Counter()
        for sent in train:
            for w,p in sent:

                words.append(w)
                tags.append(p)
                chars.update(w)

                wc[w]+=1
        words.append("_UNK_")
        chars.add("<*>")
        chars.add("_UNK_")

        vc , ix_to_char = make_indexes_to_data(set(chars))
        CUNK = vc["_UNK_"]
        nchars  = len(vc)

    if option == 'c':
        words=[]
        tags=[]
        wc=Counter()
        for sent in train:
            for w,p in sent:
                words.append(w)
                if len(w) >= 3:
                    pref = '*prefix*' + w[:3]
                    suff = '*suffix*' + w[-3:]
                tags.append(p)
                wc[w]+=1
        words.append("_UNK_")
        words.append("unk-suffix")
        words.append("unk-prefix")

    vw, ix_to_word = make_indexes_to_data(set(words))
    vt, ix_to_tag = make_indexes_to_data(set(tags))

    UNK = vw["_UNK_"]

    nwords = len(vw)
    print nwords
    print 'aaa'
    ntags  = len(vt)

    print("yesh")
    model = dy.Model()
    trainer = dy.AdamTrainer(model)

    if option == 'c':
        WORDS_LOOKUP = model.add_lookup_parameters((nwords, 50))
    else:
        WORDS_LOOKUP = model.add_lookup_parameters((nwords, 50))

    if option == 'b' or option == 'd':
        CHARS_LOOKUP = model.add_lookup_parameters((nchars, 50))

    # MLP on top of biLSTM outputs 100 -> 32 -> ntags
    pH = model.add_parameters((200, 10*2))
    pO = model.add_parameters((len(set(tags)),  200))

    model.populate(sys.argv[2])

    if option == 'c':
        # word-level LSTMs
        fwdRNN = dy.VanillaLSTMBuilder(1, 50, 10, model) # layers, in-dim, out-dim, model
        bwdRNN = dy.VanillaLSTMBuilder(1, 50, 10, model)
    elif option == 'd':
        fwdRNN = dy.VanillaLSTMBuilder(1, 100, 10, model) # layers, in-dim, out-dim, model
        bwdRNN = dy.VanillaLSTMBuilder(1, 100, 10, model)

        cFwdRNN = dy.VanillaLSTMBuilder(1, 50, 25, model)
        cBwdRNN = dy.VanillaLSTMBuilder(1, 50, 25, model)
    else:
        # word-level LSTMs
        fwdRNN = dy.VanillaLSTMBuilder(1, 50, 10, model) # layers, in-dim, out-dim, model
        bwdRNN = dy.VanillaLSTMBuilder(1, 50, 10, model)

    if option == 'b':
        # char-level LSTMs
        cFwdRNN = dy.VanillaLSTMBuilder(1, 50, 25, model)
        cBwdRNN = dy.VanillaLSTMBuilder(1, 50, 25, model)        

    test_file = open(sys.argv[3], "w")
    test = read_data_test(input_file)
    for s in test:
        words = [w for w in s]
        tags = [t for w,t in tag_sent(words)]
        for k in range(1, len(words)-1):
            test_file.write(words[k] + ' ' + tags[k])
            print (words[k] + ' ' + tags[k])
            test_file.write("\n")
        test_file.write("\n")
        print ("\n")

