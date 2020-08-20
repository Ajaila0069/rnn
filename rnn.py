import numpy as np
from nltk.tokenize import RegexpTokenizer

def normalize(lis):
    maximum = max(lis)
    new_list = [i/maximum for i in lis]
    return new_list

class rnn:

    def __init__(self, x, lines, names, n_a=50, seq_len=10):
        self.train_x = x
        self.numa = n_a
        self.seq_len = seq_len
        self.lines = lines
        self.times = names

        self.tokenizer = RegexpTokenizer(r'\w+')
        self.lineset = [x.lower().strip() for x in self.lines]
        #print(self.lineset)
        self.linelist = [self.tokenizer.tokenize(line) for line in self.lineset]
        self.tokens = self.tokenizer.tokenize(self.train_x)

        #print(self.tokens)
        #print(self.linelist)

        self.uniquex = np.unique(self.tokens).tolist()
        self.uniquex.append('\n')
        self.xsize = len(self.uniquex)
        self.ysize = self.xsize

        self.ch_index = dict((c, i) for (i, c) in enumerate(sorted(self.uniquex)))
        self.index_ch = dict((i, c) for (i, c) in enumerate(sorted(self.uniquex)))

        self.parameters = self.initialize_parameters(self.numa, self.xsize, self.ysize)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def get_initial_loss(self):
        return -np.log(1.0/self.xsize)*self.seq_len

    def smooth(self, loss, cur_loss):
        return loss * 0.999 + cur_loss * 0.001

    def initialize_parameters(self, n_a, n_x, n_y):
        Wax = np.random.randn(n_a, n_x)*0.01  # input to hidden
        Waa = np.random.randn(n_a, n_a)*0.01  # hidden to hidden
        Wya = np.random.randn(n_y, n_a)*0.01  # hidden to output
        ba = np.zeros((n_a, 1))  # hidden bias
        by = np.zeros((n_y, 1))  # output bias

        parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
        return parameters

    def update_parameters(self, gradients, lr):

        self.parameters['Wax'] += -lr * gradients['dWax']
        self.parameters['Waa'] += -lr * gradients['dWaa']
        self.parameters['Wya'] += -lr * gradients['dWya']
        self.parameters['ba'] += -lr * gradients['db']
        self.parameters['by'] += -lr * gradients['dby']

    def rnn_cell_forward(self, xt, a_prev):

        Wax = self.parameters["Wax"]
        Waa = self.parameters["Waa"]
        Wya = self.parameters["Wya"]
        ba = self.parameters["ba"]
        by = self.parameters["by"]

        a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
        #print(a_next.shape)
        yt_pred = self.softmax(np.dot(Wya, a_next) + by)
        #print(yt_pred.shape)

        cache = (a_next, a_prev, xt, self.parameters)

        return a_next, yt_pred, cache

    def rnn_forward(self, X, Y, a0):
        x, a, y_hat = {}, {}, {}
        a[-1] = np.copy(a0)
        loss = 0
        for t in range(len(X)):
            x[t] = np.zeros((self.xsize, 1))
            if (X[t] != None):
                #print(len(x))
                x[t][X[t]] = 1
            a[t], y_hat[t], _ = self.rnn_cell_forward(x[t], a[t-1])
            #print(len(y_hat[t]))
            loss -= np.log(y_hat[t][Y[t], 0])
        cache = (y_hat, a, x)
        return loss, cache

    def rnn_cell_backward(self, dy, gradients, x, a, a_prev):
        gradients['dWya'] += np.dot(dy, a.T)
        gradients['dby'] += dy
        da = np.dot(self.parameters['Wya'].T, dy) +         gradients['da_next']  # backprop into h
        daraw = (1 - a * a) * da  # backprop through tanh nonlinearity
        gradients['db'] += daraw
        gradients['dWax'] += np.dot(daraw, x.T)
        gradients['dWaa'] += np.dot(daraw, a_prev.T)
        gradients['da_next'] = np.dot(self.parameters['Waa'].T, daraw)
        return gradients

    def rnn_backward(self, X, Y, cache):
        gradients = {}
        (y_hat, a, x) = cache
        Waa, Wax, Wya, by, ba = self.parameters['Waa'], self.parameters['Wax'], self.parameters['Wya'], self.parameters['by'], self.parameters['ba']
        gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
        gradients['db'], gradients['dby'] = np.zeros_like(ba), np.zeros_like(by)
        gradients['da_next'] = np.zeros_like(a[0])

        for t in reversed(range(len(X))):
            dy = np.copy(y_hat[t])
            dy[Y[t]] -= 1
            gradients = self.rnn_cell_backward(
                dy, gradients, x[t], a[t], a[t-1])

        return gradients, a

    def clip(self, gradients, maxValue):
        dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']

        for gradient in [dWax, dWaa, dWya, db, dby]:
            np.clip(gradient, -maxValue, maxValue, out=gradient)

        gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

        return gradients

    def optimize(self, X, Y, a_prev, learning_rate = 0.01):
        loss, cache = self.rnn_forward(X, Y, a_prev)

        gradients, a = self.rnn_backward(X, Y, cache)

        gradients = self.clip(gradients, 5)

        self.update_parameters(gradients, learning_rate)

        return loss, gradients, a[len(X)-1]

    def sample(self, seed):
        Waa, Wax, Wya, by, ba = self.parameters['Waa'], self.parameters['Wax'], self.parameters['Wya'], self.parameters['by'], self.parameters['ba']
        vocab_size = by.shape[0]
        n_a = Waa.shape[1]
        x = np.zeros((vocab_size, 1))
        a_prev = np.zeros((n_a, 1))
        indices = []

        idx = -1

        counter = 0
        newline_character = self.ch_index['\n']

        while (idx != newline_character and counter != 50):
            a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + ba)
            z = np.dot(Wya, a) + by
            y = self.softmax(z)

            idx = np.random.choice(list(range(vocab_size)), p=y.flatten())

            indices.append(idx)

            x = np.zeros((vocab_size, 1))
            x[idx] = 1

            a_prev = a

            counter +=1

        if (counter == 50):
            indices.append(self.ch_index['\n'])

        return indices

    def train(self, its=10000, checkpoint=500):
        loss = self.get_initial_loss()
        a_prev = np.zeros((self.numa, 1))

        #tokens = [x.lower().strip() for x in self.lines]
        #print(tokens)

        tokens = self.linelist
        #print(tokens)

        for i in range(its):

            index = i % len(tokens)

            #print(tokens[index])

            x = [None] + [self.ch_index[ch] for ch in tokens[index]]
            y = x[1:] + [self.ch_index['\n']]

            new_loss, grads, a_prev = self.optimize(x, y, a_prev)
            loss = self.smooth(loss, new_loss)

            if (i != 0) and (i % checkpoint == 0):
                seed = 0
                print("iteration: ", i)
                for i in range(self.times):
                    indices = self.sample(seed)
                    stuff = [self.index_ch[i] for i in indices]
                    text = ''.join((' ' + str(i)) for i in stuff)
                    print(text, end='')
                print('\n')

        return text



with open('data.txt', 'r') as t:
    text = t.read()
    text = text.lower()

lines = open('data.txt').readlines()
whereswaldo = rnn(text, lines, 8)
text = whereswaldo.train(400000, 1000)

with open('final.txt', 'w') as f:
    f.write(text)
    f.close()
