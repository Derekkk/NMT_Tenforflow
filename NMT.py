# File: Neural Machine Translation using IWSLT'15 English-Vietnamese data
# -*- coding: utf-8 -*-
# @Time    : 11/27/2018 1:58 PM
# @Author  : Derek Hu
import time
start = time.perf_counter()
import tensorflow as tf
from tensorflow.contrib import rnn
from nltk.tokenize import word_tokenize
import sys
import re
import collections
import pickle
import argparse
import numpy as np
import os
import nltk
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec


train_en_path = "data/iwslt15/train_2.en"
train_de_path = "data/iwslt15/train_2.vi"
valid_en_path = "data/iwslt15/tst2013.en"
valid_de_path = "data/iwslt15/tst2013.vi"

def add_arguments(parser):
    parser.add_argument("--num_hidden", type=int, default=512, help="Network size.")
    parser.add_argument("--num_layers", type=int, default=2, help="Network depth.")
    parser.add_argument("--beam_width", type=int, default=10, help="Beam width for beam search decoder.")
    parser.add_argument("--glove", action="store_true", help="Use glove as initial word embedding.")
    parser.add_argument("--embedding_size", type=int, default=300, help="Word embedding size.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--keep_prob", type=float, default=0.8, help="Dropout keep prob.")
    parser.add_argument("--toy", default=False, help="Use only 50K samples of data")

    #parser.add_argument("--train", default=False, help="mode selection")
    #parser.add_argument("--test", default=False, help="mode selection")
    parser.add_argument("mode", default="translate", help="mode selection")

parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()


# Part 1: data processing
def clean_str(sentence):
    sentence = re.sub("[#.]+", "#", sentence)
    return sentence

def get_text_list(data_path, toy):
    with open(data_path, "r", encoding="utf-8") as f:
        if not toy:
            return [clean_str(x.strip()) for x in f.readlines()]
        else:
            return [clean_str(x.strip()) for x in f.readlines()][:50000]

def build_dict(step, toy=False):
    if step == "train":
        train_en_list = get_text_list(train_en_path, toy)
        train_de_list = get_text_list(train_de_path, toy)

        words_en = list()
        words_de = list()
        for sentence in train_en_list:
            for word in word_tokenize(sentence):
                words_en.append(word)
        for sentence in train_de_list:
            for word in word_tokenize(sentence):
                words_de.append(word)

        # English
        word_counter_en = collections.Counter(words_en).most_common(50000)
        word_dict_en = dict()
        word_dict_en["<padding>"] = 0
        word_dict_en["<unk>"] = 1
        word_dict_en["<s>"] = 2
        word_dict_en["</s>"] = 3
        for word, _ in word_counter_en:
            word_dict_en[word] = len(word_dict_en)
        with open("word_dict_en.pickle", "wb") as f:
            pickle.dump(word_dict_en, f)
        # De
        word_counter_de = collections.Counter(words_de).most_common(50000)
        word_dict_de = dict()
        word_dict_de["<padding>"] = 0
        word_dict_de["<unk>"] = 1
        word_dict_de["<s>"] = 2
        word_dict_de["</s>"] = 3
        for word, _ in word_counter_de:
            word_dict_de[word] = len(word_dict_de)
        with open("word_dict_de.pickle", "wb") as f:
            pickle.dump(word_dict_de, f)

    elif step == "valid":
        with open("word_dict_en.pickle", "rb") as f:
            word_dict_en = pickle.load(f)
        with open("word_dict_de.pickle", "rb") as f:
            word_dict_de = pickle.load(f)

    reversed_dict_en = dict(zip(word_dict_en.values(), word_dict_en.keys()))
    reversed_dict_de = dict(zip(word_dict_de.values(), word_dict_de.keys()))
    en_max_len = 50
    de_max_len = 50
    return word_dict_en, word_dict_de, reversed_dict_en, reversed_dict_de, en_max_len, de_max_len

def build_dataset(step, word_dict_en, word_dict_de, en_max_len, de_max_len, toy=False):
    if step == "train":
        en_list = get_text_list(train_en_path, toy)
        de_list = get_text_list(train_de_path, toy)
    elif step == "valid":
        en_list = get_text_list(valid_en_path, toy)
        de_list = get_text_list(valid_de_path, toy)
    else:
        raise NotImplementedError
    # English
    x = [word_tokenize(d) for d in en_list]
    x = [[word_dict_en.get(w, word_dict_en["<unk>"]) for w in d] for d in
         x]  # if not existed, return word_dict_en["<unk>"]
    x = [d[:en_max_len] for d in x]
    x = [d + (en_max_len - len(d)) * [word_dict_en["<padding>"]] for d in x]

    if step == "valid":
        return x, de_list
    else:
        # DE: using split()
        y = [word_tokenize(d) for d in de_list]
        y = [[word_dict_de.get(w, word_dict_de["<unk>"]) for w in d] for d in
             y]  # if not existed, return word_dict_en["<unk>"]
        y = [d[:(de_max_len - 1)] for d in y]
        return x, y


def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        print("-------" * 5, "epoch: ", epoch, "-------" * 5)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]

def get_init_embedding(reversed_dict, embedding_size):
    glove_file = "glove/glove.42B.300d.txt"
    word2vec_file = get_tmpfile("word2vec_format.vec")
    glove2word2vec(glove_file, word2vec_file)
    print("Loading Glove vectors...")
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)

    word_vec_list = list()
    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = word_vectors.word_vec(word)
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32)

        word_vec_list.append(word_vec)
    # Assign random vector to <s>, </s> token
    word_vec_list[2] = np.random.normal(0, 1, embedding_size)
    word_vec_list[3] = np.random.normal(0, 1, embedding_size)
    return np.array(word_vec_list)

# Part 2: Define Seq2Seq model using tensorflow seq2seq library
class Model(object):
    def __init__(self, reversed_dict_en, reversed_dict_de, article_max_len, summary_max_len, args, forward_only=False):
        self.vocabulary_size_en = len(reversed_dict_en)
        self.vocabulary_size_de = len(reversed_dict_de)
        self.embedding_size = args.embedding_size
        self.num_hidden = args.num_hidden
        self.num_layers = args.num_layers
        self.learning_rate = args.learning_rate
        self.beam_width = args.beam_width
        if not forward_only:
            self.keep_prob = 0.8
        else:
            self.keep_prob = 1.0
        self.cell = tf.nn.rnn_cell.LSTMCell
        with tf.variable_scope("decoder/projection"):
            self.projection_layer = tf.layers.Dense(self.vocabulary_size_de, use_bias=False)

        self.batch_size = tf.placeholder(tf.int32, (), name="batch_size")
        self.X = tf.placeholder(tf.int32, [None, article_max_len], name='encoder_inputs')
        self.X_len = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')
        self.decoder_input = tf.placeholder(tf.int32, [None, summary_max_len], name='decoder_inputs')
        self.decoder_len = tf.placeholder(tf.int32, [None], name='decoder_inputs_length')
        self.decoder_target = tf.placeholder(tf.int32, [None, summary_max_len])
        self.global_step = tf.Variable(0, trainable=False)

        with tf.name_scope("embedding_en"):
            self.init_embeddings_en = tf.random_uniform([self.vocabulary_size_en, self.embedding_size], -1.0, 1.0)
            self.embeddings_en = tf.get_variable("embeddings_en", initializer=self.init_embeddings_en, trainable=True)
            self.encoder_emb_inp = tf.transpose(tf.nn.embedding_lookup(self.embeddings_en, self.X), perm=[1, 0, 2]) #Tensor("embedding/transpose:0", shape=(50, ?, 300), dtype=float32)
        with tf.name_scope("embedding_de"):
            self.embeddings_de = tf.get_variable("embeddings_de",
                                                 initializer=tf.random_uniform([self.vocabulary_size_de, self.embedding_size], -1.0, 1.0),
                                                 trainable=True)
            self.decoder_emb_inp = tf.transpose(tf.nn.embedding_lookup(self.embeddings_de, self.decoder_input), perm=[1, 0, 2]) #Tensor("embedding/transpose_1:0", shape=(15, ?, 300), dtype=float32)

        with tf.name_scope("encoder"):
            fw_cells = [self.cell(self.num_hidden) for _ in range(self.num_layers)]
            bw_cells = [self.cell(self.num_hidden) for _ in range(self.num_layers)]
            fw_cells = [rnn.DropoutWrapper(cell) for cell in fw_cells]
            bw_cells = [rnn.DropoutWrapper(cell) for cell in bw_cells]

            #  time_major = True: [max_time, batch_size, depth]
            encoder_outputs, encoder_state_fw, encoder_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                fw_cells, bw_cells, self.encoder_emb_inp,
                sequence_length=self.X_len, time_major=True, dtype=tf.float32)

            self.encoder_output = tf.concat(encoder_outputs, 2) #Tensor("encoder/concat:0", shape=(50, ?, 300), dtype=float32)
            encoder_state_c = tf.concat((encoder_state_fw[0].c, encoder_state_bw[0].c), 1)
            encoder_state_h = tf.concat((encoder_state_fw[0].h, encoder_state_bw[0].h), 1)
            self.encoder_state = rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

        with tf.name_scope("decoder"), tf.variable_scope("decoder") as decoder_scope:
            decoder_cell = self.cell(self.num_hidden * 2)

            if not forward_only: # train
                attention_states = tf.transpose(self.encoder_output, [1, 0, 2])
                # attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                #     self.num_hidden * 2, attention_states, memory_sequence_length=self.X_len, normalize=True)
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    self.num_hidden * 2, attention_states, memory_sequence_length=self.X_len, scale=True)
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                                   attention_layer_size=self.num_hidden * 2)
                # copy the encoder state for decoder state initialization
                initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
                initial_state = initial_state.clone(cell_state=self.encoder_state)
                helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_emb_inp, self.decoder_len, time_major=True)
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True, scope=decoder_scope)
                self.decoder_output = outputs.rnn_output
                self.logits = tf.transpose(
                    self.projection_layer(self.decoder_output), perm=[1, 0, 2])
                self.logits_reshape = tf.concat(
                    [self.logits, tf.zeros([self.batch_size, summary_max_len - tf.shape(self.logits)[1], self.vocabulary_size_de])], axis=1)
                print("logits: ",self.logits)
                print(self.logits_reshape)

            else: # inference
                tiled_encoder_output = tf.contrib.seq2seq.tile_batch(
                    tf.transpose(self.encoder_output, perm=[1, 0, 2]), multiplier=self.beam_width)
                tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(self.encoder_state, multiplier=self.beam_width)
                tiled_seq_len = tf.contrib.seq2seq.tile_batch(self.X_len, multiplier=self.beam_width)
                #attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                #    self.num_hidden * 2, tiled_encoder_output, memory_sequence_length=tiled_seq_len, normalize=True)
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                     self.num_hidden * 2, tiled_encoder_output, memory_sequence_length=tiled_seq_len, scale=True)

 
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                                   attention_layer_size=self.num_hidden * 2)
                initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size * self.beam_width)
                initial_state = initial_state.clone(cell_state=tiled_encoder_final_state)
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=decoder_cell,
                    embedding=self.embeddings_de,
                    start_tokens=tf.fill([self.batch_size], tf.constant(2)),
                    end_token=tf.constant(3),
                    initial_state=initial_state,
                    beam_width=self.beam_width,
                    output_layer=self.projection_layer
                )
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, output_time_major=True, maximum_iterations=summary_max_len, scope=decoder_scope)
                self.prediction = tf.transpose(outputs.predicted_ids, perm=[1, 2, 0])

        with tf.name_scope("loss"):
            if not forward_only:
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits_reshape, labels=self.decoder_target)
                weights = tf.sequence_mask(self.decoder_len, summary_max_len, dtype=tf.float32)
                self.loss = tf.reduce_sum(crossent * weights / tf.to_float(self.batch_size))
                self.cost = tf.contrib.seq2seq.sequence_loss(
                                self.logits_reshape,
                                self.decoder_target,
                                weights)

                params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.update = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)


# Part 3: train function
def train():
    print("Building dictionary...")
    word_dict_en, word_dict_de, reversed_dict_en, reversed_dict_de, en_max_len, de_max_len = build_dict("train", False)
    print("Loading training dataset...")
    train_x, train_y = build_dataset("train", word_dict_en, word_dict_de, en_max_len, de_max_len, False)
    print("the numbe of training set: ", len(train_x), len(train_y))
    if not os.path.exists("saved_model"):
        os.mkdir("saved_model")
    with tf.Session() as sess:
        model = Model(reversed_dict_en, reversed_dict_de, en_max_len, de_max_len, args)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        summary_writer = tf.summary.FileWriter('tensorboard/', sess.graph)
        summary_writer.close()

        batches = batch_iter(train_x, train_y, args.batch_size, args.num_epochs)
        num_batches_per_epoch = (len(train_x) - 1) // args.batch_size + 1

        print("\nIteration starts.")
        print("Number of batches per epoch :", num_batches_per_epoch)

        for batch_num, (batch_x, batch_y) in enumerate(batches):
            batch_x_len = list(map(lambda x: len([y for y in x if y != 0]), batch_x))
            batch_decoder_input = list(map(lambda x: [word_dict_de["<s>"]] + list(x), batch_y))
            batch_decoder_len = list(map(lambda x: len([y for y in x if y != 0]), batch_decoder_input))
            batch_decoder_output = list(map(lambda x: list(x) + [word_dict_de["</s>"]], batch_y))

            batch_decoder_input = list(
                map(lambda d: d + (de_max_len - len(d)) * [word_dict_de["<padding>"]], batch_decoder_input))
            batch_decoder_output = list(
                map(lambda d: d + (de_max_len - len(d)) * [word_dict_de["<padding>"]], batch_decoder_output))

            train_feed_dict = {
                model.batch_size: len(batch_x),
                model.X: batch_x,
                model.X_len: batch_x_len,
                model.decoder_input: batch_decoder_input,
                model.decoder_len: batch_decoder_len,
                model.decoder_target: batch_decoder_output
            }

            _, step, loss, cost, pred = sess.run([model.update, model.global_step,
                                                  model.loss, model.cost, model.logits],
                                                  feed_dict=train_feed_dict)
            if step % 100 == 0:
                print("step: {}, batch: {}, loss: {}".format(step, batch_num % num_batches_per_epoch, cost))

            if step % num_batches_per_epoch == 0:
                hours, rem = divmod(time.perf_counter() - start, 3600)
                minutes, seconds = divmod(rem, 60)
                saver.save(sess, "./saved_model/model.ckpt", global_step=step)
                print(" Epoch {0}: Model is saved.".format(step // num_batches_per_epoch),
                "Elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds) , "\n")


# Part 4: test
def read_text(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        return [clean_str(x.strip()).split(' ') for x in f.readlines()]

def test():
    print("Loading dictionary...")
    word_dict_en, word_dict_de, reversed_dict_en, reversed_dict_de, en_max_len, de_max_len = build_dict("valid", False)
    print("Loading validation dataset...")
    valid_x, valid_y = build_dataset("valid", word_dict_en, word_dict_de, en_max_len, de_max_len, False)
    valid_x_len = [len([y for y in x if y != 0]) for x in valid_x]

    with tf.Session() as sess:
        print("Loading saved model...")
        model = Model(reversed_dict_en, reversed_dict_de, en_max_len, de_max_len, args, forward_only=True)
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state("./model/")
        saver.restore(sess, ckpt.model_checkpoint_path)

        batches = batch_iter(valid_x, valid_y, args.batch_size, 1)

        print("Writing summaries to 'result.txt'...")
        if os.path.exists("result.txt"):
            os.remove("result.txt")
        for batc_num, (batch_x, batch_y) in enumerate(batches):
            batch_x_len = [len([y for y in x if y != 0]) for x in batch_x]

            valid_feed_dict = {
                model.batch_size: len(batch_x),
                model.X: batch_x,
                model.X_len: batch_x_len,
            }

            prediction = sess.run(model.prediction, feed_dict=valid_feed_dict)
            prediction_output = [[reversed_dict_de[y] for y in x] for x in prediction[:, 0, :]]
            print("batch num: ", batc_num)
            with open("result.txt", "a") as f:
                for line in prediction_output:
                    summary = list()
                    for word in line:
                        if word == "</s>":
                            break
                        summary.append(word)
                        #if word not in summary:
                        #    summary.append(word)
                    print(" ".join(summary), file=f)
        print('Summaries are saved to "result.txt"...')

        print("calculate bleu score: ")        
        true_text = read_text(valid_de_path)
        predict_text = read_text("result.txt")
        print(len(true_text), len(predict_text))
        bleu_2 = 0
        cc = nltk.translate.bleu_score.SmoothingFunction()
        for i in range(len(true_text)):
            current_bleu = nltk.translate.bleu_score.sentence_bleu([true_text[i]], predict_text[i], smoothing_function=cc.method1)
            bleu_2 += current_bleu

        print("bleu: ", bleu_2/len(true_text))

# part5: traslate
def translate():
    print("Loading dictionary...")
    word_dict_en, word_dict_de, reversed_dict_en, reversed_dict_de, en_max_len, de_max_len = build_dict("valid", False)
    with tf.Session() as sess:
        print("Loading saved model...")
        model = Model(reversed_dict_en, reversed_dict_de, en_max_len, de_max_len, args, forward_only=True)
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state("./model/")
        saver.restore(sess, ckpt.model_checkpoint_path)

        while True:
            input_sent = input("input sentence: ")
            if input_sent == "":
                print("please input sentence in English")
                continue
            # English
            input_sent = word_tokenize(input_sent)
            input_sent = [word_dict_en.get(w, word_dict_en["<unk>"]) for w in input_sent]  # if not existed, return word_dict_en["<unk>"]
            input_sent = input_sent[:en_max_len]
            input_sent = [input_sent + (en_max_len - len(input_sent)) * [word_dict_en["<padding>"]]]
            input_len = [len([y for y in x if y != 0]) for x in input_sent]
            feed_dict = {
                model.batch_size: 1,
                model.X: input_sent,
                model.X_len: input_len,
            }
            prediction = sess.run(model.prediction, feed_dict=feed_dict)
            prediction_output = [[reversed_dict_de[y] for y in x] for x in prediction[:, 0, :]]
            print("translated: ")
            summary = list()
            for word in prediction_output[0]:
                if word == "</s>":
                    break
                summary.append(word)
            print(" ".join(summary))

if __name__ == '__main__':
    print("start")
    #translate()
    #test()
    if len(sys.argv) == 1:
        print("please specify the parameter: train or test or translate")
        print("for training, you should type: python3 NMT.py train")
        print("for testing, you should type: python3 NMT.py test")
        print("for translating, you should type: python3 NMT.py translate")
    else:
        if sys.argv[1] == "train":
            train()
        elif sys.argv[1] == "test":
            test()
        elif sys.argv[1] == "translate":
            translate()     
        else:
            print("please specify the parameter: train or test or translate")
            print("for training, you should type: python3 NMT.py train")
            print("for testing, you should type: python3 NMT.py test")
            print("for translating, you should type: python3 NMT.py translate")

 
