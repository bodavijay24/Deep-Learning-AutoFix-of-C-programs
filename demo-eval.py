import sys

if len(sys.argv) < 3:
    print("python3 demo-eval.py <input-csv-file> <output-csv-file>")
    exit(0)

ip = sys.argv[1]
op = sys.argv[2]


##required imports
import pandas as pd
import numpy as np
import copy

# importing tensorflow modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Vocabulary:
    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token
    OOV_token = 3

    def __init__(self):

        self.word2index = {"PAD": self.PAD_token, "SOS": self.SOS_token, "EOS": self.EOS_token, "OOV": self.OOV_token}
        self.word2count = {}
        self.index2word = {self.PAD_token: "PAD", self.SOS_token: "SOS", self.EOS_token: "EOS", self.OOV_token: "OOV"}
        self.num_words = 4
        self.num_toks = 0
        self.longest_tok = 0

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1

    def add_tokens(self, lst):
        tok_len = 0
        for tok in lst:
            tok_len += 1
            self.add_word(tok)
        if tok_len > self.longest_tok:
            # This is the longest sentence
            self.longest_tok = tok_len
        # Count the number of sentences
        self.num_toks += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]


# Extracting each tokens

def words(main_data):
    word = []
    lst = []
    for i in main_data:
        lst.append(eval(i))
        word += eval(i)

    return word, lst


# Restricting vocabulary

def restrict(vocab, word, n):
    freqs = {k: v for k, v in sorted(vocab.word2count.items(), key=lambda it: it[1], reverse=True)}
    rest = dict(list(freqs.items())[:n])
    recnt = list(freqs.keys())[:n]

    for i in range(len(word)):
        for j in range(len(word[i])):
            val = word[i][j]
            if (val not in rest):
                word[i][j] = "OOV"

    current_vocab = {token: index + 4 for index, token in enumerate(recnt)}
    current_vocab["PAD"] = 0
    current_vocab["SOS"] = 1
    current_vocab["EOS"] = 2
    current_vocab["OOV"] = 3

    vocab_inverse = {idx: w for w, idx in current_vocab.items()}

    return current_vocab, vocab_inverse


# Mapping indices to tokens

def generate(lst, vocab):
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            val = lst[i][j]
            #             print(val)
            lst[i][j] = vocab[val]
        lst[i] = [1] + lst[i] + [2]
    return lst


k_val = 1200
e_max_len = 60
d_max_len = 60

# path = "/content/drive/MyDrive/Colab Notebooks/ASE_data/"
path=""

data = pd.read_csv(path + "train.csv")
val_data = pd.read_csv(path + ip)

"""## Train Data"""

main_data = data.loc[:, ['sourceLineTokens', 'targetLineTokens']]

vocab = Vocabulary()
t_vocab = Vocabulary()

word, orig = words(main_data['sourceLineTokens'])
t_word, t_orig = words(main_data['targetLineTokens'])
vocab.add_tokens(word)
vocab.add_tokens(t_word)

# Source to k tokens
topk_tokens = copy.deepcopy(orig)
source_vocab, vocab_inverse = restrict(vocab, topk_tokens, k_val)
source = copy.deepcopy(topk_tokens)
source = generate(source, source_vocab)

# Target top k tokens
t_topk_tokens = copy.deepcopy(t_orig)
target_vocab, t_vocab_inverse = restrict(vocab, t_topk_tokens, k_val)

target = copy.deepcopy(t_topk_tokens)
target = generate(target, target_vocab)

## VALIDATION DATA(PREPROCESSING OF LOADED INPUT DATA)

mval_data = val_data.loc[:,['sourceLineTokens']]

v_word,v_orig = words(mval_data['sourceLineTokens'])
v_topk_tokens = copy.deepcopy(v_orig)
_,_ = restrict(vocab,v_topk_tokens,k_val)

valid_source = copy.deepcopy(v_topk_tokens)
valid_source =generate(valid_source,source_vocab)

# tv_word,tv_orig = words(mval_data['targetLineTokens'])
# tv_topk_tokens = copy.deepcopy(tv_orig)
#
# _,_ =restrict(vocab,tv_topk_tokens,k_val)
# val_target = copy.deepcopy(tv_topk_tokens)
#
# val_target = generate(val_target,source_vocab)



X_padded = pad_sequences(source,padding='post',maxlen=e_max_len)
y_padded = pad_sequences(target,padding='post',maxlen=d_max_len)




valid_X_padded = pad_sequences(valid_source,padding='post',maxlen=e_max_len)
# valid_y_padded = pad_sequences(val_target,padding='post',maxlen=d_max_len)

#2 dimensional encoder input data (validation)
encoder_val_data = valid_X_padded.copy().astype('float32')

num_encoder_tokens = len(source_vocab)
num_decoder_tokens = len(target_vocab)
mx_encoder_len  =  e_max_len
mx_decoder_len  =  d_max_len
latent_dim = 256
embedd_size = 64
batch_size = 64


model3 = tf.keras.models.load_model("/home/vijay/Desktop/AllAssgns/Assignment 2/saved_model")

# model3.layers

"""# Inference"""

inf_encoder_inputs = model3.input[0]  # input_1
inf_encoder_outputs, state_h_enc, state_c_enc = model3.layers[4].output  # lstm_1
inf_encoder_states = [state_h_enc, state_c_enc]
inf_encoder_model = keras.Model(inf_encoder_inputs, inf_encoder_states)
# inf_encoder_model

inf_decoder_inputs = model3.input[1]  # input_1
decoder_state_input_h = keras.Input(shape=(latent_dim,),name="ip1")
decoder_state_input_c = keras.Input(shape=(latent_dim,),name="ip2")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

inf_embedd = model3.layers[3](inf_decoder_inputs)
inf_decoder_outputs, i_state_h, i_state_c = model3.layers[5](inf_embedd, initial_state=decoder_states_inputs)
inf_decoder_states = [i_state_h, i_state_c]

inf_decoder_outputs = model3.layers[6](inf_decoder_outputs)

inf_decoder_model = keras.Model(
    [inf_decoder_inputs]+decoder_states_inputs,[inf_decoder_outputs] + inf_decoder_states)




def generate_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = inf_encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))

    target_seq[0, 0] = source_vocab["SOS"]

    decoded_tokens = []
    while True:
        output_tokens, h, c = inf_decoder_model.predict([target_seq] + states_value)

        predicted_index = np.argmax(output_tokens[-1, -1, :])
        predicted_token = vocab_inverse[predicted_index]
        if predicted_token == "EOS" or len(decoded_tokens) > e_max_len:
            return decoded_tokens
        decoded_tokens.append(predicted_token)
        # Update the target sequence (of length 1).

        target_seq[0, 0] = predicted_index

        # Update states
        states_value = [h, c]





#
# for seq_index in range(3):
#     # Take one sequence (part of the training set)
#     # for trying out decoding.
#     input_seq = encoder_input_data[seq_index : seq_index + 1]
#     predicted_output = generate_sequence(input_seq)
#     print("-")
#     print("Input sentence:", orig[seq_index])
#     print("Target sentence: ",t_orig[seq_index])
#     print("Decoded sentence:", predicted_output)

# from tqdm import tnrange



acc = 0
exported= {}
lst=[]
import pyprind
import sys

progress = pyprind.ProgBar(len(v_orig), stream=sys.stdout)

# for seq_index in tnrange(5,desc="Loop Status"):
for seq_index in range(len(v_orig)):

    input_seq = encoder_val_data[seq_index : seq_index + 1]
    predicted_output = generate_sequence(input_seq)
    # print("-",end=",")
    # print("Input sentence:", v_orig[seq_index])
    lst.append(predicted_output)
    # print("Decoded sentence:", predicted_output)
    progress.update()

exported["predicted"] = lst
exported["sourceLineTokens"] = v_orig


df = pd.DataFrame( { "sourceLineTokens":exported["sourceLineTokens"], "fixedTokens":exported["predicted"]} )
df.to_csv(path+op,index=False)

