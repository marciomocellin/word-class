
# def conectar_com_banco(usuario):
#    if usuario in 'fnord':
#        server = 'fnord' 
#        database = 'fnord'
#        username = 'fnord' 
#        password = 'fnord' 
#    elif usuario in 'fnord':
#        server = 'fnord' 
#        database = 'fnord'
#        username = 'fnord' 
#        password = 'fnord'
#    elif usuario in 'fnord':
#        server = 'fnord' 
#        database = 'fnord'
#        username = 'fnord' 
#        password = 'fnord' 
#    else:
#        print('funcao_nao_encontrado')
#    import pyodbc
#    cnxn = pyodbc.connect('DRIVER={fnord};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
#    return(cnxn)

# cursor=conectar_com_banco('fnord')

import pandas as pd
# ID_varredura = pd.read_sql("""select 
#                             from fnord..TB_fnord
#                             where SKU in ('
#                             """+"', '".join(SKU)+"')", cursor)

# read_csv('SKU.csv', sep= ';', encoding='latin-1', dtype = {'ST_SKU': str , 'PRODUTO': str })
# SKU=pd.read_csv('SKU.csv')
# SKU.example = SKU.example.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

import numpy as np

import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional
print(tf.__version__)

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('portuguese'))

vocab_size = 5000 
embedding_dim = 64
max_length = 20
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>' #OOV = Out of Vocabulary
training_portion = .8

articles = []
labels = []

with open("SKU.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[1])
        article = row[0]
        for word in STOPWORDS:
            token = ' ' + word + ' '
            article = article.replace(token, ' ')
            article = article.replace(' ', ' ')
        articles.append(article)

print(len(labels))
print(len(articles))

train_size = int(len(articles) * training_portion)

train_articles = articles[0: train_size]
train_labels = labels[0: train_size]

validation_articles = articles[train_size:]
validation_labels = labels[train_size:]

print("train_size",  train_size)
print(f"train_articles {len(train_articles)}")
print("train_labels", len(train_labels))
print("validation_articles", len(validation_articles))
print("validation_labels", len(validation_labels))

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_articles)

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print("len train_sequnces[0]: ", len(train_sequences[0]))
print("len train_padded[0]: ", len(train_padded[0]))

print("len train_sequences[1]: ", len(train_sequences[1]))
print("len train_padded[1]: ", len(train_padded[1]))

print("len train_sequences[10]: ", len(train_sequences[10]))
print("len train_padded[10]: ", len(train_padded[10]))

validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(len(validation_sequences))
print(validation_padded.shape)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
#columns.
training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))
index = pd.DataFrame.from_dict(label_tokenizer.word_index, orient='index').sort_values(by=[0]).index
print(label_tokenizer.word_index)

print(training_label_seq[0])
print(training_label_seq[1])
print(training_label_seq[2])
print(training_label_seq.shape)
print('-------------')
print(validation_label_seq[0])
print(validation_label_seq[1])
print(validation_label_seq[2])
print(validation_label_seq.shape)

model = Sequential()

model.add(Embedding(vocab_size, embedding_dim))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(embedding_dim)))
model.add(Dense(352, activation='softmax'))

model.summary()

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

num_epochs = 10
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)

import matplotlib.pyplot as plt

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

txt = ['ventilador de mesa']
# txt = ['cafeteira de mesa']
# txt = ['boneca bee hugs unicórnio 0882 bee toys']
#txt = ['cafeteira']
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_length)
pred = model.predict(padded)
#labels = ['sport', 'bussiness', 'politics', 'tech', 'entertainment'] #orig

#print(pred)
print(np.argmax(pred))
print(index[np.argmax(pred)-1])


