mport numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
with open('pizza.txt', 'r', encoding='utf-8') as file:
    text = file.read()
token = Tokenizer()
token.fit_on_texts([text])
total_worlds = len(token.word_index) + 1
input_squences = []
for l in text.split("\n"): 
    tokenlst = token.texts_to_sequences([l])[0]
    for i in range(1,len(tokenlst)):
        ngram = tokenlst[:1+i]
        input_squences.append(ngram)
setence_token = input_squences[10] 
sentence = []
for t in setence_token:
    sentence.append(list((token.word_index).keys())[list((token.word_index).values()).index(t)])
print(sentence)
max_sequence_len = max([len(seq) for seq in input_squences])
input_squences = np.array(pad_sequences(input_squences, maxlen=max_sequence_len, padding='pre'))
X = input_squences[:, :-1]
y = input_squences[:, -1]
y = np.array(tf.keras.utils.to_categorical(y, num_classes=total_worlds)) 
model = Sequential()
model.add(Embedding(total_worlds, 100, input_length=max_sequence_len-1))
model.add(LSTM(150))
model.add(Dense(total_worlds, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=1)
seed_text = "Hello There"
next_words = 5

for _ in range(next_words):
    token_list = token.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)
    output_word = ""
    for word, index in token.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word