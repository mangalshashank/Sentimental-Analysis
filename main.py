import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

df = pd.read_csv("/home/201112079/AI-Sem-6-main/data.csv")

texts = df['text'].values
labels = df['label'].values

labels = np.array(labels)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 64),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(padded_sequences, labels, epochs=10, batch_size=1)

# save the model
model.save("my_model.h5")

# evaluate model on new data
new_text = ["This product is amazing! It exceeded my expectations."]
new_sequences = tokenizer.texts_to_sequences(new_text)
new_padded_sequences = pad_sequences(new_sequences, maxlen=padded_sequences.shape[1])

loaded_model = tf.keras.models.load_model("my_model.h5")
prediction = loaded_model.predict(new_padded_sequences)[0][0]

if prediction > 0.5:
    print("Positive sentiment")
else:
    print("Negative sentiment")
