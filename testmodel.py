import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load the saved model
loaded_model = tf.keras.models.load_model("my_model.h5")

# load the test data from test.csv file
test_df = pd.read_csv("datatest.csv")

# get the texts and labels from the test data
texts = test_df['text'].values
labels = test_df['label'].values

# load the tokenizer and convert the input texts to sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# pad the sequences to ensure equal length
padded_sequences = pad_sequences(sequences, maxlen=loaded_model.input_shape[1])

# evaluate the model on the test data
eval_results = loaded_model.evaluate(padded_sequences, labels)

# make predictions on the test data
predictions = loaded_model.predict(padded_sequences)
y_pred = (predictions > 0.5).astype(int)

# compute the confusion matrix
cm = confusion_matrix(labels, y_pred)
print("Confusion matrix:")
print(cm)

# compute other evaluation metrics
report = classification_report(labels, y_pred)
print("Classification report:")
print(report)

# plot the accuracy and loss graphs
accuracy = eval_results[1]
loss = eval_results[0]

plt.plot([0, 1], [0, 1], 'k--')
plt.plot([accuracy], [loss], marker='o', markersize=5, color="red")
plt.xlabel('Accuracy')
plt.ylabel('Loss')
plt.title('Model Evaluation')
plt.show()
