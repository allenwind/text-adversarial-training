import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn import metrics

from adversarial_trainer import AdversarialTrainer
from pooling import MaskGlobalMaxPooling1D
from pooling import MaskGlobalAveragePooling1D
from dataset import SimpleTokenizer, find_best_maxlen
from dataset import load_THUCNews_title_label
from dataset import load_weibo_senti_100k
from dataset import load_simplifyweibo_4_moods
from dataset import load_hotel_comment

# baseline 0.880625
# adversarial 0.892525

X, y, classes = load_hotel_comment()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=7384672)

num_classes = len(classes)
tokenizer = SimpleTokenizer()
tokenizer.fit(X_train)

# maxlen = find_best_maxlen(X_train)
maxlen = 256

def create_dataset(X, y, maxlen=maxlen):
    X = tokenizer.transform(X)
    X = sequence.pad_sequences(
        X, 
        maxlen=maxlen,
        dtype="int32",
        padding="post",
        truncating="post",
        value=0.0
    )
    y = tf.keras.utils.to_categorical(y)
    return X, y

X_train, y_train = create_dataset(X_train, y_train)

adversarial = True # 是否开启对抗训练
num_words = len(tokenizer)
embedding_dims = 128

inputs = Input(shape=(maxlen,))
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs)
x = Embedding(num_words, embedding_dims,
    embeddings_initializer="normal",
    input_length=maxlen,
    mask_zero=True,
    name="embedding")(inputs)
x = Dropout(0.2)(x)
x = Conv1D(filters=128,
           kernel_size=3,
           padding="same",
           activation="relu",
           strides=1)(x)
x, _ = MaskGlobalMaxPooling1D()(x, mask=mask)
x = Dense(128)(x)
x = Dropout(0.2)(x)
x = Activation("relu")(x)
outputs = Dense(num_classes, activation="softmax")(x)

if adversarial:
    model = AdversarialTrainer(inputs, outputs)
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"],
                  epsilon=0.5)
else:
    model = Model(inputs, outputs)
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
model.summary()

batch_size = 32
epochs = 10
callbacks = []
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callbacks,
          validation_split=0.1
)
X_test, y_test = create_dataset(X_test, y_test)
model.evaluate(X_test, y_test)
