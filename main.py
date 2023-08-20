import cv2
import matplotlib.pyplot as plt
import time
import os
import tensorflow as tf
import pickle
import numpy as np
from PIL import Image, ImageFile

with open('train_encoded_images_ResNet.p', 'rb') as f:
    enc_imgs = pickle.load(f, encoding="bytes")
with open('captions_all.pkl', 'rb') as f:
    captions_all = pickle.load(f)
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)
with open('embedding_matrix.pkl', 'rb') as f:
    embedding_matrix = pickle.load(f)

def get_captions(captions, imid, words):
  c = captions[imid]
  for i in range(len(c)):
      cap = c[i]
      # print(c)
      cap_word = [words[  j] for j in cap]
      print(cap_word)
  # return cap_word

class RnnDecoder(tf.keras.Model):
  def __init__(self, embedding_size, dict_size, units):
    super(RnnDecoder, self).__init__()
    self.feature_layer = tf.keras.layers.Dense(embedding_size, activation='relu')
    self.embed = tf.keras.layers.Embedding(dict_size, embedding_size, weights = [embedding_matrix])
    self.embed.trainable = False
    self.rnn = tf.keras.layers.GRU(units,return_state=True,  recurrent_initializer='glorot_uniform')
    self.hidden_layer = tf.keras.layers.Dense(units)
    self.output_layer = tf.keras.layers.Dense(dict_size)


  def call(self, features, labels):
    # labels = labels[:,-1] # remove the _END word since the first input to the lstm is the features
    embeddings = self.embed(labels)
    features1 = self.feature_layer(features)
    em  = tf.squeeze(embeddings)

    rnn_input = tf.concat([features1, em],axis = 1)
    # rnn_input = features1+ em
    rnn_input = tf.expand_dims(rnn_input,1)

    rnn_out, h = self.rnn(rnn_input)
    # print(rnn_out.shape)
    h1 = self.hidden_layer(rnn_out)
    y = self.output_layer(h1)
    return y, h
class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units
    # self.feature_layer = tf.keras.layers.Dense(embedding_dim, activation='relu')
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights = [embedding_matrix])
    # self.embedding.trainable = False
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)
    self.attention = BahdanauAttention(self.units)


  def call(self, x, features, hidden):
  # defining attention as a separate model
      context_vector, attention_weights = self.attention(features, hidden)
      # print(context_vector.shape)

      # x shape after passing through embedding == (batch_size, 1, embedding_dim)
      x = self.embedding(x)
      # print(x.shape)
      # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
      x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

      # passing the concatenated vector to the GRU
      output, state = self.gru(x)

      # shape == (batch_size, max_length, hidden_size)
      x = self.fc1(output)

      # x shape == (batch_size * max_length, hidden_size)
      x = tf.reshape(x, (-1, x.shape[2]))

      # output shape == (batch_size * max_length, vocab)
      x = self.fc2(x)

      return x, state, attention_weights


  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # attention_hidden_layer shape == (batch_size, 64, units)
    attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                         self.W2(hidden_with_time_axis)))

    # score shape == (batch_size, 64, 1)
    # This gives you an unnormalized score for each image feature.
    score = self.V(attention_hidden_layer)

    # attention_weights shape == (batch_size, 64, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

# @tf.function
def train_step(img_tensor, target):
  loss = 0

  # initializing the hidden state for each batch
  # because the captions are not related from image to image
  hidden = Rnn_model.reset_state(batch_size=target.shape[0])
  # print(target.shape)
  # dec_input = tf.expand_dims(1 * target.shape[0], 1)
  dec_input = tf.cast(np.ones((target.shape[0],1)), tf.float32)

  with tf.GradientTape() as tape:
      # features = encoder(img_tensor)

      for i in range(1, target.shape[1]):
          # passing the features through the decoder
          predictions, hidden,_ = Rnn_model(dec_input,img_tensor, hidden)

          loss += loss_function(target[:, i], predictions)

          # using teacher forcing
          dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))

  trainable_variables = Rnn_model.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)

  optimizer.apply_gradients(zip(gradients, trainable_variables))
  # print(loss)
  # print(total_loss)

  return loss, total_loss

def test_step(features):
  max_length = 17
  print(features.shape)
  # attention_plot = np.zeros((max_length, attention_features_shape))

  hidden = Rnn_model.reset_state(batch_size=1)


  dec_input = tf.cast(np.ones((features.shape[0],1)), tf.float32)
  result = []

  for i in range(max_length):
      predictions, hidden, attention_weights = Rnn_model(dec_input,
                                                        features,
                                                        hidden)

      # attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
      # p = tf.math.argmax(tf.nn.softmax(output),1).numpy()

      # pred_caption.append(p)

      predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
      # predicted_word = tf.compat.as_text(index_to_word(predicted_id).numpy())
      result.append(predicted_id)

      if predicted_id == 2:
          return result

      dec_input = tf.expand_dims([predicted_id], 0)

  # attention_plot = attention_plot[:len(result), :]
  return result
tf.keras.backend.clear_session()
units = 256
dict_size = len(words)
embed_size = 300
Rnn_model = RNN_Decoder(embed_size, units, dict_size) # attention
batch_size = 100
# batches = np.floor(len(enc_imgs) / batch_size)
# rand_ind = np.random.permutation(len(enc_imgs))
inp_arr = []
cap_arr = []
im_name = []
im_count = 0
for i in range(1, 30000 + 1):  # set to the total number of images here
    im_count = im_count + 1

    try:

        # im_name.append(im_count)
        for j in range(len(captions_all[i])):
            # print(im_count)
            inp_arr.append(enc_imgs[i])

            im_name.append(im_count)

            # print(i)
            cap_arr.append(captions_all[i][j])
    except:
        # print('NOT')
        blah = 4
im_name = np.asarray(im_name)
im_name = im_name.reshape(im_name.shape[0],-1)
# im_name.dtype
caps = tf.convert_to_tensor(cap_arr)
im_ids = tf.convert_to_tensor(im_name)
print(caps.shape)
print(im_ids.shape)
caps_ds = tf.data.Dataset.from_tensor_slices((im_name, cap_arr))
dataset = caps_ds
BATCH_SIZE = 488
# BUFFER_SIZE = 1000
BUFFER_SIZE = 1000


def map_func(img_name, cap):
    # print(int(img_name))
    # print(img_name.shape)
    # img_tensor = inp_arr[int(img_name)]
    img_tensor = enc_imgs[int(img_name)]
    return img_tensor, cap


dataset = dataset.map(lambda item1, item2: tf.numpy_function(
    map_func, [item1, item2], [tf.float32, tf.int32]),
                      num_parallel_calls=tf.data.AUTOTUNE)

train_ds1 = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_ds1 = train_ds1.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
cardinality = tf.data.experimental.cardinality(train_ds1)
cardinality

start_epoch = 0
# import wandb
# wandb.login()
loss_plot = []

EPOCHS = 5

wandb.init(
    project="GRU",
    # Set entity to specify your username or team name
    # ex: entity="wandb",
    config={
        "optimizer": "ADAM",
        "loss": "sparse_categorical_crossentropy",
        "metric": "loss",
        "epoch": EPOCHS,
        "batch_size": BATCH_SIZE
    })
config = wandb.config

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(train_ds1):

        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss
        wandb.log({"batch loss": batch_loss.numpy() / int(target.shape[1])})
        if batch % 50 == 0:
            average_batch_loss = batch_loss.numpy() / int(target.shape[1])
            print(f'Epoch {epoch + 1} Batch {batch} Loss {average_batch_loss:.4f}')

    # storing the epoch end loss value to plot later
    ep_loss = total_loss / (batch)
    loss_plot.append(total_loss / (batch))
    wandb.log({"loss": total_loss / (batch)})
    # if epoch % 5 == 0:
    #   ckpt_manager.save()
    print(img_tensor.shape)
    print(batch)
    print(f'Epoch {epoch + 1} Loss {total_loss / (batch):.6f}')
    print(f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')