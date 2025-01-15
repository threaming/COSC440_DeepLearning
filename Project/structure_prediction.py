import os
# suppress silly log messages from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import random
import structure_prediction_utils as utils
from tensorflow import keras
import matplotlib as plt


class ProteinStructurePredictor1(keras.Model):
    def __init__(self):
        super().__init__()
        
        # Fully connected network for one-hot input
        self.fc1 = keras.layers.Dense(2048, activation='linear')  # First fully connected layer to match 64*64
        self.leaky_relu1 = keras.layers.LeakyReLU(alpha=0.01)     # Leaky ReLU
        self.fc2 = keras.layers.Dense(4096, activation='linear')   # Second fully connected layer
        self.leaky_relu2 = keras.layers.LeakyReLU(alpha=0.01)     # Leaky ReLU
        
        # Positional distance branch
        self.dist_max = keras.layers.MaxPooling2D((4, 4), padding='same')
        
        # Attention mechanism
        self.leaky_relu_att = keras.layers.LeakyReLU(alpha=0.01)

        # Decoding layers
        self.decoder_conv1 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')
        self.decoder_upsamp1 = keras.layers.UpSampling2D((2, 2))
        self.decoder_conv2 = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')
        self.decoder_upsamp2 = keras.layers.UpSampling2D((2, 2))
        self.decoder_conv3 = keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same')

    def call(self, inputs, mask=None):
        primary_one_hot = inputs['primary_one_hot']  # Shape: [batch_size, Nres, 21]

        # Precomputed positional distance information
        r = tf.range(0, utils.NUM_RESIDUES, dtype=tf.float32)
        distances = tf.abs(tf.expand_dims(r, -1) - tf.expand_dims(r, -2))
        distances_bc = tf.expand_dims(tf.broadcast_to(distances, [primary_one_hot.shape[0], utils.NUM_RESIDUES, utils.NUM_RESIDUES]), -1)
        distances_bc = distances_bc / utils.NUM_RESIDUES

        # Flatten the primary_onehot for the fully connected layers (Nres x 21 -> batch_size x (Nres * 21))
        primary_one_hot_flattened = tf.reshape(primary_one_hot, [primary_one_hot.shape[0], -1])  # Shape: [batch_size, Nres * 21]

        # Pass through the fully connected layers with Leaky ReLU activations
        x = self.fc1(primary_one_hot_flattened)
        x = self.leaky_relu1(x)
        x = self.fc2(x)
        one_hot_fc_out = self.leaky_relu2(x)
        
        # Reshape to a 2D format (64 x 64) for attention
        one_hot_fc_out = tf.reshape(one_hot_fc_out, [one_hot_fc_out.shape[0], 64, 64, 1])  # Shape: [batch_size, 64, 64]

        # Positional distance branch
        dist_out = self.dist_max(distances_bc)

        # Pixelwise attention
        attn_out = one_hot_fc_out * dist_out
        attn_out = self.leaky_relu_att(attn_out)

        # Decoding layers
        x = self.decoder_conv1(attn_out)
        x = self.decoder_upsamp1(x)
        x = self.decoder_conv2(x)
        x = self.decoder_upsamp2(x)
        decoded = self.decoder_conv3(x)

        return decoded


class ProteinStructurePredictor2(keras.Model):
    def __init__(self):
        super().__init__()
        
        # Fully connected network for one-hot input
        self.oh_conv1 = keras.layers.Conv1D(16, (3), activation='relu', padding='same')
        self.oh_max1 = keras.layers.MaxPooling1D(2, padding='same')
        self.oh_conv2 = keras.layers.Conv1D(16, (3), activation='relu', padding='same')
        self.oh_max2 = keras.layers.MaxPooling1D(2, padding='same')
        self.fc1 = keras.layers.Dense(4096, activation='linear')  # First fully connected layer to match 64*64
        self.leaky_relu1 = keras.layers.LeakyReLU(alpha=0.01)     # Leaky ReLU
        
        # Positional distance branch
        self.dist_max = keras.layers.MaxPooling2D((4, 4), padding='same')
        
        # Attention mechanism
        self.leaky_relu_att = keras.layers.LeakyReLU(alpha=0.01)

        # Decoding layers
        self.decoder_conv1 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')
        self.decoder_upsamp1 = keras.layers.UpSampling2D((2, 2))
        self.decoder_conv2 = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')
        self.decoder_upsamp2 = keras.layers.UpSampling2D((2, 2))
        self.decoder_conv3 = keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same')

    def call(self, inputs, mask=None):
        primary = inputs['primary']  # Shape: [batch_size, Nres]
        primary = primary/utils.NUM_AMINO_ACIDS
        primary = tf.expand_dims(primary, -1)  # Shape: [batch_size, Nres, 21, 1]


        # Precomputed positional distance information
        r = tf.range(0, utils.NUM_RESIDUES, dtype=tf.float32)
        distances = tf.abs(tf.expand_dims(r, -1) - tf.expand_dims(r, -2))
        distances_bc = tf.expand_dims(tf.broadcast_to(distances, [primary.shape[0], utils.NUM_RESIDUES, utils.NUM_RESIDUES]), -1)
        distances_bc = distances_bc / utils.NUM_RESIDUES

        # Pass through two convolutions and a fully connected layers
        x = self.oh_conv1(primary)
        x = self.oh_max1(x)
        x = self.oh_conv2(x)
        x = self.oh_max2(x)
        x = tf.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        one_hot_fc_out = x
        
        # Reshape to a 2D format (64 x 64) for attention
        one_hot_fc_out = tf.reshape(one_hot_fc_out, [one_hot_fc_out.shape[0], 64, 64, 1])  # Shape: [batch_size, 64, 64]

        # Positional distance branch
        dist_out = self.dist_max(distances_bc)

        # Pixelwise attention
        attn_out = one_hot_fc_out * dist_out
        attn_out = self.leaky_relu_att(attn_out)

        # Decoding layers
        x = self.decoder_conv1(attn_out)
        x = self.decoder_upsamp1(x)
        x = self.decoder_conv2(x)
        x = self.decoder_upsamp2(x)
        decoded = self.decoder_conv3(x)

        return decoded


def get_n_records(batch):
    return batch['primary_onehot'].shape[0]
def get_input_output_masks(batch):
    inputs = {'primary_one_hot':batch['primary_onehot'], 'primary':batch['primary']}
    outputs = batch['true_distances']
    masks = batch['distance_mask']

    return inputs, outputs, masks
def train(model, train_dataset, validate_dataset=None, train_loss=utils.mse_loss):
    '''
    Trains the model
    '''

    avg_loss = 0.
    avg_mse_loss = 0.
    mse_train_loss_rec = 0
    mse_val_loss_rec = 0
    rec_it = 0


    def print_loss():
        if validate_dataset is not None:
            validate_loss = 0.

            validate_batches = 0.
            for batch in validate_dataset.batch(1024):
                validate_inputs, validate_outputs, validate_masks = get_input_output_masks(batch)
                validate_preds = model.call(validate_inputs, validate_masks)

                validate_loss += tf.reduce_sum(utils.mse_loss(validate_preds, validate_outputs, validate_masks)) / get_n_records(batch)
                validate_batches += 1
            validate_loss /= validate_batches
        else:
            validate_loss = float('NaN')
        print(f'train loss {avg_loss:.3f} train mse loss {avg_mse_loss:.3f} validate mse loss {validate_loss:.3f}')
        return validate_loss
    
    first = True
    for batch in train_dataset:
        inputs, labels, masks = get_input_output_masks(batch)
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            outputs = model(inputs, masks)
            #model.summary()
            l = train_loss(outputs, labels, masks)
            batch_loss = tf.reduce_sum(l)
            gradients = tape.gradient(batch_loss, model.trainable_weights)
            avg_loss = batch_loss / get_n_records(batch)
            avg_mse_loss = tf.reduce_sum(utils.mse_loss(outputs, labels, masks)) / get_n_records(batch)
        
        model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        mse_train_loss_rec += avg_mse_loss
        mse_val_loss_rec += print_loss()
        rec_it += 1

        if first:
            print(model.summary())
            first = False

    return mse_train_loss_rec/rec_it, mse_val_loss_rec/rec_it

def test(model, test_records, viz=False):
    mse_loss_rec = np.array([])
    for batch in test_records.batch(1024):
        test_inputs, test_outputs, test_masks = get_input_output_masks(batch)
        test_preds = model.call(test_inputs, test_masks)
        test_loss = tf.reduce_sum(utils.mse_loss(test_preds, test_outputs, test_masks)) / get_n_records(batch)
        mse_loss_rec = np.append(mse_loss_rec, test_loss)
        print(f'test mse loss {test_loss:.3f}')

    if viz:
        print(model.summary())
        r = random.randint(0, test_preds.shape[0])
        utils.display_two_structures(test_preds[r], test_outputs[r], test_masks[r])

    return mse_loss_rec.sum()/mse_loss_rec.size

def main(data_folder, model_ver='model1'):
    training_records = utils.load_preprocessed_data(data_folder, 'training.tfr')
    validate_records = utils.load_preprocessed_data(data_folder, 'validation.tfr')
    test_records = utils.load_preprocessed_data(data_folder, 'testing.tfr')

    # Select Model
    if (model_ver == 'model1'):
      model = ProteinStructurePredictor1()
    elif (model_ver == 'model2'):
      model = ProteinStructurePredictor2()
    else:
      print("ERROR: Selected model not available")
      return 0

    model.optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    model.batch_size = 256
    epochs = 10
    print(f"Running {model_ver} with {model.batch_size} batch size and {epochs} epochs")
    # Iterate over epochs.
    train_loss = []
    val_loss = []
    test_loss = []
    for epoch in range(epochs):
        epoch_training_records = training_records.shuffle(buffer_size=256).batch(model.batch_size, drop_remainder=False)
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        train_loss_tmp, val_loss_tmp = train(model, epoch_training_records, validate_records)
        train_loss.append(train_loss_tmp)
        val_loss.append(val_loss_tmp)
        test_loss.append(test(model, test_records, True))
  
    # show losses
    utils.display_loss(train_loss, val_loss, test_loss)

    # save data
    if (model_ver == 'model1'):
      np.savez('model_1.npz', np.asarray(train_loss), np.asarray(val_loss), np.asarray(test_loss))
      #model.export(data_folder + 'model1')
    if (model_ver == 'model2'):
      np.savez('model_2.npz', np.asarray(train_loss), np.asarray(val_loss), np.asarray(test_loss))
      #model.export(data_folder + 'model2')


if __name__ == '__main__':
    local_home = os.path.expanduser("~")  # on my system this is /Users/jat171
    data_folder = local_home + '/uc/teaching/440/project/2024/data/'

    main(data_folder)