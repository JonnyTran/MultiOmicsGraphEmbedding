import os
import time

import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import Input, Conv2D, Dropout, MaxPooling1D, Lambda, Embedding, Bidirectional, LSTM, Convolution1D
from keras.models import Model
from keras.regularizers import l2
from keras.utils import multi_gpu_model
# from kegra.layers.graph import GraphConvolution
from spektral.layers import GraphConv
from tensorflow.keras import backend as K

from moge.evaluation.metrics import f1
from .static_graph_embedding import NeuralGraphEmbedding


class GCNEmbedding(NeuralGraphEmbedding):
    def __init__(self, d: int, batch_size: int, vocabulary_size, word_embedding_size, y_label: str, n_classes: int):
        self.y_label = y_label
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        self.word_embedding_size = word_embedding_size
        self.n_classes = n_classes
        super(GCNEmbedding, self).__init__(d, method_name="GCN_embedding")

    def create_network(self):
        input = Input(batch_shape=(None, None))  # (batch_number, sequence_length)
        x = Embedding(input_dim=self.vocabulary_size,
                      output_dim=self.word_embedding_size,
                      input_length=None, mask_zero=True, trainable=True)(input)  # (batch_number, sequence_length, 5)
        print("Embedding", x)

        x = Lambda(lambda y: K.expand_dims(y, axis=2), name="lstm_lambda_1")(x)  # (batch_number, sequence_length, 1, 5)
        x = Conv2D(filters=320, kernel_size=(26, 1), activation='relu',
                   data_format="channels_last", name="lstm_conv_1")(x)  # (batch_number, sequence_length-5, 1, 192)
        x = Lambda(lambda y: K.squeeze(y, axis=2), name="lstm_lambda_2")(x)  # (batch_number, sequence_length-5, 192)
        print("conv2D", x)
        #     x = BatchNormalization(center=True, scale=True, name="conv1_batch_norm")(x)
        x = MaxPooling1D(pool_size=13, padding="same")(x)
        x = Dropout(0.2)(x)

        x = Convolution1D(filters=192, kernel_size=6, activation='relu', name="lstm_conv_2")(x)
        print("conv1d_2", x)
        #     x = BatchNormalization(center=True, scale=True, name="conv2_batch_norm")(x)
        x = MaxPooling1D(pool_size=3, padding="same")(x)
        print("max pooling_2", x)
        x = Dropout(0.2)(x)

        x = Bidirectional(LSTM(160, return_sequences=False, return_state=False))(x)  # (batch_number, 320+320)
        print("brnn", x)
        #     x = Dropout(0.2)(x)
        #     x = Dense(_d, activation='linear', name="embedding_output")(x)
        #     x = BatchNormalization(center=True, scale=True, name="embedding_output_normalized")(x)
        print("embedding", x)
        return Model(input, x, name="encoder_network")

    def build_keras_model(self, multi_gpu=False):
        K.clear_session()

        with tf.device("/cpu:0" if multi_gpu else "/gpu:0"):
            input_seqs = Input(batch_shape=(None, None), dtype=tf.int8, name="input_seqs")
            subnetwork = Input(batch_shape=(None, None), sparse=False, dtype=tf.float32, name="subnetwork")
            print("subnetwork", subnetwork)

            # build create_network to use in each siamese 'leg'
            encoder_net = self.create_network()

            # encode each of the inputs into a list of embedding vectors with the conv_lstm_network
            x = encoder_net(input_seqs)
            print("embeddings", x)

            #     x = Dropout(0.5)(x)
            x = GraphConv(128, name="embedding_output",
                          activation='relu',
                          kernel_regularizer=l2(5e-4),
                          use_bias=True)([x, subnetwork])
            x = Dropout(0.5)(x)

            y_pred = GraphConv(self.n_classes, support=1,
                               activation='sigmoid',
                               use_bias=True)([x, subnetwork])

            self.model = Model(inputs=[input_seqs, subnetwork],
                               outputs=y_pred)

        # Multi-gpu parallelization
        if multi_gpu:
            self.model = multi_gpu_model(self.model, gpus=4, cpu_merge=True, cpu_relocation=True)

        # Compile & train
        self.model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["top_k_categorical_accuracy", f1
                     ],
        )
        print("Network total weights:", self.model.count_params())

    def learn_embedding(self, generator_train, generator_test, tensorboard=True, histogram_freq=0,
                        embeddings=False, early_stopping=False,
                        epochs=50, validation_steps=None,
                        seed=0, **kwargs):
        self.generator_train = generator_train
        self.generator_test = generator_test
        try:
            self.hist = self.model.fit_generator(generator_train, epochs=epochs, shuffle=False,
                                                 validation_data=generator_test,
                                                 validation_steps=validation_steps,
                                                 callbacks=self.get_callbacks(early_stopping, tensorboard,
                                                                              histogram_freq, embeddings),
                                                 use_multiprocessing=True, workers=8, verbose=2, **kwargs)
        except KeyboardInterrupt:
            print("Stop training")

    def get_callbacks(self, early_stopping=0, tensorboard=True, histogram_freq=0, embeddings=False, write_grads=False):
        callbacks = []
        if tensorboard:
            if not hasattr(self, "tensorboard"):
                self.build_tensorboard(histogram_freq=histogram_freq, embeddings=embeddings, write_grads=write_grads)
            callbacks.append(self.tensorboard)

        if early_stopping > 0:
            if not hasattr(self, "early_stopping"):
                self.early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=early_stopping, verbose=0,
                                                    mode='auto',
                                                    baseline=None, restore_best_weights=False)
            callbacks.append(self.early_stopping)

        if len(callbacks) == 0: callbacks = None
        return callbacks

    def build_tensorboard(self, histogram_freq, embeddings: bool, write_grads):
        if not hasattr(self, "log_dir"):
            self.log_dir = "logs/{}_{}".format(type(self).__name__[0:20], time.strftime('%m-%d_%H-%M%p').strip(" "))
            print("log_dir:", self.log_dir)

        if embeddings:
            x_test, node_labels = self.generator_test.load_data(return_node_names=True, y_label=self.y_label)
            if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)
            node_labels.to_csv(os.path.join(self.log_dir, "metadata.tsv"), sep="\t")

        self.tensorboard = TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=histogram_freq,
            write_grads=write_grads, write_graph=False, write_images=False,
            batch_size=self.batch_size,
            update_freq="epoch",
            embeddings_freq=1 if embeddings else 0,
            embeddings_metadata=os.path.join(self.log_dir, "metadata.tsv") if embeddings else None,
            embeddings_data=x_test if embeddings else None,
            embeddings_layer_names=["embedding_output"] if embeddings else None,
        )
        # Add params text to tensorboard
