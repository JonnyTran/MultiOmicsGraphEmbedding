from kegra.layers.graph import GraphConvolution
from keras.layers import LSTM
from keras.regularizers import l2

from .siamese_triplet_online_embedding import *


class GCNEmbedding(SiameseOnlineTripletGraphEmbedding):
    def __init__(self, d=128, num_labels=None, batch_size=256, lr=0.001, epochs=10, directed_proba=0.5, weighted=True,
                 compression_func="sqrt", negative_sampling_ratio=2.0, max_length=1400, truncating="post", seed=0,
                 verbose=False, conv1_kernel_size=12, conv1_batch_norm=False, max1_pool_size=6, conv2_kernel_size=6,
                 conv2_batch_norm=True, max2_pool_size=3, lstm_unit_size=320, dense1_unit_size=1024,
                 dense2_unit_size=512, directed_distance="euclidean", undirected_distance="euclidean",
                 source_target_dense_layers=True, embedding_normalization=False, **kwargs):
        self.num_labels = num_labels
        super(GCNEmbedding, self).__init__(d, None, batch_size, lr, epochs, directed_proba, weighted,
                                           compression_func,
                                           negative_sampling_ratio, max_length, truncating, seed, verbose,
                                           conv1_kernel_size,
                                           conv1_batch_norm, max1_pool_size, conv2_kernel_size, conv2_batch_norm,
                                           max2_pool_size,
                                           lstm_unit_size, dense1_unit_size, dense2_unit_size, directed_distance,
                                           undirected_distance,
                                           source_target_dense_layers, embedding_normalization, **kwargs)

    def create_lstm_network(self):
        input = Input(batch_shape=(None, None))  # (batch_number, sequence_length)
        x = Embedding(5, 4, input_length=None, mask_zero=True, trainable=True)(
            input)  # (batch_number, sequence_length, 5)
        print("Embedding", x)

        x = Lambda(lambda y: K.expand_dims(y, axis=2), name="lstm_lambda_1")(x)  # (batch_number, sequence_length, 1, 5)
        x = Conv2D(filters=320, kernel_size=(26, 1), activation='relu',
                   data_format="channels_last", name="lstm_conv_1")(x)  # (batch_number, sequence_length-5, 1, 192)
        x = Lambda(lambda y: K.squeeze(y, axis=2), name="lstm_lambda_2")(x)  # (batch_number, sequence_length-5, 192)
        print("conv2D", x)
        #     x = BatchNormalization(center=True, scale=True, name="conv1_batch_norm")(x)
        x = MaxPooling1D(pool_size=13, padding="same")(x)
        x = Dropout(0.2)(x)

        #     x = Convolution1D(filters=192, kernel_size=6, activation='relu', name="lstm_conv_2")(x)
        #     print("conv1d_2", x)
        # #     x = BatchNormalization(center=True, scale=True, name="conv2_batch_norm")(x)
        #     x = MaxPooling1D(pool_size=3, padding="same")(x)
        #     print("max pooling_2", x)
        #     x = Dropout(0.5)(x)

        x = Bidirectional(LSTM(64, return_sequences=False, return_state=False))(x)  # (batch_number, 320+320)
        print("brnn", x)
        x = Dropout(0.5)(x)

        x = Dense(self._d, activation='linear', name="embedding_output")(x)
        #     x = BatchNormalization(center=True, scale=True, name="embedding_output_normalized")(x)

        print("embedding", x)
        return Model(input, x, name="lstm_network")

    def build_keras_model(self, multi_gpu=False):
        K.clear_session()
        if multi_gpu:
            device = "/cpu:0"
            allow_soft_placement = True
        else:
            device = "/gpu:0"
            allow_soft_placement = False

        with tf.device(device):
            input_seqs = Input(batch_shape=(None, None), dtype=tf.int8, name="input_seqs")
            labels_directed = Input(batch_shape=(None, None), sparse=False, dtype=tf.float32,
                                    name="labels_directed")
            chromosome_name = Input(batch_shape=(None, 323), sparse=False, dtype=tf.uint8,
                                    name="chromosome_name")
            transcript_start = Input(batch_shape=(None, 1), sparse=False, dtype=tf.float32,
                                     name="transcript_start")
            transcript_end = Input(batch_shape=(None, 1), sparse=False, dtype=tf.float32,
                                   name="transcript_end")
            print("labels_directed", labels_directed)

            # build create_lstm_network to use in each siamese 'leg'
            lstm_network = self.create_lstm_network()

            # encode each of the inputs into a list of embedding vectors with the conv_lstm_network
            x = lstm_network(input_seqs)
            print("embeddings", x)
            support = 1
            #     x = Dropout(0.5)(x)
            x = GraphConvolution(128, support,
                                 activation='relu',
                                 kernel_regularizer=l2(5e-4),
                                 use_bias=False)([x, labels_directed])
            x = Dropout(0.5)(x)

            x = GraphConvolution(64, support,
                                 activation='relu',
                                 use_bias=False)([x, labels_directed])
            x = Dropout(0.5)(x)

            y_pred = Dense(self.num_labels,
                           activation='sigmoid',
                           #                    kernel_regularizer=l1()
                           )(x)

            siamese_net = Model(inputs=[input_seqs, labels_directed, chromosome_name, transcript_start, transcript_end],
                                outputs=y_pred)

        # Multi-gpu parallelization
        if multi_gpu:
            siamese_net = multi_gpu_model(siamese_net, gpus=4, cpu_merge=True, cpu_relocation=True)

        # Compile & train
        siamese_net.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["top_k_categorical_accuracy",
                     #              precision_m, recall_m
                     ],
        )
        print("Network total weights:", siamese_net.count_params())

    def learn_embedding(self, generator_train, generator_test, tensorboard=True, histogram_freq=0,
                        embeddings=False, early_stopping=False,
                        multi_gpu=False, subsample=True, n_steps=500, validation_steps=None,
                        edge_f=None, is_weighted=False, no_python=False, rebuild_model=False, seed=0,
                        **kwargs):
        try:
            self.hist = self.siamese_net.fit_generator(generator_train, epochs=self.epochs, shuffle=False,
                                                       validation_data=generator_test,
                                                       validation_steps=validation_steps,
                                                       callbacks=self.get_callbacks(early_stopping, tensorboard,
                                                                                    histogram_freq, embeddings),
                                                       use_multiprocessing=True, workers=8, **kwargs)
        except KeyboardInterrupt:
            print("Stop training")
        finally:
            self.save_network_weights()
