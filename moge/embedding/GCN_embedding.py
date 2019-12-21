from keras_gcn import GraphConv

from .siamese_triplet_online_embedding import *


class GCNEmbedding(SiameseOnlineTripletGraphEmbedding):
    def __init__(self, d=128, margin=0.2, batch_size=256, lr=0.001, epochs=10, directed_proba=0.5, weighted=True,
                 compression_func="sqrt", negative_sampling_ratio=2.0, max_length=1400, truncating="post", seed=0,
                 verbose=False, conv1_kernel_size=12, conv1_batch_norm=False, max1_pool_size=6, conv2_kernel_size=6,
                 conv2_batch_norm=True, max2_pool_size=3, lstm_unit_size=320, dense1_unit_size=1024,
                 dense2_unit_size=512, directed_distance="euclidean", undirected_distance="euclidean",
                 source_target_dense_layers=True, embedding_normalization=False, **kwargs):
        super(GCNEmbedding, self).__init__(d, margin, batch_size, lr, epochs, directed_proba, weighted,
                                           compression_func,
                                           negative_sampling_ratio, max_length, truncating, seed, verbose,
                                           conv1_kernel_size,
                                           conv1_batch_norm, max1_pool_size, conv2_kernel_size, conv2_batch_norm,
                                           max2_pool_size,
                                           lstm_unit_size, dense1_unit_size, dense2_unit_size, directed_distance,
                                           undirected_distance,
                                           source_target_dense_layers, embedding_normalization, **kwargs)

    def build_keras_model(self, multi_gpu=False):
        if multi_gpu:
            device = "/cpu:0"
        else:
            device = "/gpu:0"
        K.clear_session()
        tf.reset_default_graph()

        with tf.device(device):
            input_seqs = Input(batch_shape=(None, None), dtype=tf.int8, name="input_seqs")
            labels_directed = Input(batch_shape=(None, None), sparse=True, dtype=tf.float32,
                                    name="labels_directed")
            chromosome_name = Input(batch_shape=(None, None), sparse=True, dtype=tf.uint8,
                                    name="chromosome_name")
            transcript_start = Input(batch_shape=(None, None), sparse=True, dtype=tf.float32,
                                     name="transcript_start")
            transcript_end = Input(batch_shape=(None, None), sparse=True, dtype=tf.float32,
                                   name="transcript_end")
            print("labels_directed", labels_directed) if self.verbose else None

            # build create_lstm_network to use in each siamese 'leg'
            self.lstm_network = self.create_lstm_network()

            # encode each of the inputs into a list of embedding vectors with the conv_lstm_network
            embeddings = self.lstm_network(input_seqs)
            print("embeddings", embeddings) if self.verbose else None

            conv_layer = GraphConv(units=192, )([embeddings, labels_directed])

            self.siamese_net = Model(
                inputs=[input_seqs, labels_directed, chromosome_name, transcript_start, transcript_end],
                outputs=conv_layer)

            # Multi-gpu parallelization
            if multi_gpu:
                self.siamese_net = multi_gpu_model(self.siamese_net, gpus=4, cpu_merge=True, cpu_relocation=False)

            # Compile & train
            self.siamese_net.compile(
                loss=None,
                optimizer=Adadelta(),
                metrics=["accuracy"],
            )
            print("Network total weights:", self.siamese_net.count_params()) if self.verbose else None

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
