import os
import time

import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import Input, Conv2D, Dropout, MaxPooling1D, Lambda, Embedding, Bidirectional, LSTM, Convolution1D, \
    BatchNormalization, Dense
from keras.models import Model
from keras.regularizers import l2
from keras.utils import multi_gpu_model
# from kegra.layers.graph import GraphConvolution
# from spektral.layers import GraphConv
from keras_gat import GraphAttention
from keras_transformer.extras import ReusableEmbedding
from keras_transformer.position import TransformerCoordinateEmbedding
from keras_transformer.transformer import TransformerBlock
from tensorflow.keras import backend as K

from moge.evaluation.metrics import f1
from .static_graph_embedding import NeuralGraphEmbedding


class GCNEmbedding(NeuralGraphEmbedding):
    def __init__(self, d: int, attn_heads: int, batch_size: int, vocabulary_size: int, word_embedding_size: int,
                 max_length: int, y_label: str, n_classes: int, multi_gpu=False):
        self.y_label = y_label
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        self.word_embedding_size = word_embedding_size

        self.max_length = max_length

        self.num_heads = attn_heads
        super(GCNEmbedding, self).__init__(d, method_name="GCN_embedding")
        self.build_keras_model(multi_gpu)

    def create_encoder_network(self, batch_norm=True):
        input_seqs = Input(shape=(None,), name="input_seqs")  # (batch_number, sequence_length)
        x = Embedding(input_dim=self.vocabulary_size,
                      output_dim=self.vocabulary_size - 1,
                      input_length=None, mask_zero=True, trainable=True)(
            input_seqs)  # (batch_number, sequence_length, 5)
        print("Embedding", x)

        x = Lambda(lambda y: K.expand_dims(y, axis=2), name="lstm_lambda_1")(x)  # (batch_number, sequence_length, 1, 5)
        x = Conv2D(filters=320, kernel_size=(26, 1), activation='relu',
                   data_format="channels_last", name="lstm_conv_1")(x)  # (batch_number, sequence_length-5, 1, 192)
        x = Lambda(lambda y: K.squeeze(y, axis=2), name="lstm_lambda_2")(x)  # (batch_number, sequence_length-5, 192)
        print("conv2D", x)
        if batch_norm:
            x = BatchNormalization(center=True, scale=True, name="conv1_batch_norm")(x)
        x = MaxPooling1D(pool_size=13, padding="same")(x)
        x = Dropout(0.2)(x)

        x = Convolution1D(filters=192, kernel_size=6, activation='relu', name="lstm_conv_2")(x)
        print("conv1d_2", x)
        if batch_norm:
            x = BatchNormalization(center=True, scale=True, name="conv2_batch_norm")(x)
        x = MaxPooling1D(pool_size=3, padding="same")(x)
        print("max pooling_2", x)
        x = Dropout(0.2)(x)

        x = Bidirectional(LSTM(320, return_sequences=False, return_state=False),
                          merge_mode='concat')(x)  # (batch_number, 320+320)
        print("brnn", x)
        x = Dropout(0.2)(x)

        x = Dense(self._d, activation='linear', name="encoder_output")(x)
        if batch_norm:
            x = BatchNormalization(center=True, scale=True, name="encoder_output_normalized")(x)

        print("embedding", x)
        return Model(input_seqs, x, name="encoder_model")

    def create_transformer_net(self, batch_norm=True, transformer_depth=1):
        input_seqs = Input(shape=(None,), name="input_seqs")  # (batch_number, sequence_length)
        #     segment_ids = Input(shape=(None, max_length), dtype='int32', name='segment_ids')
        print("input_seqs", input_seqs)
        embedding_layer = ReusableEmbedding(input_dim=self.vocabulary_size,
                                            output_dim=self.word_embedding_size,
                                            input_length=self.max_length,
                                            name='bpe_embeddings',
                                            mask_zero=False,
                                            embeddings_regularizer=l2(1e-6)
                                            )
        #     segment_embedding_layer = Embedding(2, word_embedding_size, name='segment_embeddings')
        #     add_segment_layer = Add(name='add_segment')
        next_step_input, embedding_matrix = embedding_layer(input_seqs)
        #     segment_embeddings = segment_embedding_layer(segment_ids)
        print("embedding_layer", next_step_input)
        next_step_input = TransformerCoordinateEmbedding(transformer_depth,
                                                         name='coordinate_embedding')(next_step_input, step=0)
        #     next_step_input = add_segment_layer([next_step_input, segment_embeddings])

        for step in range(transformer_depth):
            next_step_input = TransformerBlock(name='transformer_' + str(step),
                                               num_heads=self.num_heads,
                                               residual_dropout=0.1,
                                               attention_dropout=0.1,
                                               use_masking=True,
                                               vanilla_wiring=True)(next_step_input)
            print("transformer_block", next_step_input)
        cls_node_slice = Lambda(lambda x: x[:, 0], name='cls_node_slicer')(next_step_input)
        if batch_norm:
            cls_node_slice = BatchNormalization(center=True, scale=True, name="encoder_output_normalized")(
                cls_node_slice)
        print("cls_node_slice", cls_node_slice)
        return Model(input_seqs, cls_node_slice, name="encoder_model")

    def create_embedding_model(self):
        encodings = Input(shape=(self._d,), name="input_seqs")
        subnetwork = Input(shape=(None,), name="subnetwork")

        graph_attention_1 = GraphAttention(int(self._d / self.num_heads), name="embedding_output",
                                           dropout_rate=0.2,
                                           activation='elu',
                                           attn_heads=self.num_heads,
                                           attn_heads_reduction='concat',
                                           kernel_regularizer=l2(5e-4 / 2),
                                           attn_kernel_regularizer=l2(5e-4 / 2)
                                           )([encodings, subnetwork])
        return Model([encodings, subnetwork], graph_attention_1, name="embedding_model")

    def create_cls_model(self):
        embeddings = Input(shape=(self._d,), name="embeddings")
        subnetwork = Input(shape=(None,), name="subnetwork")
        y_pred = GraphAttention(self.n_classes,
                                attn_heads=1,
                                attn_heads_reduction='average',
                                dropout_rate=0.2,
                                activation='sigmoid',
                                kernel_regularizer=l2(5e-4),
                                attn_kernel_regularizer=l2(5e-4))([embeddings, subnetwork])

        # y_pred = Dense(self.n_classes,
        #                activation='softmax',
        #                kernel_regularizer=l1())(graph_attention_2)

        return Model([embeddings, subnetwork], y_pred, name="cls_model")

    def build_keras_model(self, multi_gpu=False):
        K.clear_session()

        with tf.device("/cpu:0" if multi_gpu else "/gpu:0"):
            input_seqs = Input(shape=(None,), dtype=tf.int8, name="input_seqs")
            subnetwork = Input(shape=(None,), name="subnetwork")
            print("input_seqs", input_seqs)
            print("subnetwork", subnetwork)
            # chromosome_name = Input(batch_shape=(None, 323), sparse=False, dtype=tf.uint8, name="chromosome_name")
            # transcript_start = Input(batch_shape=(None, 1), sparse=False, dtype=tf.float32, name="transcript_start")
            # transcript_end = Input(batch_shape=(None, 1), sparse=False, dtype=tf.float32, name="transcript_end")

        with tf.device("/cpu:0" if multi_gpu else "/gpu:1"):
            self.encoder_model = self.create_encoder_network()  # Input: [input_seqs], output: encodings
            #     encoder_model = create_transformer_net()
            encodings = self.encoder_model(input_seqs)
            print("encodings", encodings)

        with tf.device("/cpu:0" if multi_gpu else "/gpu:2"):
            self.embedding_model = self.create_embedding_model()  # Input: [encodings, subnetwork], output: embeddings
            embeddings = self.embedding_model([encodings, subnetwork])
            print("embeddings", embeddings)

        with tf.device("/cpu:0" if multi_gpu else "/gpu:3"):
            self.cls_model = self.create_cls_model()  # Input: [embeddings, subnetwork], output: y_pred
            y_pred = self.cls_model([embeddings, subnetwork])

            self.model = Model(inputs=[input_seqs, subnetwork], outputs=y_pred, name="cls_model")
            print("cls_model", self.cls_model.inputs, self.cls_model.outputs)

        # Multi-gpu parallelization
        if multi_gpu: self.model = multi_gpu_model(self.model, gpus=4, cpu_merge=True, cpu_relocation=True)

        # Compile & train
        self.model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["top_k_categorical_accuracy", f1],
        )
        print(self.model.summary())

    def learn_embedding(self, generator_train, generator_test, early_stopping: int = False,
                        tensorboard=True, histogram_freq=0, embeddings=False,
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
        finally:
            self.save_model(self.log_dir)

    def save_model(self, log_dir):
        self.encoder_model.save(os.path.join(log_dir, "encoder_model.h5"))
        self.embedding_model.save(os.path.join(log_dir, "embedding_model.h5"))
        self.cls_model.save(os.path.join(log_dir, "cls_model.h5"))
        self.model.save(os.path.join(log_dir, "model.h5"))

    def load_model(self, log_dir):
        self.encoder_model.load_weights(os.path.join(log_dir, "encoder_model.h5"))
        self.embedding_model.load_weights(os.path.join(log_dir, "embedding_model.h5"))
        self.cls_model.load_weights(os.path.join(log_dir, "cls_model.h5"))
        self.model.load_weights(os.path.join(log_dir, "model.h5"))

    def get_embeddings(self, X):
        y_pred_encodings = self.encoder_model.predict(X)
        y_pred_emb = self.embedding_model.predict([y_pred_encodings, X["subnetwork"]],
                                                  batch_size=y_pred_encodings.shape[0])
        return y_pred_emb

    def predict(self, X):
        y_pred_emb = self.get_embeddings(X)
        y_pred = self.cls_model.predict(y_pred_emb, batch_size=y_pred_emb.shape[0])
        return y_pred

    def get_callbacks(self, early_stopping=10, tensorboard=True, histogram_freq=0, embeddings=False, write_grads=False):
        callbacks = []
        if tensorboard:
            if not hasattr(self, "tensorboard"):
                self.build_tensorboard(embeddings=embeddings, histogram_freq=histogram_freq, write_grads=write_grads)
            callbacks.append(self.tensorboard)

        if early_stopping > 0:
            if not hasattr(self, "early_stopping"):
                self.early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=early_stopping, verbose=0,
                                                    mode='auto',
                                                    baseline=None, restore_best_weights=False)
            callbacks.append(self.early_stopping)

        if len(callbacks) == 0: callbacks = None
        return callbacks

    def build_tensorboard(self, embeddings: bool, histogram_freq, write_grads):
        if not hasattr(self, "log_dir"):
            self.log_dir = "logs/{}_{}".format(type(self).__name__[0:20], time.strftime('%m-%d_%H-%M%p').strip(" "))
            print("created log_dir:", self.log_dir)

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