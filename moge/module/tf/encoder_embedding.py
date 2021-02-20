import datetime
import os

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
# from keras_transformer.extras import ReusableEmbedding
# from keras_transformer.position import TransformerCoordinateEmbedding
# from keras_transformer.transformer import TransformerBlock
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dropout, MaxPooling1D, Embedding, Bidirectional, LSTM, \
    BatchNormalization, Dense, Convolution1D, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import multi_gpu_model

from moge.module.tf.metrics import precision, recall
# from kegra.layers.graph import GraphConvolution
# from spektral.layers import GraphConv
from .graph_attention_layer import GraphAttention
from .static_graph_embedding import NeuralGraphEmbedding


class EncoderEmbedding(NeuralGraphEmbedding):
    def __init__(self,
                 embedding_d: int, encoding_d: int, attn_heads: int,
                 encoding_dropout, embedding_dropout, cls_dropout,
                 lstm_units, batchnorm: bool,
                 batch_size: int, vocabulary_size: int, word_embedding_size: int,
                 loss: str,
                 max_length: int, targets: str, n_classes: int, multi_gpu=False, verbose=False):
        self.targets = targets
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.max_length = max_length
        self.vocabulary_size = vocabulary_size
        self.word_embedding_size = word_embedding_size

        self.encoding_d = encoding_d
        self.batchnorm = batchnorm
        self.encoding_dropout = encoding_dropout
        self.embedding_dropout = embedding_dropout
        self.cls_dropout = cls_dropout
        self.lstm_units = lstm_units
        self.num_heads = attn_heads
        self.loss = loss

        self.verbose = verbose
        super(EncoderEmbedding, self).__init__(embedding_d, method_name="GCN_embedding")
        self.build_keras_model(multi_gpu)

    def create_encoder_network(self, batch_norm=True, encoding_dropout=0.2, lstm_units=320, encoding_d=256):
        input_seqs = Input(shape=(None,), name="input_seqs")  # (batch_number, sequence_length)
        x = Embedding(input_dim=self.vocabulary_size,
                      output_dim=self.vocabulary_size - 1,
                      input_length=None, mask_zero=True, trainable=True)(
            input_seqs)  # (batch_number, sequence_length, 5)

        x = Convolution1D(filters=320, kernel_size=26, activation='relu',
                          data_format="channels_last", name="lstm_conv_1")(
            x)  # (batch_number, sequence_length-5, 1, 192)
        if batch_norm:
            x = BatchNormalization(center=True, scale=True, name="conv1_batch_norm")(x)
        x = MaxPooling1D(pool_size=13, padding="same")(x)
        x = Dropout(encoding_dropout)(x)

        x = Convolution1D(filters=192, kernel_size=6, activation='relu', name="lstm_conv_2")(x)
        if batch_norm:
            x = BatchNormalization(center=True, scale=True, name="conv2_batch_norm")(x)
        x = MaxPooling1D(pool_size=3, padding="same")(x)
        x = Dropout(encoding_dropout)(x)

        x = Bidirectional(LSTM(lstm_units, return_sequences=False, return_state=False),
                          merge_mode='concat')(x)  # (batch_number, 320+320)
        x = Dropout(encoding_dropout)(x)

        x = Dense(encoding_d, activation='linear', name="encoder_output")(x)
        if batch_norm:
            x = BatchNormalization(center=True, scale=True, name="encoder_output_normalized")(x)

        return Model(input_seqs, x, name="encoder_model")

    # def create_transformer_net(self, batch_norm=True, transformer_depth=1):
    #     input_seqs = Input(shape=(None,), name="input_seqs")  # (batch_number, sequence_length)
    #     #     segment_ids = Input(shape=(None, max_length), dtype='int32', name='segment_ids')
    #     print("input_seqs", input_seqs)
    #     embedding_layer = ReusableEmbedding(input_dim=self.vocabulary_size,
    #                                         output_dim=self.word_embedding_size,
    #                                         input_length=self.max_length,
    #                                         name='bpe_embeddings',
    #                                         mask_zero=False,
    #                                         embeddings_regularizer=l2(1e-6)
    #                                         )
    #     #     segment_embedding_layer = Embedding(2, word_embedding_size, name='segment_embeddings')
    #     #     add_segment_layer = Add(name='add_segment')
    #     next_step_input, embedding_matrix = embedding_layer(input_seqs)
    #     #     segment_embeddings = segment_embedding_layer(segment_ids)
    #     print("embedding_layer", next_step_input)
    #     next_step_input = TransformerCoordinateEmbedding(transformer_depth,
    #                                                      name='coordinate_embedding')(next_step_input, step=0)
    #     #     next_step_input = add_segment_layer([next_step_input, segment_embeddings])
    #
    #     for step in range(transformer_depth):
    #         next_step_input = TransformerBlock(name='transformer_' + str(step),
    #                                            num_heads=self.num_heads,
    #                                            residual_dropout=0.1,
    #                                            attention_dropout=0.1,
    #                                            use_masking=True,
    #                                            vanilla_wiring=True)(next_step_input)
    #         print("transformer_block", next_step_input)
    #     cls_node_slice = Lambda(lambda x: x[:, 0], name='cls_node_slicer')(next_step_input)
    #     if batch_norm:
    #         cls_node_slice = BatchNormalization(center=True, scale=True, name="encoder_output_normalized")(
    #             cls_node_slice)
    #     print("cls_node_slice", cls_node_slice)
    #     return Model(input_seqs, cls_node_slice, name="encoder_model")

    def create_embedding_model(self, encoding_d=128, embedding_d=128, num_heads=4, embedding_dropout=0.5):
        encodings = Input(shape=(encoding_d,), name="input_seqs")
        subnetwork = Input(shape=(None,), name="subnetwork")

        graph_attention_1 = GraphAttention(int(embedding_d / num_heads), name="embedding_gat",
                                           dropout_rate=embedding_dropout,
                                           activation=LeakyReLU(0.2),
                                           attn_heads=num_heads,
                                           attn_heads_reduction='concat',
                                           kernel_regularizer=l2(5e-4 / 2),
                                           attn_kernel_regularizer=l2(5e-4 / 2)
                                           )([encodings, subnetwork])

        return Model([encodings, subnetwork], graph_attention_1, name="embedding_model")

    def create_cls_model(self, embedding_d, cls_dropout):
        embeddings = Input(shape=(embedding_d,), name="embeddings")
        subnetwork = Input(shape=(None,), name="subnetwork")

        y_pred = GraphAttention(self.n_classes, name="cls_gat",
                                attn_heads=1,
                                attn_heads_reduction='average',
                                dropout_rate=cls_dropout,
                                activation='sigmoid',
                                kernel_regularizer=l2(5e-4),
                                attn_kernel_regularizer=l2(5e-4))([embeddings, subnetwork])

        return Model([embeddings, subnetwork], y_pred, name="cls_model")

    def build_keras_model(self, multi_gpu=False):
        K.clear_session()

        with tf.device("/cpu:0" if multi_gpu else "/gpu:0"):
            input_seqs = Input(shape=(None,), name="input_seqs")
            subnetwork = Input(shape=(None,), name="subnetwork")

        with tf.device("/cpu:0" if multi_gpu else "/gpu:1"):
            self.encoder_model = self.create_encoder_network(batch_norm=self.batchnorm,
                                                             encoding_dropout=self.encoding_dropout,
                                                             lstm_units=self.lstm_units,
                                                             encoding_d=self.encoding_d,
                                                             )  # Input: [input_seqs], output: encodings
            encodings = self.encoder_model(input_seqs)

        with tf.device("/cpu:0" if multi_gpu else "/gpu:2"):
            self.embedding_model = self.create_embedding_model(encoding_d=self.encoding_d,
                                                               embedding_d=self.embedding_d,
                                                               num_heads=self.num_heads,
                                                               embedding_dropout=self.embedding_dropout)  # Input: [encodings, subnetwork], output: embeddings
            embeddings = self.embedding_model([encodings, subnetwork])

        with tf.device("/cpu:0" if multi_gpu else "/gpu:3"):
            self.cls_model = self.create_cls_model(embedding_d=self.embedding_d,
                                                   cls_dropout=self.cls_dropout)  # Input: [embeddings, subnetwork], output: y_pred
            y_pred = self.cls_model([embeddings, subnetwork])

            self.model = Model(inputs=[input_seqs, subnetwork], outputs=y_pred, name="cls_model")
            print("cls_model", self.cls_model.inputs, self.cls_model.outputs) if self.verbose else None

        # Multi-gpu parallelization
        if multi_gpu: self.model = multi_gpu_model(self.model, gpus=4, cpu_merge=True, cpu_relocation=True)

        # Compile & train
        self.model.compile(
            loss=self.loss,
            optimizer="adam",
            metrics=["top_k_categorical_accuracy", precision, recall],
        )
        print(self.model.summary()) if self.verbose else None

    def learn_embedding(self, generator_train, generator_test, epochs=50,
                        early_stopping: int = False, model_checkpoint=False, tensorboard=True, hparams=None,
                        save_model=False, verbose=2, **kwargs):
        self.generator_train = generator_train
        self.generator_test = generator_test
        try:
            self.hist = self.model.fit_generator(generator_train, epochs=epochs, shuffle=False,
                                                 validation_data=generator_test,
                                                 callbacks=self.get_callbacks(early_stopping, tensorboard,
                                                                              model_checkpoint, hparams),
                                                 use_multiprocessing=True, workers=2, verbose=verbose, **kwargs)
        except KeyboardInterrupt:
            print("Stop training") if self.verbose else None
        finally:
            if save_model: self.save_model(self.log_dir)

    def save_model(self, log_dir):
        self.encoder_model.save(os.path.join(log_dir, "encoder_model.h5"))
        self.embedding_model.save(os.path.join(log_dir, "embedding_model.h5"))
        self.cls_model.save(os.path.join(log_dir, "cls_model.h5"))
        self.model.save(os.path.join(log_dir, "model.h5"))

    def load_model(self, log_dir):
        self.encoder_model = load_model(os.path.join(log_dir, "encoder_model.h5"))
        self.embedding_model = load_model(os.path.join(log_dir, "embedding_model.h5"))
        self.cls_model = load_model(os.path.join(log_dir, "cls_model.h5"))
        self.model = load_model(os.path.join(log_dir, "model.h5"))

    def get_embeddings(self, X):
        y_pred_encodings = self.encoder_model.predict(X)
        y_pred_emb = self.embedding_model.predict([y_pred_encodings, X["subnetwork"]],
                                                  batch_size=y_pred_encodings.shape[0])
        return y_pred_emb

    def predict(self, X):
        y_pred_emb = self.get_embeddings(X)
        y_pred = self.cls_model.predict(y_pred_emb, batch_size=y_pred_emb.shape[0])
        return y_pred

    def get_callbacks(self, early_stopping=10, tensorboard=True, model_checkpoint=True, hparams=None,
                      histogram_freq=0, write_grads=0):
        if not hasattr(self, "log_dir"):
            self.log_dir = os.path.join("logs", str.join("-", self.targets) + "_" + datetime.datetime.now().strftime(
                "%m-%d_%H-%M%p"))
            print("created log_dir:", self.log_dir) if self.verbose else None

        callbacks = []
        if tensorboard:
            if not hasattr(self, "tensorboard"):
                self.tensorboard = tf.keras.callbacks.TensorBoard(
                    log_dir=self.log_dir,
                    histogram_freq=histogram_freq,
                    write_grads=write_grads, write_graph=False, write_images=False,
                    update_freq="epoch", )
            callbacks.append(self.tensorboard)

        if early_stopping > 0:
            if not hasattr(self, "early_stopping"):
                self.early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=early_stopping, verbose=0,
                                                    mode='auto',
                                                    baseline=None, restore_best_weights=False)
            callbacks.append(self.early_stopping)

        if model_checkpoint:
            self.model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.log_dir, "model_checkpoint.hdf5"),
                monitor='top_k_categorical_accuracy',
                save_best_only=True)
            callbacks.append(self.model_checkpoint)

        if hparams:
            self.hp_callback = hp.KerasCallback("logs/hparam_tuning/" + self.log_dir.split("/")[-1],
                                                hparams)
            callbacks.append(self.hp_callback)

        if len(callbacks) == 0: return None
        return callbacks


