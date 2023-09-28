import tensorflow as tf
from collections import  defaultdict
class TimeSeriesPredictor(tf.keras.Model):
    def __init__(self, feature_list: tp.List[str], num_reccurrent_dims: int, output_size: int,
                 text_model: tf.keras.Model = None):
        super().__init__()
        self.rnn_layers = defaultdict(
            tf.keras.layers.Layer)  # у відповідність кожній нейронці свій часовий ряд, defaultdict values фіксується
        self.feature_list = feature_list
        # encoder (LSTM від кожної з ознак)
        self.text_model = False
        for feature in self.feature_list:
            if feature.split('_')[1] == 'text':
                if text_model:
                    self.text_model = True
                    text_model_ = text_model
                    text_model_.trainable = False
                    self.rnn_layers[feature] = text_model_
            else:
                self.rnn_layers[feature] = tf.keras.Sequential(
                    [
                        tf.keras.layers.LSTM(num_reccurrent_dims)  # return_sequences повертає усі значення ряду
                    ]  # [1, 2, 3, 4]
                    # LSTM_1 => 1->2->3->4--> output_1
                    # LSTM_2 => 4 ->3->2->--> output_2
                    # output = output_2 (+) output_1 (+) - або сума, або середнє або конкатенація
                )
        # decoder (Dense шари по кожній ознаці + Dense розміром з визідну характеристику)
        # f(Y*, Y) = f(X*W, Y) = (Y-X*W)^2 + lambda*sum(|W|), lambda - коеф регуляризації (L1) (LASSO regression)
        # f(Y*, Y) = f(X*W, Y) = (Y-X*W)^2 + lambda*sum(W^2), lambda - коеф регуляризації (L2) (Ridge)
        # f(Y*, Y) = f(X*W, Y) = (Y-X*W)^2 + lambda*sum(W^2) + kappa*sum(|W|), регуляризації (L2+L1) (Elastic Net)
        # |W| -> median (непотрібні фічі можна видалити) |W|
        # W^2 -> mean (просто згладжує, намагається триати ваги якогомога меншими за значенням)
        self.dense_output = tf.keras.layers.Dense(output_size)

    def call(self, x: tf.Tensor):
        rnn_outputs = []
        for feature in self.feature_list:
            if feature.split('_')[1] == 'text':
                if self.text_model:
                    rnn_outputs.append(self.rnn_layers[feature](x[feature]))
            else:
                rnn_outputs.append(self.rnn_layers[feature](x[feature]))  # х - вхідний часовий ряд
        rnn_outputs = tf.concat(rnn_outputs, axis=-1)  # скласти по колонкам
        out = self.dense_output(rnn_outputs)
        return out

    class SentimentModel(tf.keras.Model):
        def __init__(self,
                     max_token: int,
                     kernel_size: int,
                     filters: int):
            super().__init__(self)
            self.vectorizer = tf.keras.layers.TextVectorization(output_sequence_length=max_token, vocabulary=all_vocab)
            self.embeddings = tf.keras.layers.Embedding(len(all_vocab) + 2, 50, weights=tf.constant([vocabulary]))
            self.conv_1 = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')
            self.average_1 = tf.keras.layers.AveragePooling1D(2)
            self.conv_2 = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')
            self.average_2 = tf.keras.layers.AveragePooling1D(2)
            self.flatten = tf.keras.layers.Flatten()
            self.dense = tf.keras.layers.Dense(64, activation='relu')
            self.output_layer = tf.keras.layers.Dense(3, activation='softmax')

        def call(self, input_):
            o = self.vectorizer(input_)
            o = self.embeddings(o)
            o = self.conv_1(o)
            o = self.average_1(o)
            o = self.conv_2(o)
            o = self.average_2(o)
            o = self.flatten(o)
            o = self.dense(o)
            o = self.output_layer(o)
            return o
