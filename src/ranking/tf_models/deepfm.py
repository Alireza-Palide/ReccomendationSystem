import tensorflow as tf
import numpy as np
import os

class FMInteractionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(FMInteractionLayer, self).__init__()

    def call(self, inputs):
        sum_squared = tf.square(tf.reduce_sum(inputs, axis=1))
        squared_sum = tf.reduce_sum(tf.square(inputs), axis=1)
        interaction_term = 0.5 * (sum_squared - squared_sum)
        return interaction_term


class DeepFM(tf.keras.Model):
    def __init__(self, feature_specs, embedding_dim=32, dnn_layers=[256, 128, 64], dropout_rate=0.3):
        super(DeepFM, self).__init__()
        self.embedding_dim = embedding_dim
        self.feature_specs = feature_specs

        self.embeddings = {}
        for name, vocab_size in feature_specs.items():
            self.embeddings[name] = tf.keras.layers.Embedding(
                input_dim=vocab_size + 2, 
                output_dim=embedding_dim,
                name=f'emb_{name}'
            )
            
        self.fm_layer = FMInteractionLayer()
        self.linear = tf.keras.layers.Dense(1, activation=None, name='linear_term')

        self.dnn = tf.keras.Sequential(name='dnn')
        for i, size in enumerate(dnn_layers):
            self.dnn.add(tf.keras.layers.Dense(size, activation='relu', name=f'dnn_{i}'))
            self.dnn.add(tf.keras.layers.Dropout(dropout_rate))
        self.dnn.add(tf.keras.layers.Dense(1, activation=None, name='dnn_output'))

        self.final_activation = tf.keras.layers.Activation('sigmoid', name='output_prob')

    def call(self, inputs):
        embedding_list = []
        for name, layer in self.embeddings.items():
            emb = tf.expand_dims(layer(inputs[name]), axis=1)
            embedding_list.append(emb)
            
        embeddings_tensor = tf.concat(embedding_list, axis=1)

        numeric_features = []
        for name, tensor in inputs.items():
            if name not in self.embeddings:
                numeric_features.append(tf.cast(tf.expand_dims(tensor, -1), tf.float32))
        
        if numeric_features:
            numeric_tensor = tf.concat(numeric_features, axis=1)
        else:
            numeric_tensor = tf.zeros_like(inputs[list(inputs.keys())[0]], dtype=tf.float32)
            numeric_tensor = tf.expand_dims(numeric_tensor, -1)

        fm_linear = self.linear(numeric_tensor)
        fm_interactions = tf.reduce_sum(self.fm_layer(embeddings_tensor), axis=1, keepdims=True)
        fm_output = fm_linear + fm_interactions
        dnn_input_emb = tf.keras.layers.Flatten()(embeddings_tensor)
        dnn_input = tf.concat([dnn_input_emb, numeric_tensor], axis=1)
        dnn_output = self.dnn(dnn_input)

        total_logit = fm_output + dnn_output
        return self.final_activation(total_logit)
