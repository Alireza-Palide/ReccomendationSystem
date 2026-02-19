import tensorflow as tf
import numpy as np
import os

class FMInteractionLayer(tf.keras.layers.Layer):
    """
    Custom layer to compute the 2nd-order interactions for Factorization Machines.
    Standard equation: Sum over all pairs (i,j) of <Vi, Vj> * xi * xj
    Since our inputs are embeddings (xi=1), it simplifies to sum of dot products.
    Efficient calculation: 0.5 * [ (sum(V))^2 - sum(V^2) ]
    """
    def __init__(self):
        super(FMInteractionLayer, self).__init__()

    def call(self, inputs):
        # inputs shape: [batch_size, num_features, embedding_dim]
        # This is a stack of all embedding vectors for a single example.

        # 1. Sum of vectors, then squared: ( v1 + v2 + ... )^2
        sum_squared = tf.square(tf.reduce_sum(inputs, axis=1))
        
        # 2. Square of vectors, then summed: ( v1^2 + v2^2 + ... )
        squared_sum = tf.reduce_sum(tf.square(inputs), axis=1)
        
        # 3. Final FM interaction term: 0.5 * (step1 - step2)
        # Result shape: [batch_size, embedding_dim]
        interaction_term = 0.5 * (sum_squared - squared_sum)
        return interaction_term


class DeepFM(tf.keras.Model):
    def __init__(self, feature_specs, embedding_dim=32, dnn_layers=[256, 128, 64], dropout_rate=0.3):
        """
        feature_specs: A dict mapping feature names to their vocabulary size.
                       e.g., {'CustomerCode': 1700, 'TownName': 50, ...}
        """
        super(DeepFM, self).__init__()
        self.embedding_dim = embedding_dim
        self.feature_specs = feature_specs
        
        # --- 1. Embedding Layers ---
        # We need one shared embedding layer for each categorical feature.
        # Both FM and Deep parts use these same embeddings.
        self.embeddings = {}
        for name, vocab_size in feature_specs.items():
            # +2 for padding/OOV bins
            self.embeddings[name] = tf.keras.layers.Embedding(
                input_dim=vocab_size + 2, 
                output_dim=embedding_dim,
                name=f'emb_{name}'
            )
            
        # --- 2. FM Component ---
        self.fm_layer = FMInteractionLayer()
        # 1st order (linear) term handles numeric features directly
        self.linear = tf.keras.layers.Dense(1, activation=None, name='linear_term')

        # --- 3. Deep Component (DNN) ---
        self.dnn = tf.keras.Sequential(name='dnn')
        for i, size in enumerate(dnn_layers):
            self.dnn.add(tf.keras.layers.Dense(size, activation='relu', name=f'dnn_{i}'))
            self.dnn.add(tf.keras.layers.Dropout(dropout_rate))
        # Final projection for DNN part
        self.dnn.add(tf.keras.layers.Dense(1, activation=None, name='dnn_output'))

        # --- 4. Final Output ---
        # Combines FM and Deep parts into a probability
        self.final_activation = tf.keras.layers.Activation('sigmoid', name='output_prob')

    def call(self, inputs):
        # inputs maps feature_name -> tensor batch
        
        # A. Process Categorical Features (Embeddings)
        embedding_list = []
        for name, layer in self.embeddings.items():
            # Lookup embedding and expand dim to [batch, 1, dim] for stacking
            emb = tf.expand_dims(layer(inputs[name]), axis=1)
            embedding_list.append(emb)
            
        # Stack into tensor: [batch_size, num_categorical_features, embedding_dim]
        embeddings_tensor = tf.concat(embedding_list, axis=1)

        # B. Process Numeric Features
        # Collect all inputs not used in embeddings
        numeric_features = []
        for name, tensor in inputs.items():
            if name not in self.embeddings:
                # Ensure numeric features are float32 and shape [batch, 1]
                numeric_features.append(tf.cast(tf.expand_dims(tensor, -1), tf.float32))
        
        if numeric_features:
            numeric_tensor = tf.concat(numeric_features, axis=1)
        else:
            # Handle case if there are no numeric features
            numeric_tensor = tf.zeros_like(inputs[list(inputs.keys())[0]], dtype=tf.float32)
            numeric_tensor = tf.expand_dims(numeric_tensor, -1)

        # --- FM Calculation ---
        # 1. Linear Term (Wait*Numeric features)
        fm_linear = self.linear(numeric_tensor)
        # 2. Interaction Term (sum over embedding dimensions to get scalar)
        fm_interactions = tf.reduce_sum(self.fm_layer(embeddings_tensor), axis=1, keepdims=True)
        fm_output = fm_linear + fm_interactions

        # --- Deep Calculation ---
        # Flatten embeddings for DNN input: [batch_size, num_cat * dim]
        dnn_input_emb = tf.keras.layers.Flatten()(embeddings_tensor)
        # Concatenate with numeric: [batch_size, (num_cat*dim) + num_numeric]
        dnn_input = tf.concat([dnn_input_emb, numeric_tensor], axis=1)
        dnn_output = self.dnn(dnn_input)

        # --- Combine ---
        # Final Logit = FM_part + DNN_part
        total_logit = fm_output + dnn_output
        return self.final_activation(total_logit)
