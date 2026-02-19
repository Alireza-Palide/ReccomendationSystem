import tensorflow as tf
import tensorflow_recommenders as tfrs
import os

class UserTower(tf.keras.Model):
    def __init__(self, layer_sizes, vocab_dir):
        super().__init__()
        self.vocab_dir = vocab_dir

        self.user_embedding = self._create_embedding("CustomerCode", 32)
        self.town_embedding = self._create_embedding("TownName", 16)
        self.cluster_embedding = self._create_embedding("Cluster", 8)
        self.group_header_embedding = self._create_embedding("GroupHeaderName", 8)
        self.area_embedding = self._create_embedding("Area", 4)
        self.region_embedding = self._create_embedding("RegionCategory", 4)
        
        self.dense_layers = tf.keras.Sequential()
        for size in layer_sizes:
            self.dense_layers.add(tf.keras.layers.Dense(size, activation="relu"))
            self.dense_layers.add(tf.keras.layers.Dropout(0.4)) 
            
        self.projection = tf.keras.layers.Dense(layer_sizes[-1]) 

    def _create_embedding(self, name, dim):
        path = os.path.join(self.vocab_dir, f"{name}.txt")
        try:
            vocab_size = sum(1 for _ in open(path, encoding="utf-8")) + 2
        except FileNotFoundError:
            vocab_size = 10 
            print(f"Warning: Vocab file {path} not found. Using default size 10.")
        return tf.keras.layers.Embedding(vocab_size, dim)

    def call(self, inputs):
        user_vec = self.user_embedding(inputs["CustomerCode"])
        town_vec = self.town_embedding(inputs["TownName"])
        cluster_vec = self.cluster_embedding(inputs["Cluster"])
        group_header_vec = self.group_header_embedding(inputs["GroupHeaderName"])
        area_vec = self.area_embedding(inputs["Area"])
        region_vec = self.region_embedding(inputs["RegionCategory"])
        
        tenure = tf.expand_dims(tf.cast(inputs["TenureYears"], tf.float32), -1)
        
        feature_vector = tf.concat([
            user_vec, town_vec, cluster_vec, 
            group_header_vec, area_vec, region_vec,
            tenure
        ], axis=1)
        
        return self.projection(self.dense_layers(feature_vector))

class ItemTower(tf.keras.Model):
    def __init__(self, layer_sizes, vocab_dir):
        super().__init__()
        self.vocab_dir = vocab_dir
        
        self.item_embedding = self._create_embedding("ProductCode", 32)
        self.group_header_embedding = self._create_embedding("ProductGroupHeader", 16)
        self.group_name_embedding = self._create_embedding("ProductGroupName", 16)
        
        self.dense_layers = tf.keras.Sequential()
        for size in layer_sizes:
            self.dense_layers.add(tf.keras.layers.Dense(size, activation="relu"))
            
        self.projection = tf.keras.layers.Dense(layer_sizes[-1])

    def _create_embedding(self, name, dim):
        path = os.path.join(self.vocab_dir, f"{name}.txt")
        try:
            vocab_size = sum(1 for _ in open(path, encoding="utf-8")) + 2
        except FileNotFoundError:
            vocab_size = 10
            print(f"Warning: Vocab file {path} not found. Using default size 10.")
        return tf.keras.layers.Embedding(vocab_size, dim)

    def call(self, inputs):
        item_vec = self.item_embedding(inputs["ProductCode"])
        group_header_vec = self.group_header_embedding(inputs["ProductGroupHeader"])
        group_name_vec = self.group_name_embedding(inputs["ProductGroupName"])

        price = tf.expand_dims(tf.cast(inputs["Price"], tf.float32), -1)
        bestseller = tf.expand_dims(tf.cast(inputs["IsBestSeller"], tf.float32), -1)
        
        feature_vector = tf.concat([
            item_vec, group_header_vec, group_name_vec,
            price, bestseller
        ], axis=1)
        
        return self.projection(self.dense_layers(feature_vector))

class RetrievalModel(tfrs.Model):
    def __init__(self, layer_sizes, vocab_dir, candidates_dataset):
        super().__init__()
        self.query_model = UserTower(layer_sizes, vocab_dir)
        self.candidate_model = ItemTower(layer_sizes, vocab_dir)
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidates_dataset.map(self.candidate_model),
                ks=[10,20,50] 
            )
        )
 
    def compute_loss(self, features):
        query_embeddings = self.query_model(features)
        candidate_embeddings = self.candidate_model(features)
        return self.task(query_embeddings, candidate_embeddings)
