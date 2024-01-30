import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers

from v2.models.base import AbstractModel
class TransformerBlock(layers.Layer, metaclass=AbstractModel):
    def __init__(self, config: DictConfig):
        super(TransformerBlock, self).__init__()
        self.config = config 
        self.att = layers.MultiHeadAttention(num_heads=self.config.model.num_heads, key_dim=self.config.model.embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(self.config.model.ff_dim, activation="gelu"), layers.Dense(self.config.model.feat_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.config.model.rate)
        self.dropout2 = layers.Dropout(self.config.model.rate)

    def build_block(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def __call__(self):
        pass
    
    def _train(self, X_train, X_valid, y_train, y_valid, fold):
        inp = layers.Input(shape=(self.config.model.seq_len, self.config.model.feat_dim))
        embeddings = []
        # 11 embedding layers
        for k in range(11):
            emb = layers.Embedding(self.config.model.max_features[k], 8)(inp[:, :, k])
            embeddings.append(emb)
        x = layers.Concatenate()([inp[:,:,11:]]+embeddings)
        x = layers.Dense(self.config.model.feat_dim)(x)

        for k in range(self.config.model.n_blocks):
            x_old = x 
            transformer_block = TransformerBlock(self.config)(x)
            x = 0.9*transformer_block + 0.1*x_old 
        
        # Classification head
        
        x = layers.Dense(64, activation="relu")(x[:, -1, :])
        x = layers.Dense(32, activation="relu")(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        
        model = keras.Model(inputs=inp, outputs=outputs)
        optimizer = tf.keras.optimizer.Adam(learning_rate=self.config.model.lr)
        loss = tf.keras.losses.BinaryCrossentropy()
        model.compile(loss=loss, optimizer=optimizer)
        h = model.fit(X_train,y_train, 
                      validation_data = (X_valid,y_valid),
                      batch_size=self.config.model.batch_size, epochs=self.config.model.epochs, verbose=self.config.model.verbose,
                      callbacks = [
                          tf.keras.callbacks.LearningRateScheduler(verbose=True)
                      ])
        if not os.path.exists(self.config.model.path): 
            os.makedirs(self.config.model.path)
        # save weights of current fold model
        model.save_weights(f'{self.config.model.path}transformer_fold_{fold+1}.h5')
        
    def predict(self, results, test_x) -> np.ndarray:
        # INFER VALID DATA
        print('Inferring validation data...')
        p = model.predict(X_valid, batch_size=512, verbose=VERBOSE).flatten()

        print()
        print(f'Fold {fold+1} CV=', amex_metric_mod(y_valid, p) )
        print()
        true = np.concatenate([true, y_valid])
        oof = np.concatenate([oof, p])
        
        # CLEAN MEMORY
        del model, X_train, y_train, X_valid, y_valid, p
        gc.collect()
        
        # 