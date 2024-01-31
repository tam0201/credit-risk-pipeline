import os
import numpy as np
import pandas as pd
from omegaconf import DictConfig
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers

from v2.models.base import AbstractModel, ModelResult

def amex_metric_mod(y_true, y_pred):
    ### Custom metric for AMEX competition 
    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)

def lrfn(config, epoch):
    import math
    if epoch < config.lrfn.LR_RAMPUP_EPOCHS:
        lr = (config.lrfn.LR_MAX - config.lrfn.LR_START) / config.lrfn.LR_RAMPUP_EPOCHS * epoch + config.lrfn.LR_START
    elif epoch < config.lrfn.LR_RAMPUP_EPOCHS + config.lrfn.LR_SUSTAIN_EPOCHS:
        lr = config.lrfn.LR_MAX
    else:
        decay_total_epochs = config.lrfn.EPOCHS - config.lrfn.LR_RAMPUP_EPOCHS - config.lrfn.LR_SUSTAIN_EPOCHS - 1
        decay_epoch_index = epoch - config.lrfn.LR_RAMPUP_EPOCHS - config.lrfn.LR_SUSTAIN_EPOCHS
        phase = math.pi * decay_epoch_index / decay_total_epochs
        cosine_decay = 0.5 * (1 + math.cos(phase))
        lr = (config.lrfn.LR_MAX - config.lrfn.LR_MIN) * cosine_decay + config.lrfn.LR_MIN
    return lr

class TransformerTrainer(layers.Layer, metaclass=AbstractModel):
    def __init__(self, config: DictConfig):
        super(TransformerTrainer, self).__init__()
        self.config = config 
        self.att = layers.MultiHeadAttention(num_heads=self.config.model.num_heads, key_dim=self.config.model.embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(self.config.model.ff_dim, activation="gelu"), layers.Dense(self.config.model.feat_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.config.model.rate)
        self.dropout2 = layers.Dropout(self.config.model.rate)

    def _build_block(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def _train(self, X_train, X_valid, y_train, y_valid, fold):
        inp = layers.Input(shape=(self.config.model.seq_len, self.config.model.feat_dim))
        embeddings = []
        # 11 embedding layers
        for k in range(self.config.model.emb_layers):
            emb = layers.Embedding(self.config.model.max_features[k], 8)(inp[:, :, k])
            embeddings.append(emb)
        x = layers.Concatenate()([inp[:,:,11:]]+embeddings)
        x = layers.Dense(self.config.model.feat_dim)(x)

        for k in range(self.config.model.n_blocks):
            x_old = x 
            transformer_block = self._build_block(x, training=True)
            x = 0.9*transformer_block + 0.1*x_old 
        
        # Classification head
        x = layers.GRU(units=128, return_sequences=False)(x) 
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
                          tf.keras.callbacks.LearningRateScheduler(lrfn)
                      ])
        if not os.path.exists(self.config.model.path): 
            os.makedirs(self.config.model.path)
        # save weights of current fold model
        model.save_weights(f'{self.config.model.path}transformer_fold_{fold+1}.h5')
        return model 
    
    def predict(self, model, x_valid, y_valid, fold) -> np.ndarray:
        # INFER VALID DATA
        print('Inferring validation data...')
        p = model.predict(x_valid, batch_size=512, verbose=self.config.model.verbose).flatten()

        print()
        print(f'Fold {fold+1} CV=', amex_metric_mod(y_valid, p) )
        print()
        true = np.concatenate([true, y_valid])
        oof = np.concatenate([oof, p])
        return (true, oof)
    
    def train(self):
        true = np.array([])
        oof = np.array([])
        for fold in range(5):
            # INDICES OF TRAIN AND VALID FOLDS
            valid_idx = [2*fold+1, 2*fold+2]
            train_idx = [x for x in [1,2,3,4,5,6,7,8,9,10] if x not in valid_idx]
            print('#'*25)
            print(f'### Fold {fold+1} with valid files', valid_idx)

            # READ TRAIN DATA FROM DISK
            X_train = []; y_train = []
            for k in train_idx:
                X_train.append( np.load(f'{self.config.data.path}data_{k}.npy'))
                y_train.append( pd.read_parquet(f'{self.config.data.path}targets_{k}.pqt') )
            X_train = np.concatenate(X_train,axis=0)
            y_train = pd.concat(y_train).target.values
            print('### Training data shapes', X_train.shape, y_train.shape)

            # READ VALID DATA FROM DISK
            X_valid = []; y_valid = []
            for k in valid_idx:
                X_valid.append( np.load(f'{self.config.data.path}data_{k}.npy'))
                y_valid.append( pd.read_parquet(f'{self.config.data.path}targets_{k}.pqt') )
            X_valid = np.concatenate(X_valid,axis=0)
            y_valid = pd.concat(y_valid).target.values
            print('### Validation data shapes', X_valid.shape, y_valid.shape)
            print('#'*25)

            # BUILD AND TRAIN MODEL
            K.clear_session()
            model = self._train(X_train, X_valid, y_train, y_valid, fold)
            (true, oof) = self.predict(model, X_valid, y_valid, fold)
        print('#'*25)
        print(f'Overall CV =', amex_metric_mod(true, oof) )

        return ModelResult(
            oof_preds = oof,
            models=model,
            preds=None, 
            scores={"Overall CV": amex_metric_mod(true, oof)}
        )
    