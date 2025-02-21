import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import polars as pl
import joblib

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
def build_bilstm(trial):
    model_BiLSTM = Sequential([
            Input(shape=(100, features)),


            Bidirectional(LSTM(
                units = trial.suggest_int("units_1", 64, 256, step=32),
                return_sequences=True,
                kernel_regularizer=regularizers.l1_l2(
                    l1=trial.suggest_float("l1_1_layer1", 1e-3, 3e-1),
                    l2=trial.suggest_float("l2_1_layer1", 1e-3, 3e-1),
                    ),
                recurrent_regularizer=regularizers.l1_l2(
                    l1=trial.suggest_float("l1_2_layer1", 1e-3, 3e-1),
                    l2=trial.suggest_float("l2_2_layer1", 1e-3, 3e-1),
                    ),
                )),
            Dropout(trial.suggest_float("dropout_1", 1e-5,1e-2)),
            Bidirectional(LSTM(
                units = trial.suggest_int("units_2", 64, 256, step = 32),
                return_sequences=True,
                kernel_regularizer=regularizers.l2(
                    l2 = trial.suggest_float("l2_1_layer2", 1e-3, 3e-1)
                    ),
                recurrent_regularizer=regularizers.l2(
                    l2 = trial.suggest_float("l2_2_layer2", 1e-3, 3e-1)
                    ),
                )),

            Dropout(trial.suggest_float("dropout_2", 1e-3, 3e-1)),
            Bidirectional(LSTM(
                units = trial.suggest_int("units_3", 64, 256, step = 32),
                return_sequences=True,
                kernel_regularizer=regularizers.l2(
                    l2 = trial.suggest_float("l2_1_layer3", 1e-3, 3e-1)
                    ),
                recurrent_regularizer=regularizers.l2(
                    l2 = trial.suggest_float("l2_2_layer3", 1e-3, 3e-1)
                    ),
                )) ,

            Dropout(trial.suggest_float("dropout_3", 1e-3, 3e-1)),
            Bidirectional(LSTM(
                units= trial.suggest_int("unit_4", 32,256, step = 32) ,
                return_sequences=True)),
            Dropout(trial.suggest_float("dropout_4", 1e-3, 3e-1)),
            Bidirectional(LSTM(
                units= trial.suggest_int("unit_5", 32,256, step = 32) ,
                return_sequences=True)),

            Dense(units = trial.suggest_int("Dense_1", 64,256, step = 32), activation = 'relu'),
            Dense(units = trial.suggest_int("Dense_2", 64,256, step = 32), activation = 'relu'),
            Dense(targets)
        ])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-3, 3e-1)
    model_BiLSTM.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=trial.suggest_float("clip_norm", 1.0, 3.0)),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError(name='mse'),
                 tf.keras.metrics.MeanAbsoluteError(name='mae'),
                ]
    )
    return model_BiLSTM

def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    model = build_bilstm(trial)
    history = model.fit(
        X_train_split_reshape, y_train_split_reshape,
        validation_data=validation_data,
        epochs=100, 
        batch_size=batch_size,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=10, min_lr=1e-5)
        ],
        verbose=0
    )
    tf.keras.backend.clear_session()
    return min(history.history["val_loss"])

df = pl.read_csv("home/data.csv")
columns = [
'noperation','nsampling','valve','voltage','current','pressure','torque',
 'theta0','theta1','theta2',
 'cos1','cos2_inv','cos2_inv2','cos2_inv3',
 'd_valve','d_press','dv_sign','status']

X = df[columns].drop('torque')
y = df[['torque']]


features = len(X.columns)
targets = len(y.columns)


scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)
X_train_split, X_val, y_train_split, y_val = train_test_split(X_scaler,y, test_size= 0.25)

X_train_split_reshape = X_train_split.reshape(X_train_split.shape[0]//100, 100, features)

X_val_reshape = X_val.reshape(X_val.shape[0]//100, 100, features)

y_train_split_reshape = y_train_split.to_numpy().reshape(y_train_split.shape[0]//100,100,targets)
y_val_reshape = y_val.to_numpy().reshape(y_val.shape[0]//100,100,targets)

validation_data = (X_val_reshape, y_val_reshape)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20) 

print("Best trial:", study.best_trial.params)



