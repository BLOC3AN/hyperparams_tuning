import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Add, Activation, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import joblib
# Giả định dữ liệu đầu vào và đầu ra
# Bạn cần thay thế bằng dữ liệu thực tế của mình
# X_train: [4000, 100, 18], y_train: [4000, 100, 1]
# Ở đây, tôi tạo dữ liệu giả để minh họa



# Định nghĩa residual block cho TCN
def residual_block(x, filters, kernel_size, dilation_rate):
    conv1 = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)(x)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)(conv1)
    conv2 = Activation('relu')(conv2)
    return Add()([x, conv2])  # Residual connection

# Xây dựng mô hình TCN
def build_tcn():
    inputs = Input(shape=(100, features))  # 100 timestep, 18 features
    x = Conv1D(128, 1, padding='causal')(inputs)# Điều chỉnh số channels
    for dilation in [1, 2, 4, 8, 16, 32, 64,128]:  # Bao phủ tối đa 100 timestep
        x = residual_block(x, 128, 3, dilation)
    x = Activation('relu')(x)
    outputs = Conv1D(1, 1)(x)  # Dự đoán 1 giá trị tại mỗi timestep
    model = Model(inputs, outputs)
    return model





df = pl.read_csv("~/project_STKENG/biLSTM/AI_model_backup/data/processed/data_train.csv")
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
print(X_scaler.shape)
X_train_split, X_val, y_train_split, y_val = train_test_split(X_scaler,y, test_size= 0.25)

X_train_split_reshape = X_train_split.reshape(X_train_split.shape[0]//100, 100, features)

X_val_reshape = X_val.reshape(X_val.shape[0]//100, 100, features)

y_train_split_reshape = y_train_split.to_numpy().reshape(y_train_split.shape[0]//100,100,targets)
y_val_reshape = y_val.to_numpy().reshape(y_val.shape[0]//100,100,targets)

validation_data = (X_val_reshape, y_val_reshape)

joblib.dump(scaler,'/home/hailt/project_STKENG/biLSTM/hyperparams_tuning/models/tscaler_tcn.pkl')

# Tạo mô hình
model = build_tcn()

# Biên dịch mô hình
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='mse',
              metrics=['mse', 'mae'])

# Huấn luyện mô hình
# Bạn có thể điều chỉnh batch_size và epochs theo nhu cầu
model.fit(X_train_split_reshape, y_train_split_reshape, epochs=1000, batch_size=32, validation_split=0.2,
          callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('/home/hailt/project_STKENG/biLSTM/hyperparams_tuning/models/tcn_model.keras', save_best_only=True),
        ])

