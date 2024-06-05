import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Membaca data dari file
file_path = 'D:\Kuliah\Semester 4\Metode-Numerik\Student_Performance.csv'
data = pd.read_csv(file_path)

# Menampilkan beberapa baris pertama dari data
data.head()

# Mengambil kolom Hours Studied (TB) dan Performance Index (NT)
X = data[['Hours Studied']].values
y = data['Performance Index'].values

# Membuat model regresi linear
linear_model = LinearRegression()
linear_model.fit(X, y)

# Prediksi menggunakan model linear
y_pred_linear = linear_model.predict(X)

# Menghitung galat RMS untuk model linear
rms_linear = np.sqrt(mean_squared_error(y, y_pred_linear))

# Plot hasil regresi linear
plt.scatter(X, y, color='blue', label='Data asli')
plt.plot(X, y_pred_linear, color='red', label='Regresi Linear')
plt.xlabel('Durasi Waktu Belajar (Jam)')
plt.ylabel('Nilai Ujian')
plt.title('Regresi Linear antara Durasi Waktu Belajar dan Nilai Ujian')
plt.legend()
plt.show()

print(f'Galat RMS untuk Model Linear: {rms_linear}')

# Model Pangkat Sederhana (Metode 2)

# Mengubah data ke dalam bentuk logaritmik
X_log = np.log(X)
y_log = np.log(y)

# Membuat model regresi linear untuk data logaritmik
log_model = LinearRegression()
log_model.fit(X_log, y_log)

# Prediksi menggunakan model pangkat sederhana
y_log_pred = log_model.predict(X_log)
y_pred_power = np.exp(y_log_pred)

# Menghitung galat RMS untuk model pangkat sederhana
rms_power = np.sqrt(mean_squared_error(y, y_pred_power))

# Plot hasil regresi pangkat sederhana
plt.scatter(X, y, color='blue', label='Data asli')
plt.plot(X, y_pred_power, color='green', label='Regresi Pangkat Sederhana')
plt.xlabel('Durasi Waktu Belajar (Jam)')
plt.ylabel('Nilai Ujian')
plt.title('Regresi Pangkat Sederhana antara Durasi Waktu Belajar dan Nilai Ujian')
plt.legend()
plt.show()

print(f'Galat RMS untuk Model Pangkat Sederhana: {rms_power}')
