import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app) 

# --- VARIABEL GLOBAL ---
model = None
scaler = None
df_full_data = None
unique_states = []
unique_seasons = []
state_district_map = {}
# Salinan dari daftar kolom yang diharapkan model (akan diisi di 'try')
model_columns_expected = []

try:
    print("--- Memulai Server ---")
    model = joblib.load('linear_regression_model.pkl')
    scaler = joblib.load('minmax_scaler.pkl')
    
    # --- LOGIKA KOLOM BARU ---
    # Kita tidak lagi memuat 'label_encoders.pkl' untuk nama kolom.
    # Model itu sendiri mungkin menyimpan daftar fiturnya.
    # Kita coba ambil dari model.
    if hasattr(model, 'feature_names_in_'):
        model_columns_expected = [col.strip() for col in model.feature_names_in_]
        print(f"Berhasil memuat {len(model_columns_expected)} nama fitur dari model.")
    else:
        # Jika model tidak punya, kita 'load' label_encoders.pkl
        # TAPI kita anggap itu DAFTAR (LIST)
        print("Model tidak memiliki 'feature_names_in_', mencoba memuat 'label_encoders.pkl' sebagai gantinya...")
        
        # Ini adalah asumsi dari kode *asli* Anda.
        model_columns_original = joblib.load('label_encoders.pkl')
        
        # Jika 'label_encoders.pkl' adalah DICT (seperti error sebelumnya), ambil 'keys'-nya
        if isinstance(model_columns_original, dict):
            print("PERINGATAN: 'label_encoders.pkl' adalah dict. Menggunakan 'keys' sebagai nama kolom.")
            model_columns_expected = [col.strip() for col in model_columns_original.keys()]
            
            # Tambahkan 'Area' dan 'Crop_Year' jika tidak ada
            if 'Area' not in model_columns_expected: model_columns_expected.insert(0, 'Area')
            if 'Crop_Year' not in model_columns_expected: model_columns_expected.insert(1, 'Crop_Year')

        # Jika itu LIST (yang kita harapkan)
        elif isinstance(model_columns_original, list):
            print("Berhasil memuat 'label_encoders.pkl' sebagai list.")
            model_columns_expected = [col.strip() for col in model_columns_original]
        
        else:
            raise TypeError("Tidak bisa menentukan nama kolom model. 'label_encoders.pkl' bukan list atau dict.")
            
        # Asumsi dari error pertama: 'Area' dan 'Crop_Year' mungkin hilang dari file .pkl
        # Jadi kita pastikan mereka ada di model_columns_expected
        if 'Area' not in model_columns_expected:
            print("Menambahkan 'Area' ke daftar kolom.")
            model_columns_expected.append('Area')
        if 'Crop_Year' not in model_columns_expected:
            print("Menambahkan 'Crop_Year' ke daftar kolom.")
            model_columns_expected.append('Crop_Year')

    print(f"Daftar kolom yang diharapkan model (dibersihkan): {model_columns_expected}")
    # --- AKHIR LOGIKA KOLOM BARU ---

    csv_file_path = 'data_produksi_padi_india.csv' 
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"File CSV tidak ditemukan di: {csv_file_path}")

    df_full_data = pd.read_csv(csv_file_path)
    
    print("--- Pengecekan Kolom CSV ---")
    df_full_data.columns = df_full_data.columns.str.strip()
    
    required_csv_cols = ['State_Name', 'District_Name', 'Season', 'Crop_Year', 'Production']
    missing_cols = [col for col in required_csv_cols if col not in df_full_data.columns]
    
    if missing_cols:
        raise ValueError(f"KRITIS: Kolom penting hilang dari file CSV: {missing_cols}")
    
    print("Semua kolom CSV yang dibutuhkan (State_Name, District_Name, Season, Crop_Year, Production) ditemukan.")

    for col in df_full_data.select_dtypes(include=['object']).columns:
        df_full_data[col] = df_full_data[col].str.strip()
    
    df_full_data.dropna(subset=['Production'], inplace=True)

    unique_states = sorted(df_full_data['State_Name'].unique().tolist())
    unique_seasons = sorted(df_full_data['Season'].unique().tolist())
    state_district_map = df_full_data.groupby('State_Name')['District_Name'].unique().apply(lambda x: sorted(x.tolist())).to_dict()

    print("--- Server Siap ---")
    print("Model, scaler, kolom, dan data dropdown/chart berhasil dimuat.")

except Exception as e:
    print(f"!!! KRITIS: Gagal memuat file saat server dimulai !!!")
    print(f"Error: {e}") 
    model = None 

# ... (GET CHART DATA tidak berubah) ...
@app.route('/get_chart_data', methods=['GET'])
def get_chart_data():
    if df_full_data is None:
        return jsonify({'error': 'Data untuk chart tidak tersedia.'}), 500
    
    year_production = df_full_data.groupby('Crop_Year')['Production'].sum()
    
    season_production = df_full_data.groupby('Season')['Production'].sum()

    chart_data = {
        'year_data': {
            'labels': year_production.index.astype(str).tolist(),
            'values': year_production.values.tolist()
        },
        'season_data': {
            'labels': season_production.index.tolist(),
            'values': season_production.values.tolist()
        }
    }
    return jsonify(chart_data)


@app.route('/get_dropdown_data', methods=['GET'])
def get_dropdown_data():
    # Cek 'unique_states' yang diisi saat startup
    if not unique_states: 
        return jsonify({'error': 'Data dropdown tidak tersedia karena server gagal memuat file.'}), 500
    
    return jsonify({
        'states': unique_states,
        'seasons': unique_seasons,
        'state_district_map': state_district_map
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Server tidak siap untuk prediksi. Periksa log error di terminal.'}), 500

    try:
        data = request.get_json()
        print(f"Menerima data untuk prediksi: {data}")
        
        for key in ['State_Name', 'District_Name', 'Season']:
            if key in data:
                data[key] = data[key].strip()

        # --- PERBAIKAN: Proses 'Area' ---
        area_value = float(data['Area'])
        scaled_area = scaler.transform([[area_value, 0]])[0][0]
        data['Area'] = scaled_area
        
        data['Crop_Year'] = float(data['Crop_Year'])

        input_df = pd.DataFrame([data])
        
        # --- LOGIKA PREDIKSI BARU (LEBIH SEDERHANA) ---
        
        # 1. Buat Dummies. Kolom numerik ('Area', 'Crop_Year') akan diabaikan
        input_encoded = pd.get_dummies(input_df)
        
        # 2. Bersihkan NAMA KOLOM HASIL DUMMIES
        input_encoded.columns = [col.strip() for col in input_encoded.columns]

        # 3. Reindex
        #    'columns' akan diisi dengan daftar BERSIH yang kita buat saat startup
        #    'fill_value=0' akan menangani distrik/state yang tidak ada di input
        final_df = input_encoded.reindex(columns=model_columns_expected, fill_value=0)
        
        # 4. Prediksi
        #    Kita tidak perlu 'rename' kolom lagi.
        prediction_scaled = model.predict(final_df) 
        
        # --- AKHIR LOGIKA PREDIKSI BARU ---
        
        prediction_unscaled = scaler.inverse_transform([[0, prediction_scaled[0]]])[0][1]
        
        print(f"Prediksi berhasil: {prediction_unscaled} Ton")
        return jsonify({'prediksi_ton': prediction_unscaled})

    except Exception as e:
        print(f"Error saat proses prediksi: {e}")
        return jsonify({'error': f'Terjadi kesalahan di server: {e}'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)