import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app) 

try:
    model = joblib.load('linear_regression_model.pkl')
    scaler = joblib.load('minmax_scaler.pkl')
    model_columns_original = joblib.load('model_columns.pkl') 
    model_columns_cleaned = [col.strip() for col in model_columns_original]

    excel_file_path = 'rice_2013_2014.xlsx' 

    if not os.path.exists(excel_file_path):
        raise FileNotFoundError(f"File Excel tidak ditemukan di: {excel_file_path}")

    df_full_data = pd.read_excel(excel_file_path)
    
    df_full_data.columns = df_full_data.columns.str.strip()
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
    df_full_data = None
    unique_states, unique_seasons, state_district_map = [], [], {}

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

        area_value = float(data['Area'])
        scaled_area = scaler.transform([[area_value, 0]])[0][0]
        data['Area'] = scaled_area
        
        input_df = pd.DataFrame([data])
        input_encoded = pd.get_dummies(input_df)
        input_encoded.columns = [col.strip() for col in input_encoded.columns]

        final_df_cleaned = input_encoded.reindex(columns=model_columns_cleaned, fill_value=0)
        
        final_df_cleaned.columns = model_columns_original
        
        prediction_scaled = model.predict(final_df_cleaned)
        
        prediction_unscaled = scaler.inverse_transform([[0, prediction_scaled[0]]])[0][1]
        
        print(f"Prediksi berhasil: {prediction_unscaled} Ton")
        return jsonify({'prediksi_ton': prediction_unscaled})

    except Exception as e:
        print(f"Error saat proses prediksi: {e}")
        return jsonify({'error': f'Terjadi kesalahan di server: {e}'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)

