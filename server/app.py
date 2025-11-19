from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import datetime

app = Flask(__name__)
CORS(app)  # Mengizinkan Cross-Origin Resource Sharing

# --- Pemuatan Model dan Data saat Startup ---
try:
    # PASTIKAN FILE-FILE INI ADA DI DIREKTORI YANG SAMA
    model = joblib.load('linear_regression_model.pkl')
    scaler = joblib.load('minmax_scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    df = pd.read_csv('data_produksi_padi_india.csv')
    
    unique_states = sorted(df['State_Name'].unique())
    unique_seasons = sorted(df['Season'].unique())
    
    state_district_map = {}
    for state in unique_states:
        districts = sorted(df[df['State_Name'] == state]['District_Name'].unique())
        state_district_map[state] = districts

    print("Model, scaler, encoder, dan data dropdown berhasil dimuat.")

except Exception as e:
    print(f"Error saat memuat file: {e}")
    model = None
    state_district_map = {}
    unique_states = []
    unique_seasons = []
# -----------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_dropdown_data', methods=['GET'])
def get_dropdown_data():
    if model is None:
        return jsonify({"error": "Model tidak berhasil dimuat di server."}), 500
        
    return jsonify({
        "states": unique_states,
        "seasons": unique_seasons,
        "state_district_map": state_district_map
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model tidak berhasil dimuat di server."}), 500

    try:
        data = request.json
        state_name = data['State_Name']
        district_name = data['District_Name']
        crop_year = int(data['Crop_Year'])
        season = data['Season']
        area = float(data['Area'])

        # Encoding dan Scaling
        state_encoded = label_encoders['State_Name'].transform([state_name])[0]
        district_encoded = label_encoders['District_Name'].transform([district_name])[0]
        season_encoded = label_encoders['Season'].transform([season])[0]

        area_log = np.log1p(area)
        area_scaled = scaler.transform([[area_log, 0]])[0][0]

        features = np.array([[
            state_encoded,
            district_encoded,
            crop_year,
            season_encoded,
            area_scaled
        ]])

        prediction_scaled = model.predict(features)[0]
        prediction_log = scaler.inverse_transform([[0, prediction_scaled]])[0][1]
        prediction_ton = np.expm1(prediction_log)

        return jsonify({'prediksi_ton': prediction_ton})

    except Exception as e:
        if 'State_Name' not in request.json or 'District_Name' not in request.json:
             return jsonify({'error': f'Input tidak lengkap. Pastikan State, District, Year, Season, dan Area terisi: {str(e)}'}), 400
        
        return jsonify({'error': f'Terjadi kesalahan saat prediksi: {str(e)}'}), 500

@app.route('/get_prediction_chart_data_multi_area', methods=['POST'])
def get_prediction_chart_data_multi_area():
    if model is None:
        return jsonify({"error": "Model tidak berhasil dimuat di server."}), 500

    try:
        data = request.json
        
        # 1. Ambil data filter statis
        state_name = data['State_Name']
        district_name = data['District_Name']
        season = data['Season']
        
        # 2. Ambil data tahun & area dinamis (list of objects)
        year_area_data = data['year_area_data'] 
        
        if not year_area_data:
             return jsonify({'error': 'Data tahun dan area tidak ditemukan.'}), 400

        # 3. Encode data kategorikal (hanya sekali)
        state_encoded = label_encoders['State_Name'].transform([state_name])[0]
        district_encoded = label_encoders['District_Name'].transform([district_name])[0]
        season_encoded = label_encoders['Season'].transform([season])[0]

        predictions = []
        labels = []

        # 4. Loop untuk setiap tahun dan area yang berbeda
        for item in year_area_data:
            year = int(item['year'])
            area = float(item['area'])
            
            labels.append(str(year))
            
            # Transform & Scaling 'Area' untuk area tahun ini
            area_log = np.log1p(area)  
            area_scaled = scaler.transform([[area_log, 0]])[0][0]
            
            # Buat array fitur untuk model
            features = np.array([[
                state_encoded,
                district_encoded,
                year,
                season_encoded,
                area_scaled
            ]])
            
            # Prediksi
            prediction_scaled = model.predict(features)[0]
            prediction_log = scaler.inverse_transform([[0, prediction_scaled]])[0][1]
            prediction_ton = np.expm1(prediction_log)
            predictions.append(max(0, prediction_ton))

        # 5. Hitung persentase perubahan
        percent_changes = [0.0]
        for i in range(1, len(predictions)):
            prev_val = predictions[i-1]
            curr_val = predictions[i]
            
            if prev_val > 0:
                change = ((curr_val - prev_val) / prev_val) * 100
            elif curr_val > 0:
                change = 100.0
            else:
                change = 0.0
            percent_changes.append(change)

        return jsonify({
            'labels': labels,
            'values': predictions,
            'percent_changes': percent_changes
        })

    except KeyError as e:
        return jsonify({'error': f'Filter atau data tahun tidak lengkap: {e} tidak ditemukan.'}), 400
    except Exception as e:
        return jsonify({'error': f'Gagal membuat data chart: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)