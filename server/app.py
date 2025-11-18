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
    # Muat model, scaler, dan encoder yang sudah dilatih
    model = joblib.load('linear_regression_model.pkl')
    scaler = joblib.load('minmax_scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')

    # Muat dataset asli HANYA untuk mendapatkan daftar unik untuk dropdown
    # Ganti 'data_produksi_padi_india.csv' dengan path yang benar jika perlu
    df = pd.read_csv('data_produksi_padi_india.csv')
    
    # Buat daftar unik
    unique_states = sorted(df['State_Name'].unique())
    unique_seasons = sorted(df['Season'].unique())
    
    # Buat pemetaan State -> District
    state_district_map = {}
    for state in unique_states:
        districts = sorted(df[df['State_Name'] == state]['District_Name'].unique())
        state_district_map[state] = districts

    print("Model, scaler, encoder, dan data dropdown berhasil dimuat.")

except FileNotFoundError as e:
    print(f"Error: File model/data tidak ditemukan. {e}")
    print("Pastikan file '.pkl' dan '.csv' berada di direktori yang sama dengan app.py")
    # Set data ke None agar aplikasi tahu ada masalah
    model = None
    state_district_map = {}
    unique_states = []
    unique_seasons = []
except Exception as e:
    print(f"Error saat memuat model: {e}")
    model = None
    state_district_map = {}
    unique_states = []
    unique_seasons = []
# -----------------------------------------------

@app.route('/')
def index():
    """Menyajikan halaman HTML utama."""
    return render_template('index.html')

@app.route('/get_dropdown_data', methods=['GET'])
def get_dropdown_data():
    """Mengirimkan data untuk mengisi form dropdown."""
    if model is None:
        return jsonify({"error": "Model tidak berhasil dimuat di server."}), 500
        
    return jsonify({
        "states": unique_states,
        "seasons": unique_seasons,
        "state_district_map": state_district_map
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Memprediksi satu data point dari form utama."""
    if model is None:
        return jsonify({"error": "Model tidak berhasil dimuat di server."}), 500

    try:
        data = request.json
        
        # 1. Ambil data input
        state_name = data['State_Name']
        district_name = data['District_Name']
        crop_year = int(data['Crop_Year'])
        season = data['Season']
        area = float(data['Area'])

        # 2. Encode data kategorikal
        state_encoded = label_encoders['State_Name'].transform([state_name])[0]
        district_encoded = label_encoders['District_Name'].transform([district_name])[0]
        season_encoded = label_encoders['Season'].transform([season])[0]

        # 3. Transformasi logaritmik pada 'Area' (seperti saat training)
        area_log = np.log1p(area)

        # 4. Scaling 'Area'
        # Scaler dilatih pada [Area, Production], jadi kita beri dummy 0 untuk Production
        area_scaled = scaler.transform([[area_log, 0]])[0][0]

        # 5. Siapkan fitur untuk model
        # Urutan harus SAMA PERSIS seperti saat training
        # Berdasarkan notebook: 'State_Name', 'District_Name', 'Crop_Year', 'Season', 'Area'
        features = np.array([[
            state_encoded,
            district_encoded,
            crop_year,
            season_encoded,
            area_scaled
        ]])

        # 6. Lakukan prediksi (hasilnya masih ter-scale)
        prediction_scaled = model.predict(features)[0]

        # 7. Inverse scaling pada hasil prediksi
        # Kita beri dummy 0 untuk Area agar bisa inverse transform
        prediction_log = scaler.inverse_transform([[0, prediction_scaled]])[0][1]

        # 8. Inverse transformasi logaritmik (expm1 adalah kebalikan dari log1p)
        prediction_ton = np.expm1(prediction_log)

        return jsonify({'prediksi_ton': prediction_ton})

    except KeyError as e:
        return jsonify({'error': f'Input tidak lengkap: {e} tidak ditemukan.'}), 400
    except ValueError as e:
        return jsonify({'error': f'Input tidak valid: {e}. Pastikan angka diisi dengan benar.'}), 400
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan saat prediksi: {str(e)}'}), 500

@app.route('/get_prediction_chart_data', methods=['POST'])
def get_prediction_chart_data():
    """
    Endpoint BARU: Membuat prediksi untuk rentang tahun 
    berdasarkan filter yang diberikan.
    """
    if model is None:
        return jsonify({"error": "Model tidak berhasil dimuat di server."}), 500

    try:
        data = request.json
        
        # 1. Ambil data filter
        state_name = data['State_Name']
        district_name = data['District_Name']
        season = data['Season']
        area = float(data['Area'])
        
        # Dapatkan tahun saat ini
        current_year = datetime.datetime.now().year
        # Tentukan rentang tahun (misal: 2016 s/d tahun ini)
        START_YEAR = 2016
        END_YEAR = current_year
        YEARS_TO_PREDICT = list(range(START_YEAR, END_YEAR + 1))

        # 2. Encode data kategorikal (hanya sekali)
        state_encoded = label_encoders['State_Name'].transform([state_name])[0]
        district_encoded = label_encoders['District_Name'].transform([district_name])[0]
        season_encoded = label_encoders['Season'].transform([season])[0]

        # 3. Transform & Scaling 'Area' (hanya sekali)
        area_log = np.log1p(area)
        area_scaled = scaler.transform([[area_log, 0]])[0][0]

        predictions = []
        
        # 4. Loop untuk setiap tahun dan lakukan prediksi
        for year in YEARS_TO_PREDICT:
            features = np.array([[
                state_encoded,
                district_encoded,
                year,
                season_encoded,
                area_scaled
            ]])
            
            prediction_scaled = model.predict(features)[0]
            prediction_log = scaler.inverse_transform([[0, prediction_scaled]])[0][1]
            prediction_ton = np.expm1(prediction_log)
            
            # Atasi nilai negatif jika model memprediksi di bawah 0
            predictions.append(max(0, prediction_ton))

        # 5. Hitung persentase perubahan
        percent_changes = [0.0]  # Tahun pertama tidak ada perubahan
        for i in range(1, len(predictions)):
            prev_val = predictions[i-1]
            curr_val = predictions[i]
            
            if prev_val > 0:
                change = ((curr_val - prev_val) / prev_val) * 100
            elif curr_val > 0:
                change = 100.0  # Dari 0 ke >0 dianggap naik 100%
            else:
                change = 0.0 # Dari 0 ke 0
                
            percent_changes.append(change)

        return jsonify({
            'labels': [str(y) for y in YEARS_TO_PREDICT],
            'values': predictions,
            'percent_changes': percent_changes
        })

    except KeyError as e:
        return jsonify({'error': f'Filter tidak lengkap: {e} tidak ditemukan.'}), 400
    except Exception as e:
        return jsonify({'error': f'Gagal membuat data chart: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)