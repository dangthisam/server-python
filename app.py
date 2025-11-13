# /api/index.py

import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import whois
import dns.resolver
from urllib.parse import urlparse
from datetime import datetime

# --- THÊM THƯ VIỆN ĐỂ TẢI MODEL ---
import os
import requests

# --- CẤU HÌNH MODEL ---
# 1. ĐẶT URL DOWNLOAD TRỰC TIẾP FILE .pkl CỦA BẠN VÀO ĐÂY
MODEL_URL = "https://huggingface.co/van09/pkl/resolve/main/random_forest_model.pkl?download=true" # <-- THAY THẾ URL NÀY
# 2. Vercel chỉ cho phép ghi vào /tmp
MODEL_PATH = "/tmp/random_forest_model.pkl"

# Danh sách cột (Giữ nguyên)
TRAINING_COLUMNS = [
    'having_IP_Address', 'URL_Length', 'Shortining_Service', 
    'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix', 
    'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length', 
    'Favicon', 'port', 'HTTPS_token', 'Request_URL', 'URL_of_Anchor', 
    'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL', 
    'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 
    'age_of_domain', 'DNSRecord', 'web_traffic', 'Page_Rank', 
    'Google_Index', 'Links_pointing_to_page', 'Statistical_report'
]

# --- HÀM TẢI MODEL ---
def download_model():
    # Chỉ tải nếu file chưa tồn tại trong /tmp
    if not os.path.exists(MODEL_PATH):
        print(f"Đang tải model từ {MODEL_URL}...")
        try:
            r = requests.get(MODEL_URL, allow_redirects=True)
            r.raise_for_status() # Báo lỗi nếu request hỏng
            with open(MODEL_PATH, 'wb') as f:
                f.write(r.content)
            print("Tải model thành công!")
        except Exception as e:
            print(f"Lỗi khi tải model: {e}")
            return None
    return joblib.load(MODEL_PATH)

# --- KHỞI TẠO FLASK APP ---
# Tên biến BẮT BUỘC là 'app'
app = Flask(__name__)
CORS(app) 

# --- TẢI MODEL KHI KHỞI ĐỘNG (COLD START) ---
print("Khởi động serverless function...")
model = download_model()
print("Model đã được tải.")

# Hàm trích xuất đặc trưng (Giữ nguyên)
def extract_features_from_url(url):
    # (Toàn bộ code hàm extract_features_from_url của bạn ở đây)
    features = {}
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
    except Exception:
        return pd.DataFrame([[0]*len(TRAINING_COLUMNS)], columns=TRAINING_COLUMNS)
    # 1. having_IP_Address
    ip_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
    features['having_IP_Address'] = -1 if re.match(ip_pattern, domain) else 1
    # 2. URL_Length
    if len(url) < 54:
        features['URL_Length'] = 1
    elif 54 <= len(url) <= 75:
        features['URL_Length'] = 0
    else:
        features['URL_Length'] = -1
    # 3. Shortining_Service
    shortening_services = ['bit.ly', 'goo.gl', 't.co', 'tinyurl.com']
    features['Shortining_Service'] = -1 if any(service in domain for service in shortening_services) else 1
    # 4. having_At_Symbol
    features['having_At_Symbol'] = -1 if '@' in url else 1
    # 5. double_slash_redirecting
    features['double_slash_redirecting'] = -1 if '//' in parsed_url.path else 1
    # 6. Prefix_Suffix
    features['Prefix_Suffix'] = -1 if '-' in domain else 1
    # 7. having_Sub_Domain
    dots = domain.count('.')
    if domain.startswith('www.'):
        dots -= 1
    features['having_Sub_Domain'] = -1 if dots > 2 else (0 if dots == 2 else 1)
    # 8. SSLfinal_State
    features['SSLfinal_State'] = 1 if url.startswith('https') else -1
    # 24. age_of_domain & 9. Domain_registeration_length
    try:
        domain_info = whois.whois(domain)
        if domain_info.creation_date:
            creation_date = domain_info.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            age = (datetime.now() - creation_date).days
            features['age_of_domain'] = 1 if age >= 180 else -1
            features['Domain_registeration_length'] = 1 if age >= 365 else -1
        else:
            features['age_of_domain'] = -1
            features['Domain_registeration_length'] = -1
    except Exception:
        features['age_of_domain'] = -1
        features['Domain_registeration_length'] = -1
    # 25. DNSRecord
    try:
        dns.resolver.resolve(domain, 'A')
        features['DNSRecord'] = 1
    except Exception:
        features['DNSRecord'] = -1
    # Các cột khó (Gán giá trị mặc định)
    features['Favicon'] = 1
    features['port'] = 1
    features['HTTPS_token'] = 1
    features['Request_URL'] = 1
    features['URL_of_Anchor'] = 1
    features['Links_in_tags'] = 1
    features['SFH'] = 1
    features['Submitting_to_email'] = 1
    features['Abnormal_URL'] = 1
    features['Redirect'] = 0
    features['on_mouseover'] = 1
    features['RightClick'] = 1
    features['popUpWidnow'] = 1
    features['Iframe'] = 1
    features['web_traffic'] = -1
    features['Page_Rank'] = -1
    features['Google_Index'] = 1
    features['Links_pointing_to_page'] = 1
    features['Statistical_report'] = 1
    # Kết thúc
    feature_df = pd.DataFrame([features])
    feature_df = feature_df.reindex(columns=TRAINING_COLUMNS, fill_value=0)
    return feature_df
    # (Kết thúc hàm extract_features_from_url)

# API Endpoint (Giữ nguyên)
@app.route('/')
def home():
    return "API Phân tích Phishing đang hoạt động (deployed on Vercel)!"

@app.route('/check_url', methods=['POST'])
def check_url():
    if model is None:
        return jsonify({'error': 'Mô hình chưa được tải, kiểm tra log server.'}), 500
    try:
        data = request.json
        if 'url' not in data:
            return jsonify({'error': 'Không tìm thấy "url" trong JSON body'}), 400
        
        url_to_check = data['url']
        print(f"Đang phân tích URL: {url_to_check}")
        features_df = extract_features_from_url(url_to_check)
        prediction = model.predict(features_df)
        result_code = int(prediction[0]) 
        
        status = 'phishing' if result_code == -1 else 'an_toan'
        
        print(f"Kết quả dự đoán: {status} (code: {result_code})")
        return jsonify({
            'url': url_to_check,
            'status': status,
            'prediction_code': result_code
        })
    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán: {e}")
        return jsonify({'error': f'Lỗi máy chủ nội bộ: {str(e)}'}), 500

# --- KHÔNG CÒN app.run() ---
# (Xóa bỏ hoàn toàn if __name__ == '__main__': ...)