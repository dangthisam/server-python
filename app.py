# app.py (PHIÊN BẢN ĐẦY ĐỦ)

import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- CÁC THƯ VIỆN MỚI CẦN THIẾT ---
import re
import whois
import dns.resolver
from urllib.parse import urlparse
from datetime import datetime

# --- 1. (ĐÃ SỬA) DANH SÁCH CỘT CHÍNH XÁC ---
# Lấy từ file phishing_dataset.csv của bạn
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

# --- 2. (ĐÃ SỬA) HÀM TRÍCH XUẤT ĐẶC TRƯNG ---
def extract_features_from_url(url):
    """
    Hàm này nhận 1 URL string, tính toán các đặc trưng,
    và trả về một DataFrame 1 hàng với các cột y hệt lúc train.
    """
    features = {}
    
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
    except Exception:
        # Nếu URL không hợp lệ, trả về vector 0
        return pd.DataFrame([[0]*len(TRAINING_COLUMNS)], columns=TRAINING_COLUMNS)

    # 1. having_IP_Address
    ip_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
    features['having_IP_Address'] = -1 if re.match(ip_pattern, domain) else 1 # Sửa: -1 nếu là IP
    
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
    # Logic này cần phức tạp hơn (kiểm tra issuer, tuổi SSL)
    # Tạm thời dựa trên HTTPS và tuổi domain
    features['SSLfinal_State'] = 1 if url.startswith('https') else -1
    
    # 24. age_of_domain
    # 9. Domain_registeration_length (Thường là 1 đặc trưng, gộp với age_of_domain)
    try:
        domain_info = whois.whois(domain)
        if domain_info.creation_date:
            creation_date = domain_info.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            
            age = (datetime.now() - creation_date).days
            features['age_of_domain'] = 1 if age >= 180 else -1 # 1 nếu > 6 tháng
            features['Domain_registeration_length'] = 1 if age >= 365 else -1 # 1 nếu > 1 năm
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

    # CÁC ĐẶC TRƯNG KHÓ LẤY (Tạm thời gán giá trị mặc định)
    # Đây là các đặc trưng cần API trả phí (ví dụ: Alexa, Google API)
    # Chúng ta sẽ gán chúng bằng 0 hoặc -1 (giá trị phổ biến nhất)
    # Bạn nên kiểm tra file CSV để xem giá trị nào (1, 0, -1) là phổ biến nhất
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
    features['web_traffic'] = -1 # Rất quan trọng, -1 là traffic thấp (nghi ngờ)
    features['Page_Rank'] = -1
    features['Google_Index'] = 1
    features['Links_pointing_to_page'] = 1
    features['Statistical_report'] = 1
    
    # --- Kết thúc ---
    feature_df = pd.DataFrame([features])
    
    # Reindex để đảm bảo các cột khớp 100%
    # Bất kỳ cột nào không được tính ở trên sẽ được gán giá trị 0
    feature_df = feature_df.reindex(columns=TRAINING_COLUMNS, fill_value=0)
    
    return feature_df

# --- PHẦN KHỞI TẠO FLASK SERVER (Giữ nguyên) ---

app = Flask(__name__)
CORS(app) 

try:
    print("Đang tải mô hình AI...")
    model = joblib.load('random_forest_model.pkl')
    print("Tải mô hình thành công!")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    model = None

@app.route('/')
def home():
    return "Chào mừng đến với API Phân tích Phishing! Hãy POST đến /check_url."

# --- 3. (ĐÃ SỬA) API ENDPOINT ---
@app.route('/check_url', methods=['POST'])
def check_url():
    if model is None:
        return jsonify({'error': 'Mô hình chưa được tải, kiểm tra log server.'}), 500

    try:
        data = request.json
        if 'url' not in data:
            return jsonify({'error': 'Không tìm thấy "url" trong JSON body'}), 400
        
        url_to_check = data['url']
        
        # 1. Trích xuất đặc trưng (Sẽ gọi hàm MỚI)
        print(f"Đang phân tích URL: {url_to_check}")
        features_df = extract_features_from_url(url_to_check)
        
        # 2. Dùng mô hình dự đoán
        prediction = model.predict(features_df)
        
        # 3. Lấy kết quả
        result_code = int(prediction[0]) 
        
        # (Kiểm tra lại file CSV của bạn: 1 là lừa đảo, -1 là an toàn)
        status = 'phishing' if result_code == -1 else 'an_toan' 
        # LƯU Ý: Dựa trên file CSV, -1 MỚI LÀ PHISHING.
        # Nếu mô hình của bạn học theo đó, thì:
        if result_code == -1:
            status = 'phishing'
        else: # result_code == 1
            status = 'an_toan'

        print(f"Kết quả dự đoán: {status} (code: {result_code})")
        
        # 4. Trả kết quả
        return jsonify({
            'url': url_to_check,
            'status': status,
            'prediction_code': result_code
        })

    except Exception as e:
        # In lỗi chi tiết ra terminal
        print(f"Lỗi trong quá trình dự đoán: {e}") 
        return jsonify({'error': f'Lỗi máy chủ nội bộ: {str(e)}'}), 500

# Chạy server
if __name__ == '__main__':
    app.run(debug=True, port=5000)