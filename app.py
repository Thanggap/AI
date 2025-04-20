import json
import pandas as pd
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
# Mapping tên hiển thị sang tên cột chuẩn
display_to_column = {
    "Sex": "Sex",
    "Age": "Age",
    "Height": "Height",
    "Overweight/Obese Families": "Overweight_Obese_Family",
    "Consumption of Fast Food": "Consumption_of_Fast_Food",
    "Frequency of Consuming Vegetables": "Frequency_of_Consuming_Vegetables",
    "Number of Main Meals Daily": "Number_of_Main_Meals_Daily",
    "Food Intake Between Meals": "Food_Intake_Between_Meals",
    "Smoking": "Smoking",
    "Liquid Intake Daily": "Liquid_Intake_Daily",
    "Calculation of Calorie Intake": "Calculation_of_Calorie_Intake",
    "Physical Excercise": "Physical_Excercise",
    "Schedule Dedicated to Technology": "Schedule_Dedicated_to_Technology",
    "Type of Transportation Used": "Type_of_Transportation_Used"
}
# Load model, scaler, mapping
svc = joblib.load('svc_model.pkl')  # Lưu mô hình SVC bằng joblib.dump(svc, 'svc_model.pkl')
scaler = joblib.load('scaler.pkl')  # Lưu scaler bằng joblib.dump(scaler, 'scaler.pkl')
with open('mapping.json', encoding='utf-8') as f:
    mapping = json.load(f)

fields = [
    "Sex", "Age", "Height", "Overweight/Obese Families", "Consumption of Fast Food",
    "Frequency of Consuming Vegetables", "Number of Main Meals Daily", "Food Intake Between Meals",
    "Smoking", "Liquid Intake Daily", "Calculation of Calorie Intake", "Physical Excercise",
    "Schedule Dedicated to Technology", "Type of Transportation Used"
]

default_values = {
    "Sex": "Male",
    "Age": "36",
    "Height": "171",
    "Overweight/Obese Families": "No",
    "Consumption of Fast Food": "No",
    "Frequency of Consuming Vegetables": "Sometimes",
    "Number of Main Meals Daily": "3",
    "Food Intake Between Meals": "Sometimes",
    "Smoking": "No",
    "Liquid Intake Daily": "within the range of 1 to 2 liters",
    "Calculation of Calorie Intake": "No",
    "Physical Excercise": "in the range of 3-4 days",
    "Schedule Dedicated to Technology": "between 3 and 5 hours",
    "Type of Transportation Used": "Bike"
}

def reverse_mapping(field):
    if field in mapping:
        return {str(v).lower(): int(k) for k, v in mapping[field].items()}
    return None

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    class_mapping = mapping["Class"]
    options = {field: list(reverse_mapping(field).keys()) if reverse_mapping(field) else [] for field in fields}
    if request.method == 'POST':
        if request.is_json:  # Kiểm tra nếu request là AJAX
            input_data = request.get_json()
            processed_data = {}
            for field in fields:
                val = input_data.get(field, '').strip().lower()
                if val == "" and field in default_values:
                    val = str(default_values[field]).strip().lower()
                rev_map = reverse_mapping(field)
                if rev_map and val in rev_map:
                    processed_data[field] = [rev_map[val]]
                else:
                    processed_data[field] = [0]
            input_df = pd.DataFrame({display_to_column[k]: v for k, v in processed_data.items()})
            input_df = input_df[[display_to_column[f] for f in fields]]
            input_scaled = scaler.transform(input_df)
            pred = svc.predict(input_scaled)[0]
            result = f"Kết quả dự đoán: {pred} - {class_mapping[str(pred)]}"
            return {"result": result}  # Trả về JSON
        else:
            # Xử lý form thông thường
            input_data = {}
            for field in fields:
                val = request.form.get(field, '').strip().lower()
                if val == "" and field in default_values:
                    val = str(default_values[field]).strip().lower()
                rev_map = reverse_mapping(field)
                if rev_map and val in rev_map:
                    input_data[field] = [rev_map[val]]
                else:
                    input_data[field] = [0]
            input_df = pd.DataFrame({display_to_column[k]: v for k, v in input_data.items()})
            input_df = input_df[[display_to_column[f] for f in fields]]
            input_scaled = scaler.transform(input_df)
            pred = svc.predict(input_scaled)[0]
            result = f"Kết quả dự đoán: {pred} - {class_mapping[str(pred)]}"
    return render_template('form.html', fields=fields, options=options, default_values=default_values, result=result)

if __name__ == '__main__':
    app.run(debug=True)
