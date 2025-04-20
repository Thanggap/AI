import json
from django.http import JsonResponse

def predict_view(request):
    if request.method == "POST":
        # Lấy dữ liệu từ request
        data = json.loads(request.body)
        # Xử lý logic dự đoán
        result = "Kết quả dự đoán: ..."  # Thay bằng logic thực tế
        return JsonResponse({"result": result})
    return JsonResponse({"error": "Invalid request"}, status=400)