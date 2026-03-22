from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load model & scaler
model = pickle.load(open("models/ckd_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

html = """
<!DOCTYPE html>
<html>
<head>
<title>CKD Prediction</title>
<style>
body {
    font-family: Arial;
    background: #f4f7fa;
}

.container {
    width: 400px;
    margin: auto;
    margin-top: 50px;
    padding: 25px;
    background: white;
    border-radius: 10px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
}

h2 {
    text-align: center;
}

input {
    width: 100%;
    padding: 8px;
    margin: 5px 0 15px 0;
}

button {
    width: 100%;
    padding: 10px;
    background: blue;
    color: white;
    border: none;
}

.result {
    margin-top: 15px;
    text-align: center;
    font-weight: bold;
}
</style>
</head>

<body>
<div class="container">
<h2>🩺 CKD Prediction</h2>

<form method="post">
<input type="number" step="any" name="age" placeholder="Age" required>
<input type="number" step="any" name="bmi" placeholder="BMI" required>
<input type="number" step="any" name="sys_bp" placeholder="Systolic BP" required>
<input type="number" step="any" name="dia_bp" placeholder="Diastolic BP" required>
<input type="number" step="any" name="creatinine" placeholder="Creatinine" required>
<input type="number" step="any" name="bun" placeholder="BUN" required>
<input type="number" step="any" name="gfr" placeholder="GFR" required>
<input type="number" step="any" name="hb" placeholder="Hemoglobin" required>
<input type="number" step="any" name="sugar" placeholder="Sugar" required>

<button type="submit">Predict</button>
</form>

{% if result %}
<div class="result">{{ result }}</div>
{% endif %}

</div>
</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def home():
    result = None

    if request.method == "POST":
        try:
            data = np.array([
                float(request.form["age"]),
                float(request.form["bmi"]),
                float(request.form["sys_bp"]),
                float(request.form["dia_bp"]),
                float(request.form["creatinine"]),
                float(request.form["bun"]),
                float(request.form["gfr"]),
                float(request.form["hb"]),
                float(request.form["sugar"]),
            ]).reshape(1,-1)

            # SCALE DATA (VERY IMPORTANT)
            data = scaler.transform(data)

            pred = model.predict(data)

            if pred[0] == 1:
                result = "⚠️ High Risk of CKD"
            else:
                result = "✅ Normal"

        except Exception as e:
            result = f"Error: {e}"

    return render_template_string(html, result=result)

if __name__ == "__main__":
    app.run(debug=True)