from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np

# Cargar el modelo entrenado
model = load_model('RNA_model.h5')

# Inicializar la aplicación Flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        inputs = [float(request.form[f'feature{i+1}']) for i in range(30)]

        # Convertir los datos en un numpy array
        inputs = np.array(inputs).reshape(1, -1)  # 1 fila, 30 columnas

        # Realizar predicción
        prediction = model.predict(inputs)[0][0]

        # Determinar el resultado
        result = "Positivo (Cáncer detectado)" if prediction > 0.5 else "Negativo (No se detectó cáncer)"

        # Enviar los datos ingresados junto con el resultado
        return render_template(
            'index.html',
            prediction_text=f'Resultado: {result}',
            confidence_text=f'Confianza: {prediction:.2f}',
            input_data=inputs.tolist()
        )

    except Exception as e:
        return render_template('index.html', error_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
