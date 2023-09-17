from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import joblib
import io

app = Flask(__name__)

# Cargar el modelo entrenado al iniciar la aplicación web
model = joblib.load('modelo_final.pkl')

# Mapeo de predicciones a actividades
prediction_to_activity = {
    1: 'Moving',
    4: 'Stairs',
    6: 'Standing',
    7: 'Sitting',
    8: 'Lying',
}

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener el archivo CSV cargado por el usuario
    uploaded_file = request.files['file']
    
    if uploaded_file.filename != '':
        # Leer el archivo CSV en un DataFrame
        df = pd.read_csv(uploaded_file)

        # Verificar si todas las columnas requeridas están presentes
        required_columns = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
        if not all(col in df.columns for col in required_columns):
            return "El archivo CSV debe contener todas las columnas requeridas."

        # Filtrar el DataFrame para incluir solo las columnas requeridas
        df2 = df[required_columns]

        # Realizar predicciones con el modelo
        predictions = model.predict(df2)

        # Crear una lista de actividades a partir de las predicciones
        activities = [prediction_to_activity[p] for p in predictions]

        # Identificar el inicio y el final de cada actividad
        timeline_events = []
        current_activity = None

        for timestamp, activity in zip(df['timestamp'], activities):
            if activity != current_activity:
                if current_activity is not None:
                    # Si cambió la actividad, agregar el final de la actividad anterior
                    timeline_events.append({'timestamp': timestamp, 'activity': current_activity, 'event': 'fin'})

                # Agregar el inicio de la nueva actividad
                timeline_events.append({'timestamp': timestamp, 'activity': activity, 'event': 'inicio'})
                current_activity = activity

        # Agregar el final de la última actividad si es necesario
        if current_activity is not None:
            timeline_events.append({'timestamp': timestamp, 'activity': current_activity, 'event': 'fin'})

        # Almacenar el DataFrame con las predicciones en un objeto BytesIO
        df['prediction'] = predictions
        df['activity'] = activities
        output = io.BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)

        # Guardar el DataFrame en una variable global para que esté disponible para su descarga
        global generated_csv
        generated_csv = output.getvalue()

        # Renderizar la plantilla HTML con el timeline
        return render_template('timeline.html', timeline_events=timeline_events)

@app.route('/download', methods=['GET'])
def download():
    global generated_csv
    if generated_csv:
        return send_file(
            io.BytesIO(generated_csv),
            mimetype='text/csv',
            as_attachment=True,
            download_name='predictions.csv'
        )
    else:
        return "No hay archivo para descargar."

if __name__ == '__main__':
    app.run(debug=True)


