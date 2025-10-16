💬 PSentimientos — Análisis de Sentimientos en Español

PSentimientos es un modelo de Machine Learning diseñado para analizar el sentimiento de frases en español, determinando si su tono general es positivo o negativo.

El proyecto fue desarrollado inicialmente en Google Colab, donde se entrenó, validó y evaluó el modelo con técnicas de procesamiento de lenguaje natural (NLP).
Posteriormente, fue optimizando para producción, construyendo un servicio Flask API desplegado en Render y conectado a la landing page personal como parte de la sección de proyectos de Machine Learning.

🚀 Descripción General

Objetivo: Clasificar oraciones o reseñas según su polaridad emocional (positiva o negativa).

Modelo: Logistic Regression entrenado con características vectorizadas mediante TF-IDF.

Entrenamiento: Google Colab.

Despliegue: Flask API en Render.

Consumo: Frontend React (landing personal).

⚙️ Arquitectura del Proyecto
[Frontend React (Landing)]  →  [API Flask en Render]  →  [Modelo LogisticRegression + TF-IDF]


El usuario ingresa una frase en el formulario del portafolio.

La aplicación envía el texto a la API mediante un POST /predict.

El modelo preprocesa el texto y devuelve el sentimiento estimado junto a un valor de confianza.

🧠 Ejemplo de Uso
Solicitud (JSON)
{
  "text": "Me encantó el servicio, todo fue rápido y excelente."
}

Respuesta (JSON)
{
  "sentiment": "positivo",
  "confidence": 0.92
}

🧩 Tecnologías Utilizadas
Componente	Tecnología
Lenguaje principal	Python 3.10
Framework backend	Flask
Librerías ML	scikit-learn, joblib, pandas, nltk
Vectorización	TF-IDF
Despliegue	Render
Control de versiones	GitHub
📦 Instalación Local
# 1. Clonar el repositorio
git clone https://github.com/Rudeluy/psentimientos-backend.git
cd psentimientos-backend

# 2. Crear entorno virtual
python -m venv .venv
# Activar entorno
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar servidor
python app.py
# Servidor disponible en: http://127.0.0.1:5000

🌐 API Pública

El servicio está activo y disponible para pruebas:

🔗 https://psentimientos-backend.onrender.com

Puedes probarlo con herramientas como Postman o cURL:

curl -X POST https://psentimientos-backend.onrender.com/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "No me gustó la atención, fue muy mala."}'


Respuesta esperada:

{
  "sentiment": "negativo",
  "confidence": 0.87
}

📈 Resultados de Entrenamiento (Referencia)

Accuracy: 0.88

Precision: 0.87

Recall: 0.89

F1-score: 0.88

Validación: Train/Test split con datos en español.

🔗 Enlaces Relacionados

🌐 API en Render: https://psentimientos-backend.onrender.com

💻 Código Fuente: https://github.com/Rudeluy/psentimientos-backend

👨‍💻 Autor

Luis Véliz S.
Ingeniero Informático especializado en QA Funcional y Data Science.
Proyecto parte del portafolio personal — Machine Learning aplicado a NLP (Procesamiento de Lenguaje Natural).
