import io
import base64
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Plantilla HTML
TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Procesamiento de Imágenes</title>
</head>
<body>
    <h2>Subir imagen</h2>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Procesar</button>
    </form>

    {% if atributos %}
        <h3>Atributos con PIL</h3>
        <ul>
            <li><b>Formato:</b> {{ atributos.pil.formato }}</li>
            <li><b>Tamaño:</b> {{ atributos.pil.tamaño }}</li>
            <li><b>Modo de color:</b> {{ atributos.pil.modo }}</li>
        </ul>

        <h3>Atributos con OpenCV</h3>
        <ul>
            <li><b>Dimensiones:</b> {{ atributos.cv.ancho }}x{{ atributos.cv.alto }}</li>
            <li><b>Canales de color:</b> {{ atributos.cv.canales }}</li>
        </ul>

        <h3>Estadísticas por canal</h3>
        <ul>
            <li><b>Azul:</b> {{ atributos.stats.azul }}</li>
            <li><b>Verde:</b> {{ atributos.stats.verde }}</li>
            <li><b>Rojo:</b> {{ atributos.stats.rojo }}</li>
        </ul>

        <h3>Imagen original</h3>
        <img src="data:image/png;base64,{{ atributos.imagen_base64 }}" width="400">

        <h3>Histogramas</h3>
        <img src="data:image/png;base64,{{ atributos.histograma_base64 }}">
    {% endif %}
</body>
</html>
"""

def procesar_imagen(file_bytes):
    # -------- PIL --------
    imagen_pil = Image.open(io.BytesIO(file_bytes))
    atributos_pil = {
        "formato": imagen_pil.format,
        "tamaño": f"{imagen_pil.size[0]}x{imagen_pil.size[1]}",
        "modo": imagen_pil.mode
    }

    # -------- OpenCV --------
    arr = np.frombuffer(file_bytes, np.uint8)
    imagen_cv = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    alto, ancho, canales = imagen_cv.shape
    atributos_cv = {"ancho": ancho, "alto": alto, "canales": canales}

    # -------- Estadísticas por canal --------
    canal_azul = imagen_cv[:, :, 0]
    canal_verde = imagen_cv[:, :, 1]
    canal_rojo = imagen_cv[:, :, 2]

    stats = {
        "azul": f"min={np.min(canal_azul)}, max={np.max(canal_azul)}, "
                f"media={np.mean(canal_azul):.2f}, desvío={np.std(canal_azul):.2f}",
        "verde": f"min={np.min(canal_verde)}, max={np.max(canal_verde)}, "
                 f"media={np.mean(canal_verde):.2f}, desvío={np.std(canal_verde):.2f}",
        "rojo": f"min={np.min(canal_rojo)}, max={np.max(canal_rojo)}, "
                f"media={np.mean(canal_rojo):.2f}, desvío={np.std(canal_rojo):.2f}",
    }

    # -------- Histogramas --------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.hist(canal_azul.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
    plt.title('Histograma Azul')
    plt.subplot(1, 3, 2)
    plt.hist(canal_verde.ravel(), bins=256, range=[0, 256], color='green', alpha=0.7)
    plt.title('Histograma Verde')
    plt.subplot(1, 3, 3)
    plt.hist(canal_rojo.ravel(), bins=256, range=[0, 256], color='red', alpha=0.7)
    plt.title('Histograma Rojo')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    histograma_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # -------- Imagen original a base64 --------
    _, buffer = cv2.imencode('.png', imagen_cv)
    imagen_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "pil": atributos_pil,
        "cv": atributos_cv,
        "stats": stats,
        "imagen_base64": imagen_base64,
        "histograma_base64": histograma_base64,
    }

@app.route("/", methods=["GET", "POST"])
def index():
    atributos = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_bytes = file.read()  
            atributos = procesar_imagen(file_bytes)
    return render_template_string(TEMPLATE, atributos=atributos)

if __name__ == "__main__":
    app.run(debug=False)
