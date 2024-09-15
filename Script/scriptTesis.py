# Importamos las librerias
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import cv2
import os

# Ruta de la carpeta de entrada y salida
carpeta_entrada = "ruta_imagen_entrada"
carpeta_salida = "./"

# Leer nuestro modelo
model = YOLO("./recorte.pt")

# Procesar cada archivo en la carpeta de entrada
# Cargar la imagen
def leerimagen():
    frame = cv2.imread(carpeta_entrada)
    return frame

def update(val):
    global resultadoo_hsv, s_lower_hue, ax_dynamic, area_afectada, area_total

    resultadoo_hsv = resultadoo_hsv_original.copy()
    # Asegúrate de que los límites sean de tipo np.uint8
    lower_green = np.array([s_lower_hue.val, 40, 40], dtype=np.uint8)
    upper_green = np.array([179, 255, 255], dtype=np.uint8)
    
    # Crear una máscara que filtre el verde, dejando el resto
    mask_green = cv2.inRange(resultadoo_hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask_green)
    
    # Aplicar máscara a la imagen HSV y convertirla a RGB para mostrar
    image_lesions = cv2.bitwise_and(resultadoo_hsv, resultadoo_hsv, mask=mask_inv)
    image_lesions_rgb = cv2.cvtColor(image_lesions, cv2.COLOR_HSV2RGB)
    area_afectada = np.count_nonzero(np.any(image_lesions_rgb > 0, axis=-1))
    # Actualizar la imagen en el subplot dinámico
    ax_dynamic.clear()
    ax_dynamic.imshow(image_lesions_rgb)
    ax_dynamic.set_title('Ajuste Dinámico')
    ax_dynamic.text(10, 4500, f'Área Afectada: {area_afectada} px\n Severidad: {(area_afectada/area_total)*100} %', fontsize=12, color='white', backgroundcolor='black')  # Asegúrate de ajustar las coordenadas
    plt.draw()

def segment_image_with_grabcut(image_path):
    global ix, iy, x, y, drawing, img_show

    drawing = False
    ix, iy = -1, -1
    x, y = -1, -1

    # Cargar la imagen
    img = cv2.imread(image_path)

    # Redimensionar la imagen para que sea más pequeña (opcional)
    scale_percent = 100
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    img_show = img.copy()

    # Aplicar GrabCut con un rectángulo predefinido (ajustar según sea necesario)
    rect = (50, 50, img.shape[1] - 100, img.shape[0] - 100)  # Rectángulo de ejemplo

    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented_image = img * mask_binary[:, :, np.newaxis]

    return segmented_image

frame = leerimagen()
H, W, _ = frame.shape

resultados = model.predict(frame, imgsz=650, conf=0.85)

# Verificar si hay resultados válidos antes de continuar
if resultados:
    # Crear una máscara en blanco del mismo tamaño que la imagen original
    maskk = np.zeros((H, W), dtype=np.uint8)

    # Combinar todas las máscaras de resultados en una sola máscara
    for resultado in resultados:
        if resultado.masks is not None:  # Verificar que haya máscaras válidas
            for j, mask in enumerate(resultado.masks.data):
                mask = (mask.numpy() * 255).astype(np.uint8)
                mask = cv2.resize(mask, (W, H))
                maskk = cv2.bitwise_or(maskk, mask)

    # Aplicar la máscara a la imagen original
    resultadooo = cv2.bitwise_and(frame, frame, mask=maskk)
    area_total = cv2.countNonZero(maskk) 
    mascara1 = maskk

    # Construir el nombre del archivo de salida
    nombre_salida = "segmented_.jpg"

    # Ruta de la imagen de salida
    imagen_salida = os.path.join(carpeta_salida, nombre_salida)

    cv2.imwrite(imagen_salida, resultadooo)
    lista = segment_image_with_grabcut(imagen_salida)
    if lista is not None:
        cv2.imwrite(imagen_salida, lista) 

model = YOLO("./pustula.pt")

resultados = model.predict('./segmented_.jpg', imgsz=650, conf=0.05)

if resultados:
    maskk = np.zeros((H, W), dtype=np.uint8)
    # Combinar todas las máscaras de resultados en una sola máscara
    for resultado in resultados:
        if resultado.masks is not None:
            for j, mask in enumerate(resultado.masks.data):
                mask = (mask.numpy() * 255).astype(np.uint8)
                mask = cv2.resize(mask, (W, H))
                maskk = cv2.bitwise_or(maskk, mask)
    resultadoo = cv2.bitwise_and(frame, frame, mask=maskk)

# Convertir la imagen resultado a HSV
area_afectada = cv2.countNonZero(maskk)
area_afectada_global = area_afectada
resultadoo_hsv = cv2.cvtColor(resultadoo, cv2.COLOR_BGR2HSV)
resultadoo_hsv_original = resultadoo_hsv

# Crear figura y ejes para los sliders y las imágenes
fig, axes = plt.subplots(1, 3, figsize=(20, 10))
plt.subplots_adjust(left=0.25, bottom=0.25)

ax_static1, ax_static2, ax_dynamic = axes

# Configurar los ejes para los sliders de HSV
axcolor = 'lightgoldenrodyellow'
ax_lower_hue = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

# Crear los sliders
s_lower_hue = Slider(ax_lower_hue, 'Lower Hue', 10, 60, valinit=60)

ax_static1.imshow(cv2.cvtColor(cv2.imread(carpeta_entrada), cv2.COLOR_BGR2RGB))
ax_static1.set_title('Imagen original')
ax_static1.axis('off')

ax_static2.imshow(cv2.cvtColor(resultadooo, cv2.COLOR_BGR2RGB))
ax_static2.set_title('Segmentación Hoja')
ax_static2.axis('off')

# Añadir texto para área total y área afectada
ax_static2.text(10, 4500, f'Área Total: {area_total} px', fontsize=12, color='white', backgroundcolor='black')
ax_dynamic.text(10, 4500, f'Área Afectada: {area_afectada} px\n Severidad: {(area_afectada/area_total)*100} %', fontsize=12, color='white', backgroundcolor='black')

# Placeholder para la imagen dinámica
ax_dynamic.imshow(cv2.cvtColor(resultadoo, cv2.COLOR_BGR2RGB))
ax_dynamic.set_title('Ajuste Dinámico')
ax_dynamic.axis('off')

# Registrar la función de actualización con cada deslizador
s_lower_hue.on_changed(update)

plt.show()
