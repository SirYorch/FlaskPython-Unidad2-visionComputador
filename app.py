# Author: Jorge CUeva && Kevin Yansaguano
# Version: 1.0
# Date: 2025-11-15
# Description: Fork of the original app by Vladimir Robles Bykbaev
# Updated: 2025-11-14

# from flask import Flask, render_template, Response
# we prefer to use fastAPI for simplicity in the documentation
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


from io import BytesIO

import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F
import time

# app = Flask(__name__)
app = FastAPI()

# IP Address
_URL = 'http://172.23.128.215'
# Default Streaming Port
_PORT = '81'
# Default streaming route
_ST = '/stream'
SEP = ':'

stream_url = ''.join([_URL,SEP,_PORT,_ST])


class FPSCounter:
    def __init__(self):
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
    
    def update(self):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        return self.fps


fps_counter = FPSCounter()

# ============== LIGHTING ENHANCEMENT FUNCTIONS ==============

def histogram_equalization(image):
    """Ecualización de histograma estándar"""
    return cv2.equalizeHist(image)

def clahe_enhancement(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """CLAHE - Contrast Limited Adaptive Histogram Equalization"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


##

# EDITAR MÉTODO EXTRA

##


# def gamma_correction(image, gamma=1.5):
#     """Corrección Gamma"""
#     inv_gamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** inv_gamma) * 255 
#                       for i in np.arange(0, 256)]).astype("uint8")
#     return cv2.LUT(image, table)

# ============== NOISE FUNCTIONS ==============



def add_gaussian_noise(image, mean=0, sigma=25):
    """Añade ruido Gaussiano usando solo OpenCV"""
    noise = np.zeros_like(image, dtype=np.int16)
    cv2.randn(noise, mean, sigma)   # llena la matriz con números normales
    noisy = image.astype(np.int16) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_speckle_noise(image, scale=0.1):
    """Añade ruido Speckle usando solo OpenCV"""
    noise = np.zeros_like(image, dtype=np.float32)
    cv2.randn(noise, 0, scale * 255)
    img_f = image.astype(np.float32)
    noisy = img_f + img_f * (noise / 255.0)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """Añade ruido sal y pimienta"""
    noisy = image.copy()
    
    # Salt noise (white pixels)
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1]] = 255
    
    # Pepper noise (black pixels)
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1]] = 0
    
    return noisy

# ============== DENOISING FILTERS ==============


def denoise_pytorch_gaussian(image, ksize=5, sigma=1.0):
    """
    Denoise usando kernel Gaussiano en PyTorch
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    img_tensor = torch.from_numpy(gray.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    
    # Crear kernel Gaussiano
    ax = torch.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, ksize, ksize).float()
    
    # Aplicar convolución
    padding = ksize // 2
    convolved_output = F.conv2d(
        input=img_tensor,
        weight=kernel,
        padding=padding,
        bias=None
    )
    
    result = convolved_output.squeeze().cpu().numpy()
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result



def apply_median_filter(image, ksize=5):
    """
    Aplica filtro de Mediana
    ksize: tamaño de la máscara (debe ser impar: 3, 5, 7, 9, etc.)
    """
    return cv2.medianBlur(image, ksize)


def apply_blur_filter(image, ksize=5):
    """
    Aplica filtro Blur (promedio)
    ksize: tamaño de la máscara (puede ser cualquier número: 3, 5, 7, etc.)
    """
    return cv2.blur(image, (ksize, ksize))


def apply_gaussian_filter(image, ksize=5, sigma=0):
    """
    Aplica filtro Gaussiano
    ksize: tamaño de la máscara (debe ser impar: 3, 5, 7, etc.)
    sigma: desviación estándar (si es 0, se calcula automáticamente)
    """
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)



#============= EDGE DETECTION ==============

def edge_detection_sobel(image, use_blur=False):
    """Detección de bordes Sobel"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    if use_blur:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))
    
    return magnitude

def edge_detection_canny(image, use_blur=False, threshold1=50, threshold2=150):
    """Detección de bordes Canny"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    if use_blur:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(gray, threshold1, threshold2)
    return edges


# ============== MORPHOLOGICAL OPERATIONS ==============

def apply_morphological_operations(image, kernel_size=37):
    """
    Aplica operaciones morfológicas según el artículo:
    "Using morphological transforms to enhance the contrast of medical images"
    """
    # Asegurar que la imagen está en escala de grises
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Crear elemento estructurante (circular es mejor para imágenes médicas)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Erosión
    erosion = cv2.erode(gray, kernel, iterations=1)
    
    # Dilatación
    dilation = cv2.dilate(gray, kernel, iterations=1)
    
    # Top Hat (Original - Opening)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    top_hat = cv2.subtract(gray, opening)
    
    # Black Hat (Closing - Original)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    black_hat = cv2.subtract(closing, gray)
    
    # Enhanced: Original + (Top Hat - Black Hat)
    # Normalizar top_hat y black_hat para evitar overflow
    top_hat_norm = top_hat.astype(np.float32)
    black_hat_norm = black_hat.astype(np.float32)
    
    difference = top_hat_norm - black_hat_norm
    enhanced = gray.astype(np.float32) + difference
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    # Aplicar CLAHE adicional para mejor contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_clahe = clahe.apply(enhanced)
    
    return {
        'original': gray,
        'erosion': erosion,
        'dilation': dilation,
        'top_hat': top_hat,
        'black_hat': black_hat,
        'enhanced': enhanced,
        'enhanced_clahe': enhanced_clahe
    }

def combine_morphological_results(results, kernel_size):
    """Combina los resultados morfológicos en una sola imagen para visualización"""
    h, w = results['original'].shape
    
    # Crear imagen 3x3 (eliminamos enhanced_clahe de la visualización principal)
    combined = np.zeros((h * 3, w * 3), dtype=np.uint8)
    
    # Fila 1
    combined[0:h, 0:w] = results['original']
    combined[0:h, w:2*w] = results['erosion']
    combined[0:h, 2*w:3*w] = results['dilation']
    
    # Fila 2
    combined[h:2*h, 0:w] = results['top_hat']
    combined[h:2*h, w:2*w] = results['black_hat']
    combined[h:2*h, 2*w:3*w] = results['enhanced']
    
    # Fila 3 - Comparaciones
    # Ecualización para comparación
    hist_eq = cv2.equalizeHist(results['original'])
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_result = clahe.apply(results['original'])
    
    combined[2*h:3*h, 0:w] = hist_eq
    combined[2*h:3*h, w:2*w] = clahe_result
    combined[2*h:3*h, 2*w:3*w] = results['enhanced_clahe']
    
    # Convertir a BGR para agregar texto en color
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    
    # Añadir etiquetas
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)
    thickness = 2
    font_scale = 0.7
    
    labels = [
        (10, 30, f'Original (Kernel: {kernel_size}x{kernel_size})'),
        (w+10, 30, 'Erosion'),
        (2*w+10, 30, 'Dilation'),
        (10, h+30, 'Top Hat'),
        (w+10, h+30, 'Black Hat'),
        (2*w+10, h+30, 'Original + (TH - BH)'),
        (10, 2*h+30, 'Histogram Eq.'),
        (w+10, 2*h+30, 'CLAHE'),
        (2*w+10, 2*h+30, 'Enhanced + CLAHE'),
    ]
    
    for x, y, label in labels:
        cv2.putText(combined_bgr, label, (x, y), font, font_scale, color, thickness)
    
    return combined_bgr

# ============== BACKGROUND REMOVAL ==============

def remove_background(frame, fg_mask):
    """Aplica la máscara para extraer solo el foreground"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    foreground = cv2.bitwise_and(frame, frame, mask=fg_mask)
    return foreground, fg_mask

# ============== VIDEO PROCESSING ==============

def video_capture():
    
    res = requests.get(stream_url, stream=True)

    # median_mask = torch.ones(3, 3, dtype = torch.float32)/9.0
    # kernel = median_mask.unsqueeze(0).unsqueeze(0)
    # convolved_output = None
    # img_output = None
    
    #Los eliminamos porque vamos a hacer nosotros los nuestros
    
    
    background_frames = []
    background_median = None
    
    for chunk in res.iter_content(chunk_size=100000):
        if len(chunk) > 100:
    # while True:
            try:
                # time.sleep(0.5)  # Pequeña pausa para evitar sobrecarga
                img_data = BytesIO(chunk)
                
                
                ###Dado que esta es la imagen del esp, para cambiarla usamos una imagen estatica
                
                frame = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                
                ## creamos una imagen para pruebas durante el desarrollo
                
                # frame = cv2.imread('static/image.png')
                
                
                height, width = frame.shape[:2]
                
                fps = fps_counter.update()
                
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # EQUALIZACIÓN
                hist_eq = histogram_equalization(gray)
                
                # CLAHE
                clahe_img = clahe_enhancement(gray)
                
                # TRANSFORMACIÓN LOGARÍTMICA
                c = 255 / np.log(1 + np.max(gray))

                log_transformed = c * np.log(1 + gray.astype(np.float32))

                log_transformed = np.uint8(log_transformed)
                
                
                #frame differencing para eliminación de fondo por estimación
                # Recolectar frames para background
                if len(background_frames) < 25:
                    background_frames.append(gray.copy())
                    continue
                
                # Calcular background (solo una vez)
                if background_median is None:
                    background_median = np.median(background_frames, axis=0).astype(np.uint8)
                
                # Background subtraction
                dframe = cv2.absdiff(gray, background_median)
                _, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)
                
                foreground_color = cv2.bitwise_and(frame, frame, mask=dframe)
                
                gaussiano = add_gaussian_noise(frame)
                
                speckle = add_speckle_noise(frame)
                gaussianobn = add_gaussian_noise(gray)
                
                
                
                denoise1 = denoise_pytorch_gaussian(gaussianobn)
                
                
                mediana = apply_median_filter(frame)
                
                blur = apply_blur_filter(frame)
                
                gaussianoblur = apply_gaussian_filter(frame)
                
                
                
                #bordes canny
                
                canny = edge_detection_canny(gray)
                
                #bordes sobel
                
                sobel = edge_detection_sobel(gray)
                
                
                #bordes canny + suavizado
                
                canny_blur = edge_detection_canny( gaussianoblur)
                
                #bordes sobel + suavizado
                
                sobel_blur = edge_detection_sobel( gaussianoblur)
                

                # Crear imagen combinada
                total_image = np.zeros((height*4, width * 4, 3), dtype=np.uint8)

                # FILA 1: Original, Hist Eq, CLAHE, Log
                total_image[0:height, 0:width] = gray.reshape(height, width, 1).repeat(3, axis=2)
                total_image[0:height, width:width*2] = hist_eq.reshape(height, width, 1).repeat(3, axis=2)
                total_image[0:height, width*2:width*3] = clahe_img.reshape(height, width, 1).repeat(3, axis=2)
                total_image[0:height, width*3:width*4] = log_transformed.reshape(height, width, 1).repeat(3, axis=2)

                # FILA 2: 
                total_image[height:height*2, 0:width] = foreground_color
                total_image[height:height*2, width:width*2] = gaussiano
                total_image[height:height*2, width*2:width*3] = speckle
                total_image[height:height*2, width*3:width*4] = gaussianobn.reshape(height, width, 1).repeat(3, axis=2)


                # FILA 3: 
                total_image[height*2:height*3, 0:width] = denoise1.reshape(height, width, 1).repeat(3, axis=2)
                total_image[height*2:height*3, width:width*2] = mediana
                total_image[height*2:height*3, width*2:width*3] = blur
                total_image[height*2:height*3, width*3:width*4] = gaussianoblur
                
                # FILA 4: 
                total_image[height*3:height*4, 0:width] = canny.reshape(height, width, 1).repeat(3, axis=2)
                total_image[height*3:height*4, width:width*2] = sobel.reshape(height, width, 1).repeat(3, axis=2)
                total_image[height*3:height*4, width*2:width*3] = canny_blur.reshape(height, width, 1).repeat(3, axis=2)
                total_image[height*3:height*4, width*3:width*4] = sobel_blur.reshape(height, width, 1).repeat(3, axis=2)
                
                
                cv2.putText(total_image, f'FPS: {fps:.1f}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                cv2.putText(total_image, f'Original', (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                cv2.putText(total_image, 'Hist', (width+10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                cv2.putText(total_image, 'CLAHE', (width*2+10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                cv2.putText(total_image, 'Logaritmica', (width*3+10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                
                cv2.putText(total_image, 'bg-remove', (10, 30+height), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                cv2.putText(total_image, 'ruido gauss ', (width+10, 30+height), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                cv2.putText(total_image, 'ruido speckle', ((width*2)+10, 30+height), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                cv2.putText(total_image, 'ruido gauss B/N', ((width*3)+10, 30+height), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                
                cv2.putText(total_image, 'denoise gauss', (10, 30+(height*2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                cv2.putText(total_image, 'mediana', ((width)+10, 30+(height*2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                cv2.putText(total_image, 'blur', ((width*2)+10, 30+(height*2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                cv2.putText(total_image, 'gauss blur', ((width*3)+10, 30+(height*2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                
                cv2.putText(total_image, 'Canny', (10, 30+(height*3)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                cv2.putText(total_image, 'Sobel', ((width)+10, 30+(height*3)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                cv2.putText(total_image, 'Canny+blur', ((width*2)+10, 30+(height*3)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                cv2.putText(total_image, 'Sobel+blur', ((width*3)+10, 30+(height*3)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
                
                
                (flag, encodedImage) = cv2.imencode(".jpg", total_image)
                if not flag:
                    continue
                
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                      bytearray(encodedImage) + b'\r\n')
                
                # time.sleep(0.03)
                
            except Exception as e:
                print(f"Error: {e}")
                continue

# ============== FLASK ROUTES ==============
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/partb", response_class=HTMLResponse)
async def morph(request: Request):
    return templates.TemplateResponse("morph.html", {"request": request})

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("trabajo.html", {"request": request})

@app.get("/video_stream")
async def video_stream():
    return StreamingResponse(
        video_capture(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

media = 0
varianza = 0
desv_est = 0
tam = 0

@app.post("/act_media")
async def act_media(dato: float):
    global media
    media = dato
    return {"status": "success", "media": media}


@app.post("/act_varianza")
async def act_varianza(dato: float):
    global varianza
    varianza = dato
    return {"status": "success", "varianza": varianza}


@app.post("/act_desv")
async def act_desv(dato: float):
    global desv_est
    desv_est = dato
    return {"status": "success", "desv_est": desv_est}


@app.post("/act_desv")
async def act_tam(dato: float):
    global tam
    tam = dato
    return {"status": "success", "tamano-mascara": tam}





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)