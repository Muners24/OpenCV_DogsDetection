import cv2
from gtts import gTTS as gt
import time
import numpy as np
import subprocess
import threading
import os

PAJARO = 3
GATO = 8
VACA = 10
PERRO = 12
CABALLO = 13
OVEJA = 17

ruta = "C:\\Users\\ramon\\Downloads\\OpenCV_DogsDetection\\playAudio.py"
colores = {
    "marrón": 0,
    "roja": 1,
    "blanco": 2,
    "gris": 3,
    "cafe": 4,
    "alazán": 5,
}

# Ruta donde se guardarán los archivos de audio
ruta_audio = "C:\\Users\\ramon\\Downloads\\OpenCV_DogsDetection\\sonidos"

for color, valor in colores.items():
    # Crear texto con el nombre del color
    texto = f"{color}"

    # Establecer el lenguaje
    lenguaje = 'es-us'

    # Crear objeto de texto a voz
    speech = gt(text=texto, lang=lenguaje, slow=True)

    # Guardar el archivo de audio en la ruta especificada
    ruta_guardado = f"{ruta_audio}\\{color}.mp3"
    speech.save(ruta_guardado)
#os.system("C:\\Users\\ramon\\Downloads\\OpenCV_DogsDetection\\label.mp3")
#time.sleep(0.9)
#os.system("C:\\Users\\ramon\\Downloads\\OpenCV_DogsDetection\\color.mp3")
#time.sleep(2)
#os.system("taskkill /F /IM wmplayer.exe")