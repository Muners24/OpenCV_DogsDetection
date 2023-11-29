import cv2
from gtts import gTTS as gt
import time
import numpy as np
import subprocess
import threading
import pygame

pygame.init()

def reproducir_audioaudio(label,color):
    def reproducir():
        for i in range(len(label)):
            ruta_label = f"C:\\Users\\ramon\\Downloads\\OpenCV_DogsDetection\\sonidos\\animales\\{label[i]}.mp3"
            ruta_color = f"C:\\Users\\ramon\\Downloads\\OpenCV_DogsDetection\\sonidos\\colores\\{color[i]}.mp3"
            
            pygame.mixer.init()

            pygame.mixer.music.load(ruta_label)
            pygame.mixer.music.play()
            pygame.time.delay(900)  # Usar pygame.time.delay en lugar de time.sleep

            pygame.mixer.music.load(ruta_color)
            pygame.mixer.music.play()
            pygame.time.delay(2000)

    # Crear un hilo para la reproducción de audio
    hilo_audio = threading.Thread(target=reproducir)
    # Establecer el hilo como un hilo en segundo plano, no bloqueará la finalización del programa
    hilo_audio.daemon = True
    # Iniciar el hilo
    hilo_audio.start()

def procesar_deteccion(det, img, label, box):
    x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    # Verifica si las coordenadas de la caja delimitadora están dentro de los límites de la imagen
    #x_start = max(0, min(x_start, ancho - 1))
    #y_start = max(0, min(y_start, alto - 1))
    #x_end = max(0, min(x_end, ancho - 1))
    #y_end = max(0, min(y_end, alto - 1))
    cv2.putText(img, label, (x_start, y_start - 25), 1, 1.2, (0, 0, 0), 2)
    cv2.putText(img, "%:{:.2f}".format(det[2] * 100), (x_start, y_start - 5), 1, 1.2, (0, 0, 0), 2)
    # Verifica si la imagen se pasa correctamente y recorta la región de interés
    #if img is not None:
    cut = img[y_start:y_end, x_start:x_end]
    cut = cv2.cvtColor(cut, cv2.COLOR_BGR2HSV)
    return cut
    #else:
    #    print("Error: La imagen original es nula.")
    #    return None

def obtener_color(det,cut):
    # Perros
    perroNegroBajo = np.array([0, 0, 0], np.uint8)
    perroNegroAlto = np.array([78, 0, 37], np.uint8)
    
    perroBlancoBajo = np.array([234, 234, 228], np.uint8)
    perroBlancoAlto = np.array([230, 201, 162], np.uint8)

    perroGrisObscuroBajo = np.array([0, 0, 50], np.uint8)
    perroGrisObscuroAlto = np.array([179, 30, 120], np.uint8)

    perroRojoBajo1 = np.array([0, 100, 20], np.uint8)
    perroRojoAlto1 = np.array([8, 255, 255], np.uint8)

    perroRojoBajo2 = np.array([175, 100, 20], np.uint8)
    perroRojoAlto2 = np.array([179, 255, 255], np.uint8)

    perroDoradoBajo = np.array([20, 100, 100], np.uint8)
    perroDoradoAlto = np.array([45, 255, 255], np.uint8)

    perroCafeBajo = np.array([10, 50, 40], np.uint8)
    perroCafeAlto = np.array([11, 50, 13], np.uint8)

    perroGrisBajo = np.array([187, 0, 78], np.uint8)
    perroGrisAlto = np.array([234, 0, 228], np.uint8)  

    perroRojoBajo = np.array([0, 100, 20], np.uint8)
    perroRojoAlto = np.array([15, 255, 255], np.uint8)

    perroAmarilloBajo = np.array([20, 100, 100], np.uint8)
    perroAmarilloAlto = np.array([45, 255, 255], np.uint8)

    perroGrisOscuroBajo = np.array([0, 0, 50], np.uint8)
    perroGrisOscuroAlto = np.array([179, 30, 120], np.uint8)

    perroGrisClaroBajo = np.array([0, 0, 150], np.uint8)
    perroGrisClaroAlto = np.array([179, 30, 255], np.uint8)

    perroNaranjaBajo = np.array([13, 77, 100], np.uint8)
    perroNaranjaAlto = np.array([5, 100, 100], np.uint8)

    perroCremaBajo = np.array([20, 50, 150], np.uint8)
    perroCremaAlto = np.array([40, 95, 255], np.uint8)

    # Gatos
    gatoNaranjaBajo = np.array([5, 100, 100], np.uint8)
    gatoNaranjaAlto = np.array([20, 255, 255], np.uint8)

    gatoBlancoBajo = np.array([0, 0, 200], np.uint8)
    gatoBlancoAlto = np.array([179, 30, 255], np.uint8)

    gatoNegroBajo = np.array([0, 0, 0], np.uint8)
    gatoNegroAlto = np.array([179, 255, 30], np.uint8)

    gatoMarronBajo = np.array([0, 50, 50], np.uint8)
    gatoMarronAlto = np.array([30, 255, 150], np.uint8)

    gatoNegroBlancoBajo = np.array([0, 0, 0], np.uint8)
    gatoNegroBlancoAlto = np.array([179, 30, 150], np.uint8)

    gatoMarronBlancoBajo = np.array([0, 50, 50], np.uint8)
    gatoMarronBlancoAlto = np.array([30, 30, 200], np.uint8)

    siamesBajo = np.array([15, 100, 100], np.uint8)
    siamesAlto = np.array([35, 255, 255], np.uint8)

    atigradoBajo = np.array([0, 50, 50], np.uint8)
    atigradoAlto = np.array([30, 255, 150], np.uint8)

    blancoNegroBajo = np.array([0, 0, 0], np.uint8)
    blancoNegroAlto = np.array([179, 30, 150], np.uint8)

    gatoGrisBajo = np.array([187, 0, 78], np.uint8)
    gatoGrisAlto = np.array([234, 0, 228], np.uint8)  

    # Aves
    aveAzulBajo = np.array([90, 50, 50], np.uint8)
    aveAzulAlto = np.array([130, 255, 255], np.uint8)

    aveRojaBajo = np.array([0, 100, 100], np.uint8)
    aveRojaAlto = np.array([10, 255, 255], np.uint8)

    aveVerdeBajo = np.array([40, 50, 50], np.uint8)
    aveVerdeAlto = np.array([80, 255, 255], np.uint8)

    aveAmarillaBajo = np.array([20, 100, 100], np.uint8)
    aveAmarillaAlto = np.array([40, 255, 255], np.uint8)

    aveMarronBajo = np.array([10, 50, 50], np.uint8)
    aveMarronAlto = np.array([30, 255, 150], np.uint8)

    avePurpuraBajo = np.array([130, 50, 50], np.uint8)
    avePurpuraAlto = np.array([160, 255, 255], np.uint8)

    aveRosaBajo = np.array([160, 50, 50], np.uint8)
    aveRosaAlto = np.array([180, 255, 255], np.uint8)

    aveNaranjaBajo = np.array([10, 100, 100], np.uint8)
    aveNaranjaAlto = np.array([20, 255, 255], np.uint8)

    aveGrisBajo = np.array([187, 0, 78], np.uint8)
    aveGrisAlto = np.array([234, 0, 228], np.uint8)  


    # Caballos

    caballoCafeBajo = np.array([10, 50, 50], np.uint8)
    caballoCafeAlto = np.array([30, 255, 150], np.uint8)

    caballoNegroBajo = np.array([0, 0, 0], np.uint8)
    caballoNegroAlto = np.array([179, 255, 30], np.uint8)
   
    caballoBlancoBajo = np.array([60, 11, 100], np.uint8)
    caballoBlancoAlto = np.array([126, 44, 84], np.uint8)

    caballoGrisBajo = np.array([187, 0, 78], np.uint8)
    caballoGrisAlto = np.array([234, 0, 228], np.uint8)  

    caballoAlazanBajo = np.array([0, 100, 100], np.uint8)
    caballoAlazanAlto = np.array([10, 255, 255], np.uint8)

    # Vacas
    vacaBlancoNegroBajo = np.array([0, 0, 0], np.uint8)
    vacaBlancoNegroAlto = np.array([179, 30, 150], np.uint8)

    vacaNegroBajo = np.array([0, 0, 0], np.uint8)
    vacaNegroAlto = np.array([179, 255, 30], np.uint8)

    vacaCafeBajo = np.array([10, 50, 50], np.uint8)
    vacaCafeAlto = np.array([30, 255, 150], np.uint8)

    vacaGrisBajo = np.array([187, 0, 78], np.uint8)
    vacaGrisAlto = np.array([234, 0, 228], np.uint8)  

    vacaRojoBlancoBajo = np.array([0, 100, 100], np.uint8)
    vacaRojoBlancoAlto = np.array([10, 255, 255], np.uint8)

    # Ovejas
    ovejaBlancoBajo = np.array([60, 11, 100], np.uint8)
    ovejaBlancoAlto = np.array([26, 44, 84], np.uint8)

    ovejaNegroBajo = np.array([0, 0, 0], np.uint8)
    ovejaNegroAlto = np.array([78, 0, 37], np.uint8)

    ovejaCafeBajo = np.array([27, 55, 62], np.uint8)
    ovejaCafeAlto = np.array([30, 255, 150], np.uint8)

    ovejaGrisBajo = np.array([187, 0, 78], np.uint8)
    ovejaGrisAlto = np.array([234, 0, 228], np.uint8)    

    if det[1] == PAJARO:
        #AVES
        aveAzul = cv2.inRange(cut, aveAzulBajo, aveAzulAlto)
        aveRoja = cv2.inRange(cut, aveRojaBajo, aveRojaAlto)
        aveVerde = cv2.inRange(cut, aveVerdeBajo, aveVerdeAlto)
        aveAmarilla = cv2.inRange(cut, aveAmarillaBajo, aveAmarillaAlto)
        aveMarron = cv2.inRange(cut, aveMarronBajo, aveMarronAlto)
        avePurpura = cv2.inRange(cut, avePurpuraBajo, avePurpuraAlto)
        aveRosa = cv2.inRange(cut, aveRosaBajo, aveRosaAlto)
        aveNaranja = cv2.inRange(cut, aveNaranjaBajo, aveNaranjaAlto)
        aveGris = cv2.inRange(cut, aveGrisBajo, aveGrisAlto)

        colores = {
            "azul": aveAzul,
            "roja": aveRoja,
            "verde": aveVerde,
            "amarilla": aveAmarilla,
            "marrón": aveMarron,
            "purpura": avePurpura,
            "rosa": aveRosa,
            "naranja": aveNaranja,
            "gris": aveGris,
        }

        max_valor = 0
        color_detectado = None

        for color, imagen in colores.items():
            valor_actual = cv2.countNonZero(imagen)
            if valor_actual > max_valor:
                max_valor = valor_actual
                color_detectado = color
        if (color_detectado!=None):
            return color_detectado
        else:
            return "No se detecto el color"
        
    elif det[1] == GATO:
        #GATOS
        gatoGris = cv2.inRange(cut, gatoGrisBajo, gatoGrisAlto)
        gatoNaranja = cv2.inRange(cut, gatoNaranjaBajo, gatoNaranjaAlto)
        gatoBlanco = cv2.inRange(cut, gatoBlancoBajo, gatoBlancoAlto)
        gatoNegro = cv2.inRange(cut, gatoNegroBajo, gatoNegroAlto)
        gatoMarron = cv2.inRange(cut, gatoMarronBajo, gatoMarronAlto)
        gatoNegroBlanco = cv2.inRange(cut, gatoNegroBlancoBajo, gatoNegroBlancoAlto)
        gatoMarronBlanco = cv2.inRange(cut, gatoMarronBlancoBajo, gatoMarronBlancoAlto)
        siames = cv2.inRange(cut, siamesBajo, siamesAlto)
        atigrado = cv2.inRange(cut, atigradoBajo, atigradoAlto)
        blancoNegro = cv2.inRange(cut, blancoNegroBajo, blancoNegroAlto)

        colores = {
            "gris": gatoGris,
            "naranja": gatoNaranja,
            "blanco": gatoBlanco,
            "negro": gatoNegro,
            "marrón": gatoMarron,
            "negro y blanco": gatoNegroBlanco,
            "marrón y blanco": gatoMarronBlanco,
            "siamés": siames,
            "atigrado": atigrado,
            "blanco y negro": blancoNegro,
        }

        max_valor = 0
        color_detectado = None

        for color, imagen in colores.items():
            valor_actual = cv2.countNonZero(imagen)
            if valor_actual > max_valor:
                max_valor = valor_actual
                color_detectado = color
        if (color_detectado!=None):
            return color_detectado
        else:
            return "No se detecto el color"
        
    elif det[1] == VACA:
        #VACAS
        vacaBlancoNegro = cv2.inRange(cut, vacaBlancoNegroBajo, vacaBlancoNegroAlto)
        vacaNegro = cv2.inRange(cut, vacaNegroBajo, vacaNegroAlto)
        vacaCafe = cv2.inRange(cut, vacaCafeBajo, vacaCafeAlto)
        vacaGris = cv2.inRange(cut, vacaGrisBajo, vacaGrisAlto)
        vacaRojoBlanco = cv2.inRange(cut, vacaRojoBlancoBajo, vacaRojoBlancoAlto)
        colores = {
            "blanco y negro": vacaBlancoNegro,
            "negra": vacaNegro,
            "cafe": vacaCafe,
            "gris": vacaGris,
            "rojo y blanco": vacaRojoBlanco,
        }

        max_valor = 0
        color_detectado = None

        for color, imagen in colores.items():
            valor_actual = cv2.countNonZero(imagen)
            if valor_actual > max_valor:
                max_valor = valor_actual
                color_detectado = color
        if (color_detectado!=None):
            return color_detectado
        else:
            return "No se detecto el color"
        
    elif det[1] == PERRO:
        #PERROS
        perroNegro = cv2.inRange(cut, perroNegroBajo, perroNegroAlto)
        perroBlanco = cv2.inRange(cut, perroBlancoBajo, perroBlancoAlto)
        perroGrisObscuro = cv2.inRange(cut, perroGrisObscuroBajo, perroGrisObscuroAlto)
        perroRojo1 = cv2.inRange(cut, perroRojoBajo1, perroRojoAlto1)
        perroRojo2 = cv2.inRange(cut, perroRojoBajo2, perroRojoAlto2)
        perroDorado = cv2.inRange(cut, perroDoradoBajo, perroDoradoAlto)
        perroCafe = cv2.inRange(cut, perroCafeBajo, perroCafeAlto)
        perroCafe = cv2.inRange(cut, perroCafeBajo, perroCafeAlto)
        perroGris = cv2.inRange(cut, perroGrisBajo, perroGrisAlto)
        perroRojo = cv2.inRange(cut, perroRojoBajo, perroRojoAlto)
        perroAmarillo = cv2.inRange(cut, perroAmarilloBajo, perroAmarilloAlto)
        perroGrisOscuro = cv2.inRange(cut, perroGrisOscuroBajo, perroGrisOscuroAlto)
        perroGrisClaro = cv2.inRange(cut, perroGrisClaroBajo, perroGrisClaroAlto)
        perroNaranja = cv2.inRange(cut, perroNaranjaBajo, perroNaranjaAlto)
        perroCrema = cv2.inRange(cut, perroCremaBajo, perroCremaAlto)
        
        colores = {
        "negro": perroNegro,
        "blanco": perroBlanco,
        "gris oscuro": perroGrisObscuro,
        "rojo tipo 1": perroRojo1,
        "rojo tipo 2": perroRojo2,
        "dorado": perroDorado,
        "cafe": perroCafe,
        "gris": perroGris,
        "rojo": perroRojo,
        "amarillo": perroAmarillo,
        "gris oscuro": perroGrisOscuro,
        "gris claro": perroGrisClaro,
        "naranja": perroNaranja,
        "crema": perroCrema,
        }

        max_valor = 0
        color_detectado = None

        for color, imagen in colores.items():
            valor_actual = cv2.countNonZero(imagen)
            if valor_actual > max_valor:
                max_valor = valor_actual
                color_detectado = color
        if (color_detectado!=None):
            return color_detectado
        else:
            return "No se detecto el color"
            
    elif det[1] == CABALLO:
        # CABALLOS
        caballoCafe = cv2.inRange(cut, caballoCafeBajo, caballoCafeAlto)
        caballoNegro = cv2.inRange(cut, caballoNegroBajo, caballoNegroAlto)
        caballoBlanco = cv2.inRange(cut, caballoBlancoBajo, caballoBlancoAlto)
        caballoGris = cv2.inRange(cut, caballoGrisBajo, caballoGrisAlto)
        caballoAlazan = cv2.inRange(cut, caballoAlazanBajo, caballoAlazanAlto)

        colores={
            "cafe": caballoCafe,
            "negro": caballoNegro,
            "blanco": caballoBlanco,
            "gris": caballoGris,
            "alazán": caballoAlazan,
        }

        max_valor = 0
        color_detectado = None

        for color, imagen in colores.items():
            valor_actual = cv2.countNonZero(imagen)
            if valor_actual > max_valor:
                max_valor = valor_actual
                color_detectado = color
        if (color_detectado!=None):
            return color_detectado
        else:
            return "No se detecto el color"
        
    elif det[1] == OVEJA:
        # OVEJAS
        ovejaBlanco = cv2.inRange(cut, ovejaBlancoBajo, ovejaBlancoAlto)
        ovejaNegro = cv2.inRange(cut, ovejaNegroBajo, ovejaNegroAlto)
        ovejaCafe = cv2.inRange(cut, ovejaCafeBajo, ovejaCafeAlto)
        ovejaGris = cv2.inRange(cut, ovejaGrisBajo, ovejaGrisAlto)

        colores = {
            "blanca": ovejaBlanco,
            "negra": ovejaNegro,
            "cafe": ovejaCafe,
            "gris": ovejaGris,
        }

        max_valor = 0
        color_detectado = None

        for color, imagen in colores.items():
            valor_actual = cv2.countNonZero(imagen)
            if valor_actual > max_valor:
                max_valor = valor_actual
                color_detectado = color
        if (color_detectado!=None):
            return color_detectado
        else:
            return "No se detecto el color"

PAJARO = 3
GATO = 8
VACA = 10
PERRO = 12
CABALLO = 13
OVEJA = 17

clases = {0:"fondo",1:"aeroplane",2:"bicycle",
           3:"ave", 4:"boat",
           5:"bottle",6:"bus",
           7:"car",8:"gato",
           9:"chair",10:"vaca",
           11:"diningtable",12:"perro",
           13:"caballo",14:"motorbike",
           15:"person",16:"pottedplant",
           17:"oveja",18:"sofa",
           19:"train",20:"tvmonitor"}

prototxt = "C:\\Users\\ramon\\Downloads\\OpenCV_DogsDetection\\model\\MobileNetSSD_deploy.prototxt.txt"
model = "C:\\Users\\ramon\\Downloads\\OpenCV_DogsDetection\\model\\MobileNetSSD_deploy.caffemodel"


net = cv2.dnn.readNetFromCaffe(prototxt,model)

img=cv2.imread("C:\\Users\\ramon\\Downloads\\OpenCV_DogsDetection\\cap\\aves.jpg")
alto, ancho, _ = img.shape

img_resized = cv2.resize(img,(300,300))

blob = cv2.dnn.blobFromImage(img_resized, 0.007843, (300,300), (127.5,127.5,127.5))

#print("blob.shape:",blob.shape )

net.setInput(blob)
det=net.forward()
lista_label = []
lista_color = []
for det in det[0][0]:
    if det[2] > 0.60:
        box = det[3:7] * [ancho, alto, ancho, alto]   
        if det[1] == PAJARO:
            label = clases[det[1]]
            print(label)
            cut = procesar_deteccion(det, img, label, box)
            x,y=int(box[0]),int(box[1])
            color=obtener_color(det,cut)
            cv2.putText(img, color, (x + 100, y - 25), 1, 1.2, (0, 0, 0), 2)
            print(color)
            lista_label.append(label)
            lista_color.append(color)
        elif det[1] == GATO:
            label = clases[det[1]]
            print(label)
            cut = procesar_deteccion(det, img, label, box)
            x,y=int(box[0]),int(box[1])
            color=obtener_color(det,cut)
            cv2.putText(img, color, (x + 100, y - 25), 1, 1.2, (0, 0, 0), 2)
            print(color)
            lista_label.append(label)
            lista_color.append(color)
        elif det[1] == VACA:
            label = clases[det[1]]
            print(label)
            cut = procesar_deteccion(det, img, label, box)
            x,y=int(box[0]),int(box[1])
            color=obtener_color(det,cut)
            cv2.putText(img, color, (x + 100, y - 25), 1, 1.2, (0, 0, 0), 2)
            print(color)
            lista_label.append(label)
            lista_color.append(color)
        elif det[1] == PERRO:
            label = clases[det[1]]
            print(label)
            cut = procesar_deteccion(det, img, label, box)
            x,y=int(box[0]),int(box[1])
            color=obtener_color(det,cut)
            cv2.putText(img, color, (x + 100, y - 25), 1, 1.2, (0, 0, 0), 2)
            print(color)
            lista_label.append(label)
            lista_color.append(color)
        elif det[1] == CABALLO:
            label = clases[det[1]]
            print(label)
            cut = procesar_deteccion(det, img, label, box)
            x,y=int(box[0]),int(box[1])
            color=obtener_color(det,cut)
            cv2.putText(img, color, (x + 100, y - 25), 1, 1.2, (0, 0, 0), 2)
            print(color)
            lista_label.append(label)
            lista_color.append(color)
        elif det[1] == OVEJA:
            label = clases[det[1]]
            print(label)
            cut = procesar_deteccion(det, img, label, box)
            x,y=int(box[0]),int(box[1])
            color=obtener_color(det,cut)
            cv2.putText(img, color, (x + 100, y - 25), 1, 1.2, (0, 0, 0), 2)
            print(color)
            lista_label.append(label)
            lista_color.append(color)
        cv2.namedWindow('imagen', cv2.WINDOW_NORMAL)
        cv2.moveWindow('imagen', 0, 0)
        cv2.imshow('imagen', img)

reproducir_audioaudio(lista_label,lista_color)
print(lista_label)
print(lista_color)
cv2.waitKey(0)
cv2.destroyAllWindows()


                       