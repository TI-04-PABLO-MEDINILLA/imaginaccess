import streamlit as st  # interfaz web: streamlit
from PIL import Image  # manejo de imágenes.
import torch  # para trabajar con tensores
from transformers import BlipProcessor, BlipForConditionalGeneration  # modelo BLIP para descripción de imágenes.
from gtts import gTTS  # Google Text To Speech: convertimos texto a voz.
import os  # manejo de rutas y archivos.
import speech_recognition as sr  # reconocer comandos por voz usando el micrófono.
import unidecode  # para procesar el texto y eliminar acentos.
import easyocr  # extracción de texto de la imagen.
import numpy as np  # procesamiento de imágenes como array.
from deep_translator import GoogleTranslator  # traducción de la descripción generada.
import streamlit.components.v1 as components  # para insertar HTML personalizado.
from streamlit_shortcuts import add_shortcuts 
import base64  # codificación de audio para mostrarlo.
import time  # timestamps únicos para nombres de archivo.
from streamlit_shortcuts import shortcut_button   # v>=1.0
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms as T
import pandas as pd
from mutagen.mp3 import MP3   
device='cuda'

STRINGS = {
#como el español es el idioma base, este es el texto mostrado al iniciar la aplicación
    "es": {
        "app_title":           "ImaginAccess",
        "app_subtitle":        "Sube una imagen y obtén una descripción accesible en múltiples formatos.",
        "cmd_hint":            "Comandos disponibles: 'generar descripción', 'generar audio ocr', "
                               "'descargar braille', 'descargar braille ocr', 'ayuda' y 'voz'",
        "select_lang_label":   "Selecciona el idioma para el audio",
        "upload_label":        "Sube una imagen",
        "img_caption":         "Imagen cargada",
        "vqa_section":         "Pregunta a la imagen (VQA)",
        "vqa_input":           "Escribe tu pregunta sobre la imagen",
        "description_title":   "Descripción en {lang}:",
        "braille_desc_title":  "Texto en Braille (descripción):",
        "ocr_title":           "Texto detectado en la imagen (OCR):",
        "braille_ocr_title":   "Texto OCR en Braille:",
        "no_ocr":              "No se detectó texto en la imagen.",
        "spinner_generate_desc": "Generando descripción...",
        "spinner_ocr":           "Ejecutando OCR...",
        "spinner_translate":     "Traduciendo al {lang}...",
        "download_csv": "Descargar CSV",
        "download_ocr_audio":"Descargar Audio del Texto OCR",
        "download_in_braille":"Descargar en notación Braille",
        "question":"Pregunta:",
        "answer":"Respuesta",
        "answer_question":"Responder pregunta",
        "extract_ocr_text":"Extrayendo texto con OCR..."
        
    },

#traducción a inglés
"en": {
    "app_title":            "ImaginAccess",
    "app_subtitle":         "Upload an image and get an accessible description in multiple formats.",
    "cmd_hint":             "Available commands: 'generate description', 'generate OCR audio', "
                            "'download Braille', 'download Braille OCR', 'help' and 'voice'",
    "select_lang_label":    "Select the audio language",
    "upload_label":         "Upload an image",
    "img_caption":          "Uploaded image",
    "vqa_section":          "Ask the image (VQA)",
    "vqa_input":            "Type your question about the image",
    "description_title":    "Description in {lang}:",
    "braille_desc_title":   "Braille text (description):",
    "ocr_title":            "Text detected in the image (OCR):",
    "braille_ocr_title":    "OCR text in Braille:",
    "no_ocr":               "No text detected in the image.",
    "spinner_generate_desc":"Generating description...",
    "spinner_ocr":          "Running OCR...",
    "spinner_translate":    "Translating to {lang}...",
    "download_csv":         "Download CSV",
    "download_ocr_audio":   "Download OCR Audio",
    "download_in_braille":  "Download in Braille notation",
    "question":             "Question:",
    "answer":               "Answer:",
    "answer_question":      "Answer question",
    "extract_ocr_text":     "Extracting text with OCR..."
},

#traducción a francés
"fr": {
    "app_title":            "ImaginAccess",
    "app_subtitle":         "Téléversez une image et obtenez une description accessible dans plusieurs formats.",
    "cmd_hint":             "Commandes disponibles : 'générer description', 'générer audio OCR', "
                            "'télécharger braille', 'télécharger braille OCR', 'aide' et 'voix'",
    "select_lang_label":    "Sélectionnez la langue audio",
    "upload_label":         "Téléversez une image",
    "img_caption":          "Image téléversée",
    "vqa_section":          "Interroger l'image (VQA)",
    "vqa_input":            "Tapez votre question sur l'image",
    "description_title":    "Description en {lang} :",
    "braille_desc_title":   "Texte Braille (description) :",
    "ocr_title":            "Texte détecté dans l'image (OCR) :",
    "braille_ocr_title":    "Texte OCR en Braille :",
    "no_ocr":               "Aucun texte détecté dans l'image.",
    "spinner_generate_desc":"Génération de la description…",
    "spinner_ocr":          "Exécution de l'OCR…",
    "spinner_translate":    "Traduction vers le {lang}…",
    "download_csv":         "Télécharger CSV",
    "download_ocr_audio":   "Télécharger l'audio OCR",
    "download_in_braille":  "Télécharger en Braille",
    "question":             "Question :",
    "answer":               "Réponse :",
    "answer_question":      "Répondre à la question",
    "extract_ocr_text":     "Extraction du texte par OCR…"
},

#traducción a alemán
"de": {
    "app_title":            "ImaginAccess",
    "app_subtitle":         "Lade ein Bild hoch und erhalte eine barrierefreie Beschreibung in mehreren Formaten.",
    "cmd_hint":             "Verfügbare Befehle: 'Beschreibung erstellen', 'OCR-Audio erstellen', "
                            "'Braille herunterladen', 'Braille OCR herunterladen', 'hilfe' und 'sprache'",
    "select_lang_label":    "Wähle die Audiosprache",
    "upload_label":         "Bild hochladen",
    "img_caption":          "Hochgeladenes Bild",
    "vqa_section":          "Frage das Bild (VQA)",
    "vqa_input":            "Stelle deine Frage zum Bild",
    "description_title":    "Beschreibung auf {lang}:",
    "braille_desc_title":   "Braille-Text (Beschreibung):",
    "ocr_title":            "Text im Bild erkannt (OCR):",
    "braille_ocr_title":    "OCR-Text in Braille:",
    "no_ocr":               "Im Bild wurde kein Text erkannt.",
    "spinner_generate_desc":"Beschreibung wird erstellt…",
    "spinner_ocr":          "OCR wird ausgeführt…",
    "spinner_translate":    "Übersetze ins {lang}…",
    "download_csv":         "CSV herunterladen",
    "download_ocr_audio":   "OCR-Audio herunterladen",
    "download_in_braille":  "Braille-Datei herunterladen",
    "question":             "Frage:",
    "answer":               "Antwort:",
    "answer_question":      "Frage beantworten",
    "extract_ocr_text":     "Text wird mit OCR extrahiert…"
},

#traducción a italiano.
"it": {
    "app_title":            "ImaginAccess",
    "app_subtitle":         "Carica un'immagine e ottieni una descrizione accessibile in più formati.",
    "cmd_hint":             "Comandi disponibili: 'genera descrizione', 'genera audio OCR', "
                            "'scarica braille', 'scarica braille OCR', 'aiuto' e 'voce'",
    "select_lang_label":    "Seleziona la lingua per l'audio",
    "upload_label":         "Carica un'immagine",
    "img_caption":          "Immagine caricata",
    "vqa_section":          "Interroga l'immagine (VQA)",
    "vqa_input":            "Scrivi la tua domanda sull'immagine",
    "description_title":    "Descrizione in {lang}:",
    "braille_desc_title":   "Testo in Braille (descrizione):",
    "ocr_title":            "Testo rilevato nell'immagine (OCR):",
    "braille_ocr_title":    "Testo OCR in Braille:",
    "no_ocr":               "Nessun testo rilevato nell'immagine.",
    "spinner_generate_desc":"Generazione della descrizione…",
    "spinner_ocr":          "Esecuzione OCR…",
    "spinner_translate":    "Traduzione in {lang}…",
    "download_csv":         "Scarica CSV",
    "download_ocr_audio":   "Scarica audio OCR",
    "download_in_braille":  "Scarica in Braille",
    "question":             "Domanda:",
    "answer":               "Risposta:",
    "answer_question":      "Rispondere alla domanda",
    "extract_ocr_text":     "Estrazione del testo con OCR…"
},

#traducción a portugués.
"pt": {
    "app_title":            "ImaginAccess",
    "app_subtitle":         "Carregue uma imagem e obtenha uma descrição acessível em vários formatos.",
    "cmd_hint":             "Comandos disponíveis: 'gerar descrição', 'gerar áudio OCR', "
                            "'baixar braille', 'baixar braille OCR', 'ajuda' e 'voz'",
    "select_lang_label":    "Selecione o idioma do áudio",
    "upload_label":         "Carregar uma imagem",
    "img_caption":          "Imagem carregada",
    "vqa_section":          "Pergunte à imagem (VQA)",
    "vqa_input":            "Digite sua pergunta sobre a imagem",
    "description_title":    "Descrição em {lang}:",
    "braille_desc_title":   "Texto em Braille (descrição):",
    "ocr_title":            "Texto detectado na imagem (OCR):",
    "braille_ocr_title":    "Texto OCR em Braille:",
    "no_ocr":               "Nenhum texto detectado na imagem.",
    "spinner_generate_desc":"Gerando descrição…",
    "spinner_ocr":          "Executando OCR…",
    "spinner_translate":    "Traduzindo para {lang}…",
    "download_csv":         "Baixar CSV",
    "download_ocr_audio":   "Baixar áudio OCR",
    "download_in_braille":  "Baixar em Braille",
    "question":             "Pergunta:",
    "answer":               "Resposta:",
    "answer_question":      "Responder pergunta",
    "extract_ocr_text":     "Extraindo texto com OCR…"
},

#traducción a ruso.
"ru": {
    "app_title":            "ImaginAccess",
    "app_subtitle":         "Загрузите изображение и получите доступное описание в нескольких форматах.",
    "cmd_hint":             "Доступные команды: 'создать описание', 'создать аудио OCR', "
                            "'скачать Braille', 'скачать Braille OCR', 'помощь' и 'голос'",
    "select_lang_label":    "Выберите язык аудио",
    "upload_label":         "Загрузить изображение",
    "img_caption":          "Загруженное изображение",
    "vqa_section":          "Спросите изображение (VQA)",
    "vqa_input":            "Введите свой вопрос об изображении",
    "description_title":    "Описание на {lang}:",
    "braille_desc_title":   "Текст Брайля (описание):",
    "ocr_title":            "Текст, обнаруженный на изображении (OCR):",
    "braille_ocr_title":    "OCR-текст в Брайле:",
    "no_ocr":               "На изображении текст не обнаружен.",
    "spinner_generate_desc":"Генерация описания…",
    "spinner_ocr":          "Запуск OCR…",
    "spinner_translate":    "Перевод на {lang}…",
    "download_csv":         "Скачать CSV",
    "download_ocr_audio":   "Скачать OCR-аудио",
    "download_in_braille":  "Скачать в Брайле",
    "question":             "Вопрос:",
    "answer":               "Ответ:",
    "answer_question":      "Ответить на вопрос",
    "extract_ocr_text":     "Извлечение текста с помощью OCR…"
},

}


def t(key: str, **kwargs):
    """
    A partir de la clave del diccionario y el idioma seleccionado en la aplicación, devuelve el texto traducido al idioma seleccionado.
    """
    lang = st.session_state.get("ui_lang", "es")
    base = STRINGS.get(lang, STRINGS["es"])
    text = base.get(key, STRINGS["es"].get(key, key))
    return text.format(**kwargs) if kwargs else text


#Colores para cada tipo de daltonismo.
COLOR_THEMES = {
    "base": {
        "background": "#0D0D0D", "text": "#E6E6E6", "header": "#FFA500", "button": "#0072B2", "button_hover": "#5D3FD3", "success": "#004225"
    },
    "protanopia": {
        "background": "#000000", "text": "#E5E5E5", "header": "#F2BC57",  "button": "#90C3D4", "button_hover": "#66BBDD", "success": "#00496B"
    },
    "deuteranopia": {
        "background": "#000000", "text": "#F2F2F2", "header": "#FFB482", "button": "#CC6677", "button_hover": "#AA5588", "success": "#663C5E"
    },
    "tritanopia": {
        "background": "#000000", "text": "#F0F0F0", "header": "#E69F00", "button": "#FDB462", "button_hover": "#4499CC", "success": "#705900"
    }
}



#Rango de VAD para cada emoción.
EMOCIONES_VAD = {
    "Expectativa": [(7.33, 8.33), (5.35, 6.35), (6.12, 7.12)],
    "Enfado": [(2.16, 3.16), (7.36, 8.36), (5.8, 6.8)],
    "Alegría": [(6.56, 7.56), (5.16, 6.16), (6.09, 7.09)],
    "Placer": [(6.1, 7.1), (4, 5), (6.03, 7.03)],
    "Rechazo": [(2.35, 3.35), (5.64, 6.64), (4.99, 5.99)],
    "Seguridad": [(7.21, 8.21), (6.26, 7.26), (7.63, 8.63)],
    "Desagrado": [(2.88, 3.88), (5.54, 6.54), (5.19, 6.19)],
    "Compasión": [(6.21, 7.21), (5.57, 6.57), (6.84, 7.84)],
    "Nerviosismo": [(3.99, 4.99), (6.73, 7.73), (4.75, 5.75)],
    "Confusión": [(3.74, 4.74), (4.63, 5.63), (3.89, 4.89)],
    "Vergüenza": [(3.86, 4.86), (5.83, 6.83), (3.39, 4.39)],
    "Compromiso": [(6.54, 7.54), (6.74, 7.74), (6.41, 7.41)],
    "Aprecio": [(7.18, 8.18), (5.58, 6.58), (7.01, 8.01)],
    "Alegría intensa": [(7.74, 8.74), (7.65, 8.65), (6.95, 7.95)],
    "Cansancio": [(2.15, 3.15), (2.02, 3.02), (2.84, 3.84)],
    "Miedo": [(2.53, 3.53), (7.71, 8.71), (2.41, 3.41)],
    "Felicidad": [(6.12, 7.12), (4.01, 5.01), (6.89, 7.89)],
    "Dolor emocional": [(2.02, 3.02), (6.38, 7.38), (2.13, 3.13)],
    "Calma": [(6.96, 7.96), (2.71, 3.71), (6.78, 7.78)],
    "Tristeza": [(6.0,7.0), (6.03 ,7.03), (6.2, 7.2)],
    "Ternura": [(5.97, 6.97), (5.05, 6.05), (5.13, 6.13)],
    "Angustia": [(2.12, 3.12), (5.91, 6.91), (2.59, 3.59)],
    "Sorpresa": [(6.68, 7.68), (7.61, 8.61), (5.58, 6.58)],
    "Aburrimiento": [(5.5, 6.5), (3.4, 4.4), (6.0, 7.0)],
    "Neutralidad": [(6.52, 7.52), (4.01, 5.01), (5.47, 6.47)],
    "Nostalgia": [(5.55, 6.55), (5.04, 6.04), (4.92, 5.92)],
}

#Para generar la frase con coherencia, se selecciona entre femenino y masculino.
GENERO_EMOCIONES = {
    "Expectativa": 'f', "Enfado": 'm', "Alegría": 'f', "Placer": 'm', "Rechazo": 'm',
    "Seguridad": 'f', "Desagrado": 'm', "Compasión": 'f', "Nerviosismo": 'm', "Confusión": 'f',
    "Vergüenza": 'f', "Compromiso": 'm', "Aprecio": 'm', "Alegría intensa": 'f',
    "Cansancio": 'm', "Miedo": 'm', "Felicidad": 'f', "Dolor emocional": 'm', "Calma": 'f',
    "Tristeza": 'f', "Ternura": 'f', "Angustia": 'f', "Sorpresa": 'f',
    "Aburrimiento": 'm', "Neutralidad": 'f', "Nostalgia": 'f'
}

#Para generar la frase con coherencia, se selecciona entre femenino y masculino.
INTENSIDADES = {
    'apenas perceptible': {'m': 'apenas perceptible', 'f': 'apenas perceptible'},
    'leve': {'m': 'leve', 'f': 'leve'},
    'moderado': {'m': 'moderado', 'f': 'moderada'},
    'intenso': {'m': 'intenso', 'f': 'intensa'},
    'muy intenso': {'m': 'muy intenso', 'f': 'muy intensa'},
}

#En caso de que no coincida el vector VAD generado por el clasificador con algún rango, se halla la distancia euclídea entre los vectores VAD para tomar la emoción más cercana a los valores VAD aproximados.
def distancia_vad(vad1, vad2):
    return np.linalg.norm(np.array(vad1) - np.array(vad2))

def clasificar_emocion(v, a, d):
    entrada = (v, a, d)
    emocion_detectada = None

    #1. Buscamos la emoción cuyo rango incluye el VAD aproximado.
    for emocion, (v_range, a_range, d_range) in EMOCIONES_VAD.items():
        if v_range[0] <= v <= v_range[1] and a_range[0] <= a <= a_range[1] and d_range[0] <= d <= d_range[1]:
            return emocion  # está dentro del rango directamente

    #2. Debido a que no está en ninguno de los rangos definidos, buscamos la más cercana por distancia euclídea.
    min_dist = float('inf')
    emocion_mas_cercana = None

    for emocion, (v_range, a_range, d_range) in EMOCIONES_VAD.items():
        centro_vad = (
            (v_range[0] + v_range[1]) / 2,
            (a_range[0] + a_range[1]) / 2,
            (d_range[0] + d_range[1]) / 2
        )
        dist = distancia_vad(entrada, centro_vad)
        if dist < min_dist:
            min_dist = dist
            emocion_mas_cercana = emocion

    return emocion_mas_cercana

#obtenemos la intensidad a partir de la fórmula explicada en la memoria.
def obtener_intensidad(valence, arousal, dominance):
    intensidad = (arousal + abs(valence - 5) + abs(dominance - 5)) / 3

    if intensidad < 1:
        return 'apenas perceptible'
    elif intensidad < 2:
        return 'leve'
    elif intensidad < 3:
        return 'moderado'
    elif intensidad < 5:
        return 'intenso'
    else:
        return 'muy intenso'

#formamos la frase teniendo en cuento el género.
def frase_emocion(valence, arousal, dominance):
    emocion = clasificar_emocion(valence, arousal, dominance)
    intensidad = obtener_intensidad(valence, arousal, dominance)
    genero = GENERO_EMOCIONES.get(emocion, 'm')
    articulo = "una" if genero == 'f' else "un"
    
    return f"La persona siente {articulo} {emocion.lower()} {intensidad}."


def explorar_por_voz():
    """
    Navegador de archivos controlado por voz:
      • sin eco (espera a que acabe el TTS)
      • con paginación de 10 elementos
      • tolerante a frases como «foto uno punto jota pe ge»
    """
    import os, re, time, unicodedata
    from mutagen.mp3 import MP3          # pip install mutagen
    import streamlit as st

    # para escuchar después de hablar, así el micrófono no detecta lo que dice el audio
    def hablar(texto, nombre="tts", margen=0.4):
        mp3 = generar_audio(texto, idioma=st.session_state.last_lang, nombre_archivo=nombre)
        mostrar_boton_audio(mp3)
        dur = getattr(MP3(mp3).info, "length", max(2.5, len(texto.split()) * 0.4))
        time.sleep(dur + margen)

    #normalizamos para que lo que dice el usuario coincida con el nombre del archivo.
    def norm(s):
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        s = s.lower()

        #sustituimos 'punto' por '.' y eliminar espacios de extensiones
        s = re.sub(r"\bpunto\b\s*", ".", s)
        # casos frecuentes verbales de extensión
        s = re.sub(r"\.j\s*o?\s*t?\s*a?\s*p\s*e?\s*g\s*$", ".jpg",  s)   # .j p g / .jota pe ge
        s = re.sub(r"\.j\s*o?\s*t?\s*a?\s*p\s*e?\s*e?\s*g\s*$", ".jpeg", s)

        s = re.sub(r"[^a-z0-9.]", "", s)   #solo letras, números y puntos
        return s
    st.success("Has elegido el modo explorar.")
    # estado de exploración (carpeta + página)
    if "explore_dir" not in st.session_state:
        st.session_state.explore_dir  = os.path.expanduser("~")
        st.session_state.explore_page = 0

    dir_actual = os.getcwd()
    elementos  = sorted(os.listdir(dir_actual))
    carpetas   = [e for e in elementos if os.path.isdir(os.path.join(dir_actual, e))]
    imagenes   = [e for e in elementos if e.lower().endswith((".jpg", ".jpeg", ".png"))]
    listado    = carpetas + imagenes

    #mostramos de 10 en 10.
    page_size   = 10
    total_pages = max(1, (len(listado) - 1) // page_size + 1)
    page        = max(0, min(st.session_state.explore_page, total_pages - 1))
    i_ini, i_fin = page * page_size, min((page + 1) * page_size, len(listado))
    ventana     = listado[i_ini:i_fin]

    #listar elementos, el audio dice 'punto' para que el usuario entienda como decirlo.
    leer = lambda x: x.replace(".", " punto ")
    texto_listado = (
        f"Carpeta {dir_actual}. "
        f"Mostrando elementos {i_ini + 1}-{i_fin} de {len(listado)}: "
        + ", ".join(map(leer, ventana))
        + ".  Di una carpeta para entrar, una imagen para seleccionarla, "
          "'siguiente', 'anterior' o 'cancelar'."
    )
    hablar(texto_listado, nombre="listado")

    comando_raw = comando_por_voz("Escuchando…")
    comando     = norm(comando_raw)

    #interpretamos lo que el micrófono ha detectado
    if comando in ("cancelar", "salir"):
        st.session_state.pop("explore_dir",  None)
        st.session_state.pop("explore_page", None)
        st.success("🔙 Exploración cancelada.")
        return

    if comando in ("siguiente", "next", "adelante") and page < total_pages - 1:
        st.session_state.explore_page += 1
        st.stop()

    if comando in ("anterior", "previo", "atras") and page > 0:
        st.session_state.explore_page -= 1
        st.stop()

    # diccionarios normalizados → nombre real
    mapa_carpetas = {norm(c): c for c in carpetas}
    mapa_imagenes = {norm(i): i for i in imagenes}

    #también permitimos decir solo el nombre sin extensión
    mapa_imagenes.update({norm(os.path.splitext(i)[0]): i for i in imagenes})

    if comando in mapa_carpetas:
        st.session_state.explore_dir  = os.path.join(dir_actual, mapa_carpetas[comando])
        st.session_state.explore_page = 0
        st.stop()

    if comando in mapa_imagenes:
        img = mapa_imagenes[comando]
        st.session_state.external_image = os.path.join(dir_actual, img)
        st.session_state.pop("explore_dir",  None)
        st.session_state.pop("explore_page", None)
        st.success(f"📸 Imagen seleccionada: {img}")
        st.rerun() # así ya aparece la imagen.
        return

    st.warning(f"No se reconoció “{comando_raw}”. Inténtalo de nuevo.")
    st.stop()

#cargamos el clasificador de emociones. vad_model --> el modelo aproxima Valencia, Activación y Dominancia.
@st.cache_resource
def load_vad_model():
    """
    Carga YOLOv8 para detectar personas y ResNet-50 toma la salida de YOLOv8 para aproxima VAD de la persona.
    """
    from ultralytics import YOLO
    from torchvision import models

    class ResNet50VAD(nn.Module):
        def __init__(self):
            super().__init__()
            
            #tomamos el modelo original
            self.back = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2
            )
            self.back.fc = nn.Linear(self.back.fc.in_features, 3)  #cambiamos la última capa para que aproxime 3 valores (Valencia, Activación y Dominancia)

        def forward(self, x):
            return self.back(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #para utilizar GPU en caso de que fuese posible.

    #Modelo para detectar personas. Lo generamos a partir de los pesos.
    model_det = YOLO("modelos/yolov8n.pt")

    
    model_vad = ResNet50VAD().to(device)
    ruta_resnet50 = "modelos/vad_resnet50_final.pth"  #ruta al modelo que aproxima VAD. 
    #generamos el modelo a partir de los pesos.
    state = torch.load(ruta_resnet50, map_location=device)
    model_vad.load_state_dict(state, strict=True)  
    model_vad.eval()

    return model_det, model_vad #devolvemos ambos modelos para usarlos para detectar personas y posteriormente clasificar sus emociones.

#traducir la frase generada al idoma seleccionado en la aplicación
def traducir_frase(frase_es):
    """Devuelve la frase traducida al idioma seleccionado"""
    lang = st.session_state.get("ui_lang", "es")
    return traducir_texto(frase_es, idioma_objetivo=lang) if lang != "es" else frase_es

#hallar la posición de cada persona
def posicion_bbox(x1, y1, x2, y2, w, h):
    """
    Divide la imagen en tres cuadrantes y devuelve la posición de la persona dependiendo del cuadrante en el que se encuentre.
    """
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    tercio_x, tercio_y = w / 3, h / 3

    col = 0 if cx < tercio_x else 1 if cx < 2 * tercio_x else 2
    row = 0 if cy < tercio_y else 1 if cy < 2 * tercio_y else 2

    etiquetas = [
        ["Arriba a la izquierda", "Arriba en el centro", "Arriba a la derecha"],
        ["En el centro-izquierda", "En el centro",       "En el centro-derecha"],
        ["Abajo a la izquierda",  "Abajo en el centro", "Abajo a la derecha"],
    ]
    pos_es = etiquetas[row][col]
    return traducir_frase(pos_es)  #traducimos al idioma seleccionado en la aplicación.


from collections import Counter

#resumimos las emociones de la aplicación. Es especialmente útil cuando hay más de tres personas en la imagen.
def resumen_emociones(rows):
    """
    Toma como entrada una lista de diccionarios con las claves 'bbox' para las coordenadas de la persona, 
    'Pos' para la posición de la persona (ya traducida) y Emocion con la 'emoción' traducida y genera un
    resumen de las emociones que encuentra en cada posición.

    -Si hay menos de tres personas, simplemente crea una frase por cada persona y las une.
    -Si hay más de tres personas, la estructura del resumen es la siguiente: 
    
        'Hay n personas: {Posicion1} con {Emocion1}, {Posicion2} con {Emocion2}...'

    El orden es de lectura natural: fila superior y dentro de la fila izquierda, fila superior y dento de la fila el centro...
    """
    n    = len(rows)
    #Orden de lectura: empezamos en la fila superior a la izquierda y acabamos en la fila inferior a la derecha.
    orden = sorted(rows, key=lambda r: (r["bbox"][1], r["bbox"][0]))

    if n <= 3:
        frases_es = [
            f"{r['Pos']} hay una persona que siente {r['Emocion'].lower()}"
            for r in orden
        ]
        texto_es = ". ".join(frases_es) + "."
        return traducir_frase(texto_es)  #traduce el resumen.

    #en caso de detectar más de 3 personas, devolvemos el resumen con esta estructura:  'Hay n personas: {Posicion1} con {Emocion1}, {Posicion2} con {Emocion2}...'
    listados = [f"{r['Pos']} con {r['Emocion'].lower()}" for r in orden]
    texto_es = f"Hay {n} personas: " + ", ".join(listados) + "."
    return traducir_frase(texto_es)

def analizar_emociones(image):
    """
    Función que usa todas las funciones que toman las información emocional y generan las frases y el resumen para devolver el resumen final.
    
    Empieza detectando las personas presentes en la imagen con YOLOv8, aproxima el VAD de cada persona, clasifica la emoción, genera las frases 
    individuales, escribe las frases en la imagen y devuelve un resumen con toda la información en audio y notación braille.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#usa GPU si es posible.
    model_det, model_vad = load_vad_model()#carga los modelos

    #detecta las personas
    det = model_det(image)[0] #la clase de las personas es la clase 0
    bboxes = [
        box.xyxy[0].cpu().tolist()
        for box in det.boxes
        if int(box.cls) == 0 and box.conf > 0.40 #tomamos como umbral de confianza 0.4
    ]
    if not bboxes: #no ha detectado personas
        st.info("No se detectaron personas en la imagen.")
        return

    tf = T.Compose([ #transformamos las imágenes de entrada del modelo.
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    lang_ui   = st.session_state.get("ui_lang", "es") #tomamos el idioma que se usará para la traducción del texto. El idioma por defecto es el español
    lang_tts  = st.session_state.get("last_lang", "es")# lo mismo pero para el audio
    filas = []

    for i, (x1, y1, x2, y2) in enumerate(bboxes): #para cada caja delimitadora de cada persona
        crop = image.crop((int(x1), int(y1), int(x2), int(y2))) #recorta la persona
        with torch.no_grad():
            v, a, d = model_vad(tf(crop).unsqueeze(0).to(device)).cpu().squeeze().tolist() #halla VAD

        emoc  = clasificar_emocion(v, a, d) #obtiene la emoción de la persona
        intens= obtener_intensidad(v, a, d) #obtiene la intensidad de la emoción.
        art   = "una" if GENERO_EMOCIONES.get(emoc, "m") == "f" else "un" #selecciona las palabras en función del género.
        frase_es = f"la persona siente {art} {emoc.lower()} {intens}" #genera la frase
        frase_tr = traducir_frase(frase_es)#traduce la frase

        pos_tr   = posicion_bbox(x1, y1, x2, y2, image.width, image.height)
        filas.append({ #guarda en la lista la información de cada persona para generar el resumen posteriormente.
            "id": i,
            "bbox": [x1, y1, x2, y2],
            "Valence": round(v, 2), "Arousal": round(a, 2), "Dominance": round(d, 2),
            "Pos": pos_tr, "Emocion": traducir_frase(emoc),
            "Frase": f"{pos_tr.capitalize()} {frase_tr}",
        })
    #generamos un dataframe para manejar con mayor facilidad los datos.
    st.dataframe(pd.DataFrame(filas).set_index("id"))

    #halla el resumen de las emociones detectadas (de todas las personas)
    resumen = resumen_emociones(filas)#el resumen devuelto ya está traducido.
    st.success(resumen) #muestra el resumen en un recuadro de tipo success

    audio_file = generar_audio(resumen, idioma=lang_tts, nombre_archivo="resumen") #generamos el audio del resumen
    #reproducimos el audio
    mostrar_boton_audio(audio_file)
    st.audio(audio_file)

    #generamos el braille
    braille_txt, _ = texto_a_braille(resumen)
    #mostramos el braille
    st.code(braille_txt, language="text")

    #descargamos automáticamente el braille.
    with open("resumen.brf", "w", encoding="utf-8") as f:
        f.write(braille_txt)
    with open("resumen.brf", "rb") as f:
        st.download_button("📥 Descargar resumen en Braille", data=f,
                           file_name="resumen.brf", mime="text/plain")

    #mostramos todas las frases de cada persona en la imagen.
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    for r in filas:
        x1, y1, x2, y2 = r["bbox"]
        ax.add_patch(
            patches.Rectangle((x1, y1), x2-x1, y2-y1,
                              edgecolor="lime", linewidth=2, fill=False))
        ax.text(x1, max(y1-10, 0), r["Frase"],
                color="yellow", fontsize=8, backgroundcolor="black")
    ax.axis("off")
    st.pyplot(fig)

    #generamos un CSV con toda la información.
    csv = pd.DataFrame(filas).to_csv(index=False).encode("utf-8")
    st.download_button(t("download_csv"), csv,
                       "emociones_vad.csv", "text/csv")
    st.stop()          

#cambiamos los colores en función del tipo de daltonismo.
def aplicar_tema_daltonismo(nombre_tema="base"):
    """
    Cambiamos los colores de la aplicación en función del daltonismo.
    """
    tema = COLOR_THEMES.get(nombre_tema, COLOR_THEMES["base"]) #seleccionamos el tema seleccionado.
    st.session_state["tema_activo"] = nombre_tema
    st.markdown(f"""
        <style>

        .stApp {{
            background-color: {tema["background"]} !important;
            color: {tema["text"]} !important;
        }}
        h1, h2, h3 {{
            color: {tema["header"]} !important;
        }}
        p{{
        color:{tema["text"]} !important;
        font-size:22px !important;
        }}
        button {{
            background-color: {tema["button"]} !important;
            color: white !important;
            border-radius: 10px;
            font-size: 22px;
            border: none;
            color:blue;
        }}
        .stAlertContainer {{
        background-color: {tema["success"]} !important;
        border-left: 0.4rem solid {tema["success"]} !important;
        color: {tema["text"]} !important;          /* que sea legible en cada tema */
    }}
        .stButton > button:hover {{
            background-color: {tema["button_hover"]} !important;
        }}

        /* Elementos solo para lectores de pantalla */
        .sr-only{{
        position:absolute !important;
        width:1px;height:1px;padding:0;margin:-1px;
        overflow:hidden;clip:rect(0,0,0,0);border:0;
        }}

        /* Foco visible cuando se navega con Tab */
        *:focus{{
        outline:3px solid #FFA500 !important;
        outline-offset:2px;
        }}
        </style>
    """, unsafe_allow_html=True)

#cambiamos tamaños y fijamos colores para garantizar los principios WCAG 2.2. 
#   -Tamaño mínimo de 22px
#   -Colores con contraste de 4.5.1.

st.set_page_config(page_title="Imagen Accesible", layout="wide")
st.markdown(
    """
    <a href="#main" class="sr-only" tabindex="0">Saltar al contenido</a>
    <div id="status" role="status" aria-live="polite" class="sr-only"></div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<span id="main"></span>', unsafe_allow_html=True)

st.markdown('''
    <style>
    /* Tamaño mínimo de 22px */
    html, body, .stApp, .css-18e3th9, .css-1d391kg, .css-1offfwp, .css-10trblm {
        font-size: 22px !important;
    }
    /* Fondo oscuro y colores con contraste mayor a 4.5.1 */
    .stApp {
        background-color: #0D0D0D;
        color: #E6E6E6;
        font-family: Verdana, sans-serif;
    }
    h1, h2, h3 {
        color: #FFA500;  /* Naranja para encabezados en tema base */
        font-size: 2.5em;
    }
    /* Personalizamos los botones para el tema base */
    .stButton > button {
        background-color: #8A2BE2;
        color: #ffffff;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 22px;
        border: none;
        display: none;
    }
    .stButton > button:hover {
        background-color: #5D3FD3;
    }
    /* Reducimos el tamaño del select para reducir carga visual */
    div[data-baseweb="select"] {
        width: 200px !important;
        min-width: 200px !important;
    }
    </style>
''', unsafe_allow_html=True)

st.markdown('''
    <style>
    /* Fijamos tamaños y alineación para contenedores de audio y contenedores para mostrar información al usuario. */
    .stAlert {
        max-width: 800px;
        text-align: left;
    }
    audio {
        max-width: 600px;
        text-align: left;
        display: block;
    }
    /* Ocultamos los botones usados para ejecutar comandos por atajos de teclado. Reducimos carga cognitiva. */
    .hidden-buttons button {
        visibility: hidden;
        height: 0;
        padding: 0;
        margin: 0;
        display: none;
    }
    </style>
''', unsafe_allow_html=True)


#Guardamos en caché el modelo BLIP, seleccionado tras la investigación.
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return processor, model

processor, model = load_model()

#Guardamos en caché el modelo BLIP VQA, la versión de BLIP para hacer preguntas visuales.
@st.cache_resource

def load_vqa_model():
    from transformers import BlipProcessor, BlipForQuestionAnswering
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return processor, model

vqa_processor, vqa_model = load_vqa_model()

#función para generar la descripción usando el modelo BLIP.
def generar_descripcion(imagen):
    """ Genera una descripción de la imagen de entrada usando el modelo BLIP """
    entradas = processor(images=imagen, return_tensors="pt").to(model.device)
    salida = model.generate(**entradas)
    return processor.decode(salida[0], skip_special_tokens=True)

def traducir_texto(texto, idioma_objetivo="es"):
    """Traduce un texto al idioma especificado."""
    return GoogleTranslator(source="auto", target=idioma_objetivo).translate(texto)

# Función para generar audio dado un texto
def generar_audio(texto, idioma="es", nombre_archivo="audio"):
    """
    Genera el audio asociado a un texto de entrada. Usa un timestamp único (hora actual) 
    y selecciona el idioma español por defecto.
    """
    timestamp = int(time.time())
    nombre_archivo = f"{nombre_archivo}_{timestamp}.mp3"
    tts = gTTS(texto, lang=idioma)
    tts.save(nombre_archivo)
    return nombre_archivo



# Traducción de texto a braille
def texto_a_braille(texto, opcion_caracter_desconocido="keep", caracter_reemplazo="?"):
    diccionario_braille = {
        #Letras
        "a": "⠁", "b": "⠃", "c": "⠉", "d": "⠙", "e": "⠑",
        "f": "⠋", "g": "⠛", "h": "⠓", "i": "⠊", "j": "⠚",
        "k": "⠅", "l": "⠇", "m": "⠍", "n": "⠝", "o": "⠕",
        "p": "⠏", "q": "⠟", "r": "⠗", "s": "⠎", "t": "⠞",
        "u": "⠥", "v": "⠧", "w": "⠺", "x": "⠭", "y": "⠽", "z": "⠵",
        
        #Signos de puntuación y símbolos comunes
        ".": "⠲", ",": "⠂", ";": "⠆", ":": "⠒", "!": "⠖",
        "?": "⠦", "\"": "⠶", "'": "⠄", "-": "⠤", "(": "⠶", ")": "⠶",
        "/": "⠌", "@": "⠈", "&": "⠯", "*": "⠡", "+": "⠬",
        "=": "⠿", "%": "⠨⠴", "¡": "⠮", "¿": "⠹", " ": " ",
        "$": "⠫", "€": "⠩", "£": "⠣", "°": "⠘",
        "[": "⠪", "]": "⠻", "{": "⠷", "}": "⠾",
        "<": "⠣", ">": "⠜", "\\": "⠳", "~": "⠰⠣", "^": "⠘"
    }

    #Traducción de números
    diccionario_numeros = {
        "1": "⠁", "2": "⠃", "3": "⠉", "4": "⠙", "5": "⠑",
        "6": "⠋", "7": "⠛", "8": "⠓", "9": "⠊", "0": "⠚"
    }

    #Indicadores de mayúsculas, número y volver a usar letras
    indicador_mayuscula = "⠠"
    indicador_numero = "⠼"
    indicador_letra = "⠰"

    #Eliminamos acentos del texto original
    texto = unidecode.unidecode(texto)
    resultado = ""
    en_modo_numerico = False
    caracteres_desconocidos = []
    
    #traducimos todos los caracteres a braille.
    for caracter in texto:
        if caracter.isupper():
            resultado += indicador_mayuscula
            caracter = caracter.lower()
        if caracter.isdigit():
            if not en_modo_numerico:
                resultado += indicador_numero
                en_modo_numerico = True
            resultado += diccionario_numeros[caracter]
        else:
            if en_modo_numerico and caracter.isalpha():
                resultado += indicador_letra
            en_modo_numerico = False
            if caracter in diccionario_braille:
                resultado += diccionario_braille[caracter]
            else:
                if opcion_caracter_desconocido == "keep":
                    resultado += caracter
                elif opcion_caracter_desconocido == "replace":
                    resultado += caracter_reemplazo
                elif opcion_caracter_desconocido == "warn":
                    caracteres_desconocidos.append(caracter)
                    resultado += caracter_reemplazo

    return resultado, caracteres_desconocidos


#Funcionaes para comandos de voz

def comando_por_voz(mensaje="Diga un comando..."):
    """ Convierte a texto el audio detectado por el micrófono."""
    reconocedor = sr.Recognizer()
    with sr.Microphone() as fuente:
        st.write(f"🎙️ {mensaje}")
        audio = reconocedor.listen(fuente)
        try:
            return reconocedor.recognize_google(audio, language="es")
        except:
            return ""
        
def auto_descarga(file_path):
    """Descarga automáticamente un archivo en la ruta especificada"""
    components.html(f"""
        <a id='auto_descarga' href='file/{file_path}' download style='display:none;'></a>
        <script>
            setTimeout(() => document.getElementById('auto_descarga').click(), 1000);
        </script>
    """, height=0)

def mostrar_boton_audio(file_path):
    """Muestra un reproductor de audio que reproduce automáticamente el archivo indicado"""
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        html = f'''
            <audio autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        '''
        st.markdown(html, unsafe_allow_html=True)

# ── NUEVO util en la sección de helpers audio ───────────────
def reproducir_audio_en_cadena(audio_desc, audio_ocr):
    """
    Muestra un reproductor que lanza audio_desc y, al acabar, reproduce audio_ocr.
    Ambos archivos deben ser rutas locales .mp3 ya existentes.
    """
    with open(audio_desc, "rb") as f1, open(audio_ocr, "rb") as f2:
        b64_desc = base64.b64encode(f1.read()).decode()
        b64_ocr  = base64.b64encode(f2.read()).decode()

    html = f"""
    <audio id="audioDesc" autoplay>
        <source src="data:audio/mpeg;base64,{b64_desc}" type="audio/mpeg">
    </audio>

    <audio id="audioOCR">
        <source src="data:audio/mpeg;base64,{b64_ocr}" type="audio/mpeg">
    </audio>

    <script>
        const desc = document.getElementById('audioDesc');
        const ocr  = document.getElementById('audioOCR');
        desc.addEventListener('ended', () => ocr.play());
    </script>
    """
    st.markdown(html, unsafe_allow_html=True)


def mostrar_audio_con_autoplay(file_path):
    """ Reproduce automáticamente un archivo de audio y permite volver a reproducirlo pulsando la tecla r"""
    st.session_state["last_audio"] = file_path
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        html = f"""
            <audio id="audioPlayer" autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            <script>
                document.addEventListener('keydown', function(event) {{
                    if (event.key === 'r' || event.key === 'R') {{
                        const audio = document.getElementById('audioPlayer');
                        if (audio) {{
                            audio.currentTime = 0;
                            audio.play();
                        }}
                    }}
                }});
            </script>
        """
        st.markdown(html, unsafe_allow_html=True)

def anuncio(msg: str) -> None:
    """Manda un mensaje a la región aria-live para que los lectores de pantalla puedan detectarlo y leerlo en voz alta automáticamente"""
    st.markdown(
        f"<script>document.getElementById('status').innerText = `{msg}`;</script>",
        unsafe_allow_html=True,
    )
def ejecutar_comando(comando):
    """ Ejecuta automáticamente el comando detectado por voz """
    comando = comando.lower()

    #Generar audio a partir de la descripción traducida
    if "generar descripción" in comando:
        if "description_translated" in st.session_state:
            st.session_state["audio_file"] = generar_audio(
                st.session_state["description_translated"],
                idioma=st.session_state["last_lang"],
                nombre_archivo="descripcion"
            )
            mostrar_boton_audio(st.session_state["audio_file"])
            st.audio(st.session_state["audio_file"])

    #Generar audio a partir del texto OCR
    elif "generar audio ocr" in comando:
        if "ocr_text" in st.session_state:
            st.session_state["ocr_audio_file"] = generar_audio(
                st.session_state["ocr_text"],
                idioma=st.session_state["last_lang"],
                nombre_archivo="ocr"
            )
            mostrar_boton_audio(st.session_state["ocr_audio_file"])
            st.audio(st.session_state["ocr_audio_file"])

    #Descargar Braille de la descripción
    elif "descargar braille descripcion" in comando:
        with open("braille_descripcion.brf", "w", encoding="utf-8") as f:
            f.write(st.session_state["braille_text"])

    #Descargar Braille del OCR
    elif "descargar braille ocr" in comando:
        with open("braille_ocr.brf", "w", encoding="utf-8") as f:
            f.write(st.session_state["ocr_braille"])

    #Mostrar ayuda con comandos y accesos directos
    elif "ayuda" in comando or "explicación" in comando:
        texto = (
            "Te damos la bienvenida a la aplicación ImaginAccess. "
            "Puedes subir una imagen para obtener automáticamente una descripción, "
            "extraer el texto que contenga, analizar las emociones de las personas presentes "
            "e incluso formular preguntas sobre la imagen.\n\n"
            "Todo el contenido generado está disponible en formato de audio "
            "y en notación Braille.\n\n"
            "Comandos por voz disponibles:\n"
            "- «generar descripción»\n"
            "- «generar audio ocr»\n"
            "- «descargar braille descripción»\n"
            "- «descargar braille ocr»\n"
            "- «analizar emociones»\n"
            "- «voz»\n"
            "- «ayuda»\n"
            "- «quitar imagen»\n"
            "- «cambiar tema daltonismo»\n"
            "- «hacer pregunta»\n"
            "- «manejo de caracteres desconocidos»\n\n"
            "Atajos de teclado:\n"
            "- Shift + U → Generar descripción\n"
            "- Shift + I → Generar audio OCR\n"
            "- Shift + O → Descargar Braille OCR\n"
            "- Shift + P → Descargar Braille descripción\n"
            "- Shift + E → Analizar emociones\n"
            "- Shift + A → Hacer pregunta\n"
            "- Shift + H → Ayuda\n"
            "- Shift + Q → Quitar imagen\n"
            "- Shift + V → Modo voz\n"
            "- Shift + M → Manejo de caracteres desconocidos\n"
            "- Shift + S/D/F/G → Cambiar tema de daltonismo "
            "(normal, protanopia, deuteranopia, tritanopia)"
        )
        st.session_state["help_audio"] = generar_audio(texto, idioma=st.session_state["last_lang"], nombre_archivo="ayuda")
        mostrar_boton_audio(st.session_state["help_audio"])
        st.audio(st.session_state["help_audio"])
        components.html(f"""
            <a id="auto_descarga" href="file/{st.session_state['help_audio']}" download="{st.session_state['help_audio']}" style="display:none;"></a>
            <script>
                setTimeout(() => document.getElementById("auto_descarga").click(), 1000);
            </script>
        """, height=0)

    #Eliminar imagen y limpiar sesión
    elif "quitar imagen" in comando:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        components.html("""
            <script>
                window.location.reload(true);
            </script>
        """, height=0)

    #Preguntar sobre la imagen usando VQA (Visual Question Answering)
    elif "hacer pregunta" in comando:
        if "external_image" in st.session_state:
            image = Image.open(st.session_state["external_image"]).convert("RGB")
        elif "uploaded_file" in st.session_state:
            image = Image.open(st.session_state["uploaded_file"]).convert("RGB")
        else:
            st.warning("No se ha cargado ninguna imagen.")
            return

        while True:
    
            pregunta = comando_por_voz("¿Cuál es tu pregunta?").strip()
            if not pregunta:
                st.warning("No se detectó ninguna pregunta.")
                return

            respuesta = responder_vqa(image, pregunta)
            st.markdown(f"**Pregunta:** {pregunta}  \n**Respuesta:** {respuesta}")

            audio = generar_audio(respuesta, idioma=st.session_state["last_lang"],
                                nombre_archivo="respuesta")
            mostrar_boton_audio(audio)
            st.audio(audio)

            # 2 · ¿Otra pregunta?
            hablar("¿Quieres hacer otra pregunta? Di sí o no.", nombre="mas_preguntas")
            seguir = comando_por_voz("Escuchando…").strip().lower()
            st.success("Se ha detectado "+seguir)
            if seguir not in ("sí", "si", "s"):
                # sale del while y por tanto de la función
                st.success("No se han realizado más preguntas")
                return

    #Cambiar tema de daltonismo
    elif "tema" in comando:
        if "base" in comando:
            aplicar_tema_daltonismo("base")
            st.success("Tema visual cambiado a base.")
        elif "protanopia" in comando:
            aplicar_tema_daltonismo("protanopia")
            st.success("Tema visual cambiado a protanopia.")
        elif "deuteranopia" in comando:
            aplicar_tema_daltonismo("deuteranopia")
            st.success("Tema visual cambiado a deuteranopia.")
        elif "tritanopia" in comando:
            aplicar_tema_daltonismo("tritanopia")
            st.success("Tema visual cambiado a tritanopia.")
        else:
            st.warning("No se reconoció el tipo de tema.")

    #Configurar manejo de caracteres desconocidos para Braille
    elif "manejo de caracteres desconocidos" in comando:
        options_text = (
            "Las opciones para el manejo de caracteres desconocidos son: mantener, reemplazar y advertir. "
            "Por favor, diga la opción que desea."
        )
        audio_options = generar_audio(options_text, idioma=st.session_state["last_lang"], nombre_archivo="desconocidos_opciones")
        mostrar_boton_audio(audio_options)
        respuesta = comando_por_voz().strip().lower()
        if respuesta in ["mantener", "reemplazar", "advertir"]:
            st.session_state["unknown_char_option"] = respuesta
            confirmation_text = f"Has elegido {respuesta} para el manejo de caracteres desconocidos."
        else:
            st.session_state["unknown_char_option"] = "keep"
            confirmation_text = "Opción no reconocida. Se mantiene la opción por defecto: mantener."
        audio_confirmation = generar_audio(confirmation_text, idioma=st.session_state["last_lang"], nombre_archivo="desconocidos_confirmacion")
        mostrar_boton_audio(audio_confirmation)

    #Analizar emociones en imagen
    elif "analizar emociones" in comando:
        if "external_image" in st.session_state:
            image = Image.open(st.session_state["external_image"]).convert("RGB")
        elif "uploaded_file" in st.session_state:
            image = Image.open(st.session_state["uploaded_file"]).convert("RGB")
        else:
            st.warning("No se ha cargado ninguna imagen.")
            return
        analizar_emociones(image)

    #Modo de entrada de imagen por voz (dictar o explorar)
    elif "voz" in comando:
        #preguntamos (el micro NO se abre hasta que hablar() termina)
        hablar(
            "Diga, a continuación, el comando que desea ejecutar, en caso de que quiera subir una imagen, diga 'dictar' o 'explorar'",
            nombre="modo_voz"
        )
        modo = comando_por_voz("Escuchando…").strip().lower()

        if "dictar" in modo:
            hablar(
                "Indica la ruta completa del archivo de imagen, por ejemplo: "
                "ce dos puntos barra usuarios barra foto punto jota pe ge.",
                nombre="instruccion_voz"
            )
            ruta = comando_por_voz("Ruta…").strip()

            if ruta:
                hablar(
                    f"Has dicho: {ruta}. ¿Quieres continuar con esta imagen? "
                    "Di sí o no.",
                    nombre="confirmacion_voz"
                )
                respuesta = comando_por_voz("Escuchando…").strip().lower()

                if respuesta in ("sí", "si"):
                    if not os.path.exists(ruta):
                        st.error("La ruta no existe.")
                    elif not os.path.isfile(ruta):
                        st.error("La ruta apunta a una carpeta, no a un archivo.")
                    elif not ruta.lower().endswith((".jpg", ".jpeg", ".png")):
                        st.error("El archivo no es una imagen válida.")
                    else:
                        # limpiamos caché si cambia de imagen
                        for k in [
                            "description_en", "description_translated", "audio_file",
                            "ocr_text", "ocr_audio_file", "ocr_braille", "braille_text"
                        ]:
                            st.session_state.pop(k, None)
                        st.session_state.external_image = ruta
                        st.success(f"Imagen añadida: {ruta}")
                else:
                    st.info("Acción cancelada.")
            else:
                st.warning("No se detectó ninguna ruta.")
        elif "explorar" in modo:
            explorar_por_voz()   #función para elegir explorar el sistema de archivos hasta seleccionar la imagen.

        else:
            st.success(f"Ejecutando el comando... {modo}")
            ejecutar_comando(modo)

def hablar(texto, nombre="tts", margen=0.4):
    mp3 = generar_audio(texto, idioma=st.session_state.last_lang, nombre_archivo=nombre)
    mostrar_boton_audio(mp3)
    dur = getattr(MP3(mp3).info, "length", max(2.5, len(texto.split()) * 0.4))
    time.sleep(dur + margen)
    
# Manejo de la imagen
def procesar_imagen():
    """ Procesa la subida de la imagen a la aplicación """
    if "external_image" in st.session_state:
        ruta_imagen = st.session_state["external_image"]
        imagen = Image.open(ruta_imagen).convert("RGB")
        return imagen
    elif "uploaded_file" in st.session_state:
        imagen = Image.open(st.session_state["uploaded_file"]).convert("RGB")
        return imagen
    return None

def procesar_archivo_subido():
    """ Procesa la imagen ya subida a la aplicación """
    archivo_subido = st.file_uploader(t("upload_label"), type=["jpg", "png", "jpeg"])
    if archivo_subido is not None:
        if "uploaded_file" in st.session_state:
            if st.session_state["uploaded_file"].name != archivo_subido.name:
                for key in [
                    "description_en", "description_translated", "audio_file",
                    "ocr_text", "ocr_audio_file", "ocr_braille", "braille_text"
                ]:
                    st.session_state.pop(key, None)
        st.session_state["uploaded_file"] = archivo_subido
        st.session_state["image_id"] = archivo_subido.name

def responder_vqa(imagen, pregunta):
    """ Función para realizar preguntas sobre la imagen usando el modelo BLIP-VQA """
    entradas = vqa_processor(images=imagen, text=pregunta, return_tensors="pt").to(vqa_model.device)
    with torch.no_grad():
        salida = vqa_model.generate(**entradas)
    return vqa_processor.decode(salida[0], skip_special_tokens=True)


#Función principal de la aplicación
def main():
    #selección de idioma y uso del español por defecto
    idiomas_seleccionables = ["es", "en", "fr", "de", "it", "pt", "ru"]
    if "last_lang" not in st.session_state:
        st.session_state["last_lang"] = "es"
    if "ui_lang" not in st.session_state:
        st.session_state["ui_lang"] = "es"

    #Mostramos títulos y subtítulos 
    st.title(t("app_title"))
    st.write(t("app_subtitle"))
    st.markdown(t("cmd_hint"))

    #seleccionamos los colores de la interfaz en función del tema.
    aplicar_tema_daltonismo(st.session_state.get("tema_activo", "base"))

    idioma_seleccionado = st.selectbox(
        t("select_lang_label"),
        options=idiomas_seleccionables,
        index=idiomas_seleccionables.index(st.session_state["last_lang"]),
    )

    #Sincroniza idioma de audio e interfaz (el seleccionado)
    if idioma_seleccionado != st.session_state["last_lang"]:
        st.session_state["last_lang"] = idioma_seleccionado
        st.session_state["ui_lang"] = idioma_seleccionado
        #Elimina las traaducciones que había en caché en caso de que se cambie el idioma seleccionado.
        for key in ["description_translated", "audio_file", "ocr_audio_file", "ocr_braille"]:
            st.session_state.pop(key, None)
    #mensaje de ayuda.
    if 'ayuda_generada' not in st.session_state:
        texto = (
            "Te damos la bienvenida a la aplicación ImaginAccess. "
            "Puedes subir una imagen para obtener automáticamente una descripción, "
            "extraer el texto que contenga, analizar las emociones de las personas presentes "
            "e incluso formular preguntas sobre la imagen.\n\n"
            "Todo el contenido generado está disponible en formato de audio "
            "y en notación Braille.\n\n"
            "Comandos por voz disponibles:\n"
            "- «generar descripción»\n"
            "- «generar audio ocr»\n"
            "- «descargar braille descripción»\n"
            "- «descargar braille ocr»\n"
            "- «analizar emociones»\n"
            "- «voz»\n"
            "- «ayuda»\n"
            "- «quitar imagen»\n"
            "- «cambiar tema daltonismo»\n"
            "- «hacer pregunta»\n"
            "- «manejo de caracteres desconocidos»\n\n"
            "Atajos de teclado:\n"
            "- Shift + U → Generar descripción\n"
            "- Shift + I → Generar audio OCR\n"
            "- Shift + O → Descargar Braille OCR\n"
            "- Shift + P → Descargar Braille descripción\n"
            "- Shift + E → Analizar emociones\n"
            "- Shift + A → Hacer pregunta\n"
            "- Shift + H → Ayuda\n"
            "- Shift + Q → Quitar imagen\n"
            "- Shift + V → Modo voz\n"
            "- Shift + M → Manejo de caracteres desconocidos\n"
            "- Shift + S/D/F/G → Cambiar tema de daltonismo "
            "(normal, protanopia, deuteranopia, tritanopia)")
        audio_path = generar_audio(texto, idioma=st.session_state["last_lang"], nombre_archivo="ayuda")
        mostrar_boton_audio(audio_path)
        st.audio(audio_path)
        st.session_state['ayuda_generada'] = True

    procesar_archivo_subido()
    image = procesar_imagen()
    #procesamos la imagen subida y generamos descripciones, audios y braille. En caso de cambiar la imagen, se vuelve a generar todo.
    if image is not None:
        if "current_image_id" in st.session_state:
            changed = False
            if ("uploaded_file" in st.session_state and
                st.session_state["current_image_id"] != st.session_state.get("image_id", "")):
                changed = True
            if ("external_image" in st.session_state and
                st.session_state["current_image_id"] != st.session_state["external_image"]):
                changed = True
            if changed:
                for key in [
                    "description_en", "description_translated", "audio_file",
                    "ocr_text", "ocr_audio_file", "ocr_braille", "braille_text"
                ]:
                    st.session_state.pop(key, None)
        if "uploaded_file" in st.session_state:
            st.session_state["current_image_id"] = st.session_state["image_id"]
        elif "external_image" in st.session_state:
            st.session_state["current_image_id"] = st.session_state["external_image"]

        if 'description_en' not in st.session_state:
            with st.spinner(t("spinner_generate_desc")):
                anuncio(t("spinner_generate_desc"))
                st.session_state["description_en"] = generar_descripcion(image)

        if 'ocr_text' not in st.session_state:
            with st.spinner(t("spinner_ocr")):
                anuncio(t("spinner_ocr"))
                st.info("🧠 Extrayendo texto con OCR...")
                reader = easyocr.Reader([idioma_seleccionado, 'en'])
                image_np = np.array(image)
                results = reader.readtext(image_np)
                st.session_state["ocr_text"] = " ".join([r[1] for r in results]) if results else ""
        st.subheader(t("vqa_section"))
        with st.form("vqa_form", clear_on_submit=False):
            pregunta_usuario = st.text_input(t("vqa_input"), key="input_pregunta_vqa")
            submitted = st.form_submit_button(t("answer_question"))
            if submitted and pregunta_usuario:
                st.session_state["pending_vqa"] = pregunta_usuario
        if "pending_vqa" in st.session_state:
            pregunta = st.session_state.pop("pending_vqa")
            respuesta = responder_vqa(image, pregunta)
            st.session_state["respuesta_vqa"] = respuesta
            st.session_state["audio_vqa"] = generar_audio(
                respuesta, idioma=st.session_state["last_lang"], nombre_archivo="respuesta"
    )


        #Mostrar respuesta justo debajo del input
        if "respuesta_vqa" in st.session_state:
            st.markdown(t("question") +f": {pregunta_usuario}")
            st.markdown(t("answer")+f": {st.session_state['respuesta_vqa']}")
            mostrar_boton_audio(st.session_state["audio_vqa"])
            st.audio(st.session_state["audio_vqa"])

        col1, spacer, col2 = st.columns([1.2, 0.2, 1.8])
        with col1:
            st.image(image, caption=t("img_caption"), use_container_width=True)
        with col2:
            if 'description_translated' not in st.session_state:
                with st.spinner(f"🌍 Traduciendo al {idioma_seleccionado}..."):
                    anuncio(t("spinner_translate", lang=idioma_seleccionado))
                    st.session_state["description_translated"] = traducir_texto(
                        st.session_state["description_en"], idioma_objetivo=idioma_seleccionado
                    )
            st.success(t("description_title", lang=idioma_seleccionado))
            st.write(st.session_state["description_translated"])
            if 'braille_text' not in st.session_state:
                unknown_char_option = st.radio(
                    "Manejo de caracteres no soportados:",
                    ("Mantener", "Reemplazar", "Advertir")
                )
                replace_char = st.text_input("Carácter de reemplazo:", "?") if unknown_char_option == "Reemplazar" else "?"
                st.session_state["braille_text"], _ = texto_a_braille(
                    st.session_state["description_translated"],
                    unknown_char_option.lower(), replace_char
                )
            st.success(t("braille_desc_title"))
            with open("braille_descripcion.brf", "w", encoding="utf-8") as f:
                f.write(st.session_state["braille_text"])
            with open("braille_descripcion.brf", "rb") as f:
                st.download_button(t("download_in_braille"), data=f, file_name="braille_descripcion.brf", mime="text/plain")
            if "audio_file" not in st.session_state:
                st.session_state["audio_file"] = generar_audio(
                    st.session_state["description_translated"],
                    idioma=st.session_state["last_lang"], nombre_archivo="descripcion"
                )
            mostrar_boton_audio(st.session_state["audio_file"])
            st.audio(st.session_state["audio_file"])
            if st.session_state["ocr_text"]:
                st.success(t("ocr_title"))
                st.write(st.session_state["ocr_text"])
                if 'ocr_braille' not in st.session_state:
                    st.session_state["ocr_braille"], _ = texto_a_braille(
                        st.session_state["ocr_text"], opcion_caracter_desconocido="keep"
                    )
                st.success(t("braille_ocr_title"))
                st.write(st.session_state["ocr_braille"])
                with open("braille_ocr.brf", "w", encoding="utf-8") as f:
                    f.write(st.session_state["ocr_braille"])
                if "ocr_audio_file" not in st.session_state:
                    st.session_state["ocr_audio_file"] = generar_audio(
                        st.session_state["ocr_text"], idioma=st.session_state["last_lang"], nombre_archivo="ocr"
                    )
                time.sleep(7)
                mostrar_boton_audio(st.session_state["ocr_audio_file"])
                st.audio(st.session_state["ocr_audio_file"])
                with open(st.session_state["ocr_audio_file"], "rb") as audio:
                    st.download_button(t("download_ocr_audio"), data=audio, file_name="ocr.mp3", mime="audio/mpeg")
            else:
                st.warning(t("no_ocr"))

    
    with st.container():
        st.markdown(
            "<div style='display:none;' class='hidden-buttons'>",
            unsafe_allow_html=True
        )

        if shortcut_button("Generar descripción", "shift+u"):
            ejecutar_comando("generar descripción")

        if shortcut_button("Generar audio OCR", "shift+i"):
            ejecutar_comando("generar audio ocr")

        if shortcut_button("Descargar Braille ocr", "shift+o"):
            ejecutar_comando("descargar braille ocr")

        if shortcut_button("Descargar Braille Descripción", "shift+p"):
            ejecutar_comando("descargar braille descripcion")

        if shortcut_button("Ayuda", "shift+h"):
            ejecutar_comando("ayuda")

        if shortcut_button("Quitar imagen", "shift+q"):
            ejecutar_comando("quitar imagen")

        if shortcut_button("Voz", "shift+v"):
            ejecutar_comando("voz")

        if shortcut_button("Hacer pregunta", "shift+a"):
            ejecutar_comando("hacer pregunta")

        if shortcut_button("Analizar emociones", "shift+e"):
            ejecutar_comando("analizar emociones")

        # Temas de daltonismo
        if shortcut_button("Tema base", "shift+s"):
            ejecutar_comando("tema base")

        if shortcut_button("Tema protanopia", "shift+d"):
            ejecutar_comando("tema protanopia")

        if shortcut_button("Tema deuteranopia", "shift+f"):
            ejecutar_comando("tema deuteranopia")

        if shortcut_button("Tema tritanopia", "shift+g"):
            ejecutar_comando("tema tritanopia")

        if shortcut_button("Manejo de caracteres desconocidos", "shift+m"):
            ejecutar_comando("manejo de caracteres desconocidos")

        st.markdown("</div>", unsafe_allow_html=True)
if __name__ == "__main__":
    main()
