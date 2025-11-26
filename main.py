from flask import Flask, render_template, request
from flask_cors import CORS
import cv2
import os
import numpy as np
import base64


app= Flask(__name__)
CORS(app, origin="*")


def perspective(puntos):

    rect= np.zeros((4, 2), dtype=np.float32)

    s= puntos.sum(axis=1)

    rect[0]= puntos[np.argmin(s)]
    rect[2]= puntos[np.argmax(s)]

    diff= np.diff(puntos, axis=1)

    rect[1]= puntos[np.argmin(diff)]
    rect[3]= puntos[np.argmax(diff)]
    return rect



def procesar(foto, nombre):
    datos={}
    
    
    face_cascade= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    smile_cascade= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


    foto= foto.read()
    array= np.frombuffer(foto, dtype=np.uint8)
    foto= cv2.imdecode(array, cv2.IMREAD_COLOR)

    gray= cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray, 1.1, 7, 5)

    canny= cv2.Canny(gray, 50, 150)
    adaptive= cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    contours_canny, _= cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_adaptive, _= cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours= contours_canny + contours_adaptive

    for con in contours:
        peri= cv2.arcLength(con, True)
        approx= cv2.approxPolyDP(con, peri * 0.02, True)

        if len(approx)==4:
            puntos= approx.reshape(4, 2)
            rect= perspective(puntos)

            width, height= 400, 400
            salida= np.array([
                [0, 0],
                [width -1, 0],
                [width-1, height-1],
                [0, height-1]
            ], dtype=np.float32)

            H= cv2.getPerspectiveTransform(rect, salida)
            warp= cv2.warpPerspective(foto, H, (width, height))

    total=0
    feliz=0
    no_feliz=0

    if faces is not None:

        for (x, y, h, w) in faces:
            recorte= foto[y:y+w, x:x+h]

            smiles= smile_cascade.detectMultiScale(recorte, scaleFactor=1.1, minNeighbors=22)
            print("sonrisas: ", len(smiles))
            
            if len(smiles)>0:
                total +=1
                feliz+=1
                cv2.rectangle(foto, (x, y), (x+h, y+w), (0, 255, 0), 5)
                datos['sonrisa']="Genial, que feliz me ponen los que tienen la señal verde (derepente me caen bien) 😁😊"
            else:
                total +=1
                no_feliz+=1
                cv2.rectangle(foto, (x, y), (x+h, y+w), (0, 0, 255), 5)
                datos['sonrisa']= 'Parece bonita foto, pero ni fu ni fa 😒😏'
    datos['nombre']=nombre
    datos['conclu']=f"Veo que hay {total} persona(s), pero solo me gustan: {feliz}\n porque estan felices 😍, {'es broma. No me gusta nadie' if feliz==0 else ''}"
    _, buffer= cv2.imencode('.png', foto)
    b64= base64.b64encode(buffer).decode('utf-8')
    datos['foto']=b64
    return datos







@app.route('/', methods=['GET', 'POST'])
def index():

    return render_template('index.html')

@app.route('/procesar', methods=['POST'])
def pp():
    foto= request.files.get('foto')
    foto= procesar(foto, foto.filename)

    return render_template('index.html', resultado=foto)


app.run(host="0.0.0.0", port=5000)
