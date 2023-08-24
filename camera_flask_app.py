from flask import Flask, render_template, Response, request, jsonify
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
from HeartRate import startRun, resetRun
from cv2 import dnn_superres

global reset, start, bpm, dim, camera
reset=0
start=0
bpm = 0
dim = (0,0)

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#Load pretrained face detection model    
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

# model for super resolution
# sr = dnn_superres.DnnSuperResImpl_create()
# path = "./saved_model/ESPCN_x2.pb"
# sr.readModel(path)
# sr.setModel("espcn", 2)
# sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')

camera = cv2.VideoCapture(0)
# camera = cv2.VideoCapture('http://192.168.237.87:8080/video')

def detect_face(frame):
    global net, dim
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:            
            return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = (370, 480)
        frame=cv2.resize(frame,dim)
        # print(dim)
    except Exception as e:
        pass
    return frame
 

def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame, bpm, dim, camera
    # camera = cv2.VideoCapture(0)
    while True:
        success, frameTemp = camera.read() 
        # frame = cv2.rotate(frameTemp, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = frameTemp
        # frame = cv2.resize(frameTemp, (600, 400))
        if success:
            if(start):           
                # result = sr.upsample(frame)
                # frame = detect_face(result)
                frame= detect_face(frame)
                bpmTemp = startRun(frame)
                if(bpmTemp > 60) and (bpmTemp < 100):
                    bpm = bpmTemp
                    print("Heart Rate: " + str(bpm))
                    # render_template("hr.html", msg="Heart Rate: " + str(bpm))
            if(reset):
                frame = frame
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                # yield (b'--frame\r\n'
                #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                yield (b'--frame\r\n'
                        b'Content-Type:image/jpeg\r\n'
                        b'Content-Length: ' + f"{len(frame)}".encode() + b'\r\n'
                        b'\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass


@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        
        if  request.form.get('reset') == 'Reset':
            global reset

        elif  request.form.get('start') == 'Start/Stop':
            global start, bpm
            start=not start 
            bpm = 0
            if(start):
                time.sleep(4)
            resetRun()     
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')

@app.route('/heart_rate')
def heart_rate():
    global bpm
    return jsonify({'bpm': bpm})


if __name__ == '__main__':
    app.run(host='0.0.0.0')
    
camera.release()
cv2.destroyAllWindows()     