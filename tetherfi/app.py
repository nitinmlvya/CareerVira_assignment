from flask import Flask, Response
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from generate_data import normalize

app = Flask(__name__)

CLASSES = ['fake', 'real']
# === load model === #
face_liveness_model_path = 'models/model.h5'
face_liveness_model = load_model(face_liveness_model_path)

face_detection_proto_path = 'models/deploy.prototxt'
face_detection_model_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
face_detection_net = cv2.dnn.readNetFromCaffe(face_detection_proto_path, face_detection_model_path)

def transform(frame):
    return normalize(frame)

def get_bbox_of_face(frame):
    global face_detection_net
    face_bboxes = []
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    face_detection_net.setInput(blob)
    detections = face_detection_net.forward()

    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            face_bboxes.append( (start_x, start_y, end_x, end_y) )
    return face_bboxes

def drawing():
    global model
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        (h, w) = frame.shape[:2]
        if not ret:
            break

        face_bboxes = get_bbox_of_face(frame) # face detection
        print('face_bboxes: ', face_bboxes)

        frame_resized = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_LINEAR)
        frame_resized_4d = transform([frame_resized])
        y_pred = face_liveness_model.predict(frame_resized_4d)
        y_pred = y_pred.argmax(axis=1)[0]
        face_liveness_class = CLASSES[y_pred]
        # print('y_pred ', y_pred, 'face_liveness_class: ', face_liveness_class)

        for face_bbox in face_bboxes:
            frame = cv2.rectangle(frame, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (255, 0, 0), 2)
            frame = cv2.putText(frame, face_liveness_class, (face_bbox[0], face_bbox[1]), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 2, cv2.LINE_AA)

        # encode the frame in JPEG format
        (flag, encoded_image) = cv2.imencode(".jpg", frame)
        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')
    cap.release()

@app.route('/detect', methods=['GET', 'POST'])
def webhook():
    return Response(drawing(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000)