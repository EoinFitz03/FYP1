import sys, os
sys.path.append(r"C:/Users/EOIN/AppData/Local/Programs/Python/Python312/Lib/site-packages")

import face_recognition_models
print(" Models module loaded manually:", os.path.dirname(face_recognition_models.__file__))

import face_recognition
print(" face_recognition loaded successfully")
