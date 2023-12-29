import cv2
import sys
import os
import face_recognition as fr
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import io
import subprocess

import firebase_admin
from firebase_admin import credentials, storage



# Firebase 데이터베이스 인증
cred = credentials.Certificate("memo-ce15c-firebase-adminsdk-qk2rg-b5e79971f6.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'memo-ce15c.appspot.com'})


# 스토리지 참조 가져오기
bucket = storage.bucket()

# 얼굴 인코딩 및 이미지 리스트에 추가
enc_face_list = []
face_list = []

folder_path = "/photos/test"
blobs = bucket.list_blobs(prefix=folder_path, delimiter='/')
# 경로를 받아오는 과정에서 오류


# Firebase Storage에서 특정 폴더의 이미지 다운로드 및 얼굴 인코딩
for filename in blobs :

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

         # 얼굴 인코딩
        face_locations = fr.face_locations(filename)

        if face_locations:
            enc_face_list.append(fr.face_encodings(filename, known_face_locations=face_locations)[0])
            face_list.append(filename)


# 중요 인물 얼굴 받아오기
image_path = input("중요 인물의 경로 입력: ")
known_person = fr.load_image_file(image_path)

# 얼굴 좌표를 알아낸 후 잘라내기
top, right, bottom, left = fr.face_locations(known_person)[0]
known_face = np.array(known_person[top:bottom, left:right])

# face_encoding()으로 얼굴 영역 인코딩해 저장
enc_known_face = fr.face_encodings(known_face)[0]  # [0]은 첫 번째 얼굴의 인코딩을 선택합니다.


# 얼굴 비교
count = 0  # 출력 횟수 조절
known_face_list = []

for face, encoding in zip(face_list, enc_face_list):
    # 최근 50장과 중요 인물 얼굴 비교
    distance = fr.face_distance([enc_known_face], encoding)

    # 결과 출력
    if distance[0] <= 0.6:
        print("유사함")
        known_face_list.append(distance[0])

    count += 1  # 3장까지만 출력

    if count >= 3:
        break


# known_face_list에 있는 이미지들을 이동
for index, distance in enumerate(known_face_list):
    # 이미지 파일의 이름을 생성
    new_filename = f"known_face_{index} {name}.jpg"
    
    # Firebase Storage에 있는 파일을 이동
    source_blob = bucket.blob(f"{folder_path}/{new_filename}")
    destination_blob = bucket.blob(f"photos/face/{new_filename}")
    bucket.copy_blob(source_blob, destination_blob)
    source_blob.delete()
    
    print(f"Image {new_filename} moved successfully.")