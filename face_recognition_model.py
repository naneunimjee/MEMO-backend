import cv2
import os
import face_recognition as fr
from matplotlib import pyplot as plt
import numpy as np

#### 1. 사진첩의 최근 50장 가져오기 ####

folder_path = input("사진첩 경로: ")

# 폴더 내의 파일 목록을 얻어옴
image_files = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".jpeg", ".png"))]

# 최근 50개의 이미지 파일만 선택
recent_images = sorted([os.path.join(folder_path, f) for f in image_files], key=os.path.getmtime, reverse=True)[:50]

# 얼굴 인코딩 및 이미지 리스트에 추가
enc_face_list = []
face_list = []

for image_path in recent_images:
    image = fr.load_image_file(image_path)
    face_locations = fr.face_locations(image)
    
    if face_locations:
        enc_face_list.append(fr.face_encodings(image, known_face_locations=face_locations)[0])
        face_list.append(image)

#### 2. 중요 인물 얼굴 받아오기 ####

image_path = input("중요 인물의 경로 입력: ")
known_person = fr.load_image_file(image_path)

# 얼굴 좌표를 알아낸 후 잘라내기
top, right, bottom, left = fr.face_locations(known_person)[0]
known_face = np.array(known_person[top:bottom, left:right])

# face_encoding()으로 얼굴 영역 인코딩해 저장
enc_known_face = fr.face_encodings(known_face)[0]  # [0]은 첫 번째 얼굴의 인코딩을 선택합니다.

#### 3. 얼굴 비교 ####

# 등록된 얼굴 리스트 비교
for face in face_list:
    # 최근 50장과 중요 인물 얼굴 받아오기
    distance = fr.face_distance([enc_known_face], fr.face_encodings(face)[0])

    # distance 수치를 포함한 얼굴 출력
    plt.title("distance : " + str(distance))
    plt.imshow(face)
    plt.show()

    # 결과 출력
    if distance >= 0.6:
        print("다른 사람")
    else:
        print("유사")
    print()