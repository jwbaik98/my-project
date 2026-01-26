import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os

# 1. 페이지 설정
st.set_page_config(page_title="Mushroom AI Care", page_icon="🍄", layout="wide")

# --- UI 스타일 ---
def apply_custom_css(theme):
    if theme == "Dark (블랙)":
        bg, s_bg, txt, c_bg, bord = "#0e1117", "#1c1e26", "#ffffff", "#262936", "#3e4251"
    else:
        bg, s_bg, txt, c_bg, bord = "#ffffff", "#f0f2f6", "#000000", "#f8f9fa", "#d1d8e0"
    
    st.markdown(f"""
        <style>
        header {{ visibility: hidden; height: 0px !important; }}
        .block-container {{ padding-top: 0px !important; }}
        .stApp {{ background-color: {bg}; color: {txt}; }}
        .main-header {{ background: {c_bg}; padding: 12px; border-radius: 0 0 20px 20px; border: 1px solid {bord}; text-align: center; margin-bottom: 15px; }}
        .stImage img {{ max-height: 400px; object-fit: contain; width: auto !important; margin: 0 auto; display: block; }}
        [data-testid="stSidebar"] {{ background-color: {s_bg} !important; }}
        [data-testid="stSidebar"] * {{ color: {txt} !important; }}
        .result-card {{ background: {c_bg}; padding: 10px; border-radius: 10px; border: 1px solid {bord}; margin-bottom: 8px; }}
        .mode-status {{ background: #4A90E2; color: white !important; padding: 5px; border-radius: 5px; text-align: center; font-weight: bold; }}
        </style>
    """, unsafe_allow_html=True)

# --- 사이드바 ---
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    theme = st.radio("🎨 테마", ["Dark (블랙)", "Light (화이트)"])
    st.markdown("---")
    conf_v = st.slider("🎯 민감도", 0.01, 1.0, 0.25)
    iou_v = st.slider("📏 중복 제거", 0.1, 0.9, 0.35)
    st.markdown("---")
    mode = st.selectbox("🖥️ 분석 모드", ["📸 사진 분석", "📹 실시간 영상"])
    st.markdown(f'<div class="mode-status">현재: {mode}</div>', unsafe_allow_html=True)

apply_custom_css(theme)

# 2. 모델 로드 함수 정의
@st.cache_resource
def load_yolo_model():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    # 후보 경로들
    paths = [
        os.path.join(curr_dir, "best.pt"),                   # 1순위: py파일과 같은 위치
        os.path.join(curr_dir, "..", "2. Models", "best.pt"), # 2순위: 상위 폴더 모델 폴더
        os.path.join(os.path.dirname(curr_dir), "best.pt"),  # 3순위: 상위 폴더 바로 아래
        "best.pt"                                            # 4순위: 현재 실행 경로
    ]
    
    for path in paths:
        if os.path.exists(path):
            return YOLO(path), path
    return None, None

# --- 모델 실제 로드 실행 ---
model, model_path = load_yolo_model()

if model is None:
    st.error("❌ 모델 파일(best.pt)을 찾을 수 없습니다.")
    st.info("best.pt 파일을 app 폴더(현재 파이썬 파일 옆)에 복사해 넣어주세요.")
    st.stop()
else:
    st.sidebar.success(f"모델 로드 완료: {os.path.basename(model_path)}")

# --- 메인 화면 ---
st.markdown('<div class="main-header"><h2>🍄 표고버섯 AI 스마트 진단</h2></div>', unsafe_allow_html=True)

if mode == "📸 사진 분석":
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("### 🖼️ 사진 업로드")
        f = st.file_uploader("이미지", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if f:
            img = Image.open(f)
            res = model.predict(img, conf=conf_v, iou=iou_v)
            st.image(res[0].plot(), use_container_width=True)

    with col2:
        st.write("### 📊 진단 리포트")
        if f:
            boxes = res[0].boxes
            st.markdown(f'<div class="result-card"><b>총 탐지:</b> {len(boxes)}개</div>', unsafe_allow_html=True)
            
            for i, box in enumerate(boxes):
                label = model.names[int(box.cls[0])]
                score = float(box.conf[0]) * 100
                xyxy = box.xyxy[0].tolist()
                width, height = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                img_h, img_w = res[0].orig_shape
                box_area_ratio = (width * height) / (img_w * img_h)

                if box_area_ratio > 0.25:
                    k_name, s_color, detail = "배지(배양봉)", "#4A90E2", "배양 상태 확인 중"
                elif label == "Disease":
                    k_name, s_color, detail = "병해(질병)", "#FF5252", "방제 필요"
                elif label == "Culture":
                    k_name, s_color, detail = "모판(배양)", "#4A90E2", "배양기"
                else:
                    k_name, s_color = "버섯(생육)", "#4CAF50"
                    size = np.sqrt(width**2 + height**2)
                    days = int(7 + (size / 60))
                    detail = f"생육 {min(days, 14)}일차"

                st.markdown(f"""
                <div class="result-card">
                    <b>#{i+1} {k_name}</b> <span style="color:{s_color};">●</span><br>
                    <small>확률: {score:.1f}% | {detail}</small>
                </div>
                """, unsafe_allow_html=True)

elif mode == "📹 실시간 영상":
    st.write("### 📹 실시간 관찰 및 상태 진단")
    run = st.toggle("카메라 가동")
    col1, col2 = st.columns([1, 1])
    with col1: win = st.empty()
    with col2:
        st.write("### 📊 실시간 리포트")
        report_placeholder = st.empty()

    if run:
        vid = cv2.VideoCapture(0)
        cnt = 0 
        last_report = '<div style="color:gray; text-align:center;">버섯을 비춰주세요.</div>'
        
        while run:
            ret, frame = vid.read()
            if not ret: break
            cnt += 1
            if cnt % 2 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = model.predict(frame_rgb, conf=conf_v, iou=iou_v, verbose=False)
                win.image(res[0].plot(), use_container_width=True)
                boxes = res[0].boxes
                
                if len(boxes) > 0 and cnt % 5 == 0:
                    items = []
                    items.append(f'<div style="background:#4A90E2; color:white; padding:8px; border-radius:10px; margin-bottom:10px; font-weight:bold; text-align:center;">탐지: {len(boxes)}개</div>')
                    for i, box in enumerate(boxes[:5]): 
                        raw_label = model.names[int(box.cls[0])].lower()
                        score = float(box.conf[0]) * 100
                        xyxy = box.xyxy[0].tolist()
                        width, height = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                        f_h, f_w, _ = frame_rgb.shape
                        box_area_ratio = (width * height) / (f_w * f_h)

                        if box_area_ratio > 0.25:
                            k_name, s_color, detail = "배지(배양봉)", "#4A90E2", "배양 중"
                        elif 'dis' in raw_label:
                            k_name, s_color, detail = "병해", "#FF5252", "방제 필요"
                        elif 'cul' in raw_label or 'norm' in raw_label:
                            k_name, s_color, detail = "배지", "#4A90E2", "상태 양호"
                        else:
                            k_name, s_color = "생육", "#4CAF50"
                            size = np.sqrt(width**2 + height**2)
                            days = int(7 + (size / 60))
                            detail = f"생육 {min(days, 14)}일차"

                        items.append(f'<div style="border-left:4px solid {s_color}; padding:5px 10px; border-radius:5px; margin-bottom:5px; background:rgba(128,128,128,0.05); font-size:13px;"><b>#{i+1} {k_name}</b> | {score:.1f}%<br><span style="color:#666; font-size:12px;">{detail}</span></div>')
                    last_report = "".join(items)
            report_placeholder.markdown(last_report, unsafe_allow_html=True)
        vid.release()