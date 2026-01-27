import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
import json

# 1. 페이지 설정
st.set_page_config(page_title="Mushroom AI Care", page_icon="🍄", layout="wide")

# --- UI 스타일 함수 ---
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
        .main-header {{ background: {c_bg}; padding: 12px; border-radius: 0 0 20px 20px; border: 1px solid {bord}; text-align: center; margin-bottom: 15px; color: {txt}; }}
        .stImage img {{ max-height: 400px; object-fit: contain; width: auto !important; margin: 0 auto; display: block; }}
        [data-testid="stSidebar"] {{ background-color: {s_bg} !important; }}
        [data-testid="stSidebar"] * {{ color: {txt} !important; }}
        .result-card {{ background: {c_bg}; padding: 10px; border-radius: 10px; border: 1px solid {bord}; margin-bottom: 8px; color: {txt}; }}
        .mode-status {{ background: #4A90E2; color: white !important; padding: 5px; border-radius: 5px; text-align: center; font-weight: bold; }}
        </style>
    """, unsafe_allow_html=True)
    return bg, s_bg, txt, c_bg, bord # 변수를 반환하여 다른 곳에서 쓸 수 있게 함

# --- 유틸리티 함수 (공정 판독 등) ---
def classify_process_simple(box_w, box_h):
    aspect_ratio = box_h / box_w if box_w > 0 else 0
    if aspect_ratio > 1.4:
        return "Incubation", "배양 단계"
    return "Growth", "생육 단계"

# --- 2. 모델 및 매핑 데이터 로드 ---
@st.cache_resource
def load_yolo_model():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    paths = [os.path.join(curr_dir, "best.pt"), "best.pt"]
    for path in paths:
        if os.path.exists(path):
            return YOLO(path), path
    return None, None

@st.cache_data
def load_disease_mapping():
    mapping_path = os.path.join(os.path.dirname(__file__), "mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# 모델 로드
model, model_path = load_yolo_model()
disease_lookup = load_disease_mapping()

# --- 3. 사이드바 설정 ---
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    # 순서를 Light를 앞으로 두어 화이트 버전으로 기본 고정
    theme = st.radio("🎨 테마", ["Light (화이트)", "Dark (블랙)"], index=0)
    st.markdown("---")
    conf_v = st.slider("🎯 민감도", 0.01, 1.0, 0.25)
    iou_v = st.slider("📏 중복 제거", 0.1, 0.9, 0.35)
    st.markdown("---")
    mode = st.selectbox("🖥️ 분석 모드", ["📸 사진 분석", "📹 실시간 영상"])
    st.markdown(f'<div class="mode-status">현재: {mode}</div>', unsafe_allow_html=True)

# 테마 적용 및 색상 변수 가져오기
bg, s_bg, txt, c_bg, bord = apply_custom_css(theme)

if model is None:
    st.error("❌ 모델 파일(best.pt)이 없습니다.")
    st.stop()

# --- 4. 메인 화면 로직 ---
st.markdown(f'<div class="main-header"><h2>🍄 표고버섯 AI 스마트 진단</h2></div>', unsafe_allow_html=True)

if mode == "📸 사진 분석":
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("### 🖼️ 사진 업로드")
        f = st.file_uploader("이미지", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if f:
            img = Image.open(f)
            res = model.predict(img, conf=conf_v, iou=iou_v)
            st.image(res[0].plot(), use_container_width=True)
            boxes = res[0].boxes

    with col2:
        st.write("### 📊 진단 리포트")
        if f and 'boxes' in locals():
            st.markdown(f'<div class="result-card"><b>총 탐지:</b> {len(boxes)}개</div>', unsafe_allow_html=True)
            has_growth = any(any(k in model.names[int(b.cls[0])].lower() for k in ['grow', 'mush', 'obj']) for b in boxes)

            for i, box in enumerate(boxes):
                label_idx = int(box.cls[0])
                raw_label = model.names[label_idx]
                label = raw_label.lower()
                score = float(box.conf[0]) * 100
                xyxy = box.xyxy[0].tolist()
                w, h = xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]
                img_h, img_w = res[0].orig_shape
                ratio = (w * h) / (img_w * img_h)
                aspect = h / w if w > 0 else 0
                size = np.sqrt(w**2 + h**2)
                days = int(7 + (size / 60))
                growth_detail = f"생육 {min(days, 14)}일차"

                if 'dis' in label:
                    file_key = f.name.split('.')[0]
                    disease_name = disease_lookup.get(file_key, "미분류 병해")
                    k_name, s_color = f"병해({disease_name})", "#FF5252"
                    detail = f"⚠️ 즉시 격리 및 방제 필요 | {growth_detail}"
                elif any(k in label for k in ['grow', 'obj', 'mush']):
                    k_name, s_color = "버섯(생육)", "#4CAF50"
                    detail = f"{growth_detail} (정상)"
                elif ratio > 0.25 or aspect > 1.6 or 'cul' in label:
                    if has_growth: continue
                    k_name, s_color, detail = "배지(배양봉)", "#4A90E2", "배양 상태 확인 중"
                else:
                    k_name, s_color, detail = f"기타({raw_label})", "#888888", growth_detail

                st.markdown(f"""
                <div class="result-card">
                    <b>#{i+1} {k_name}</b> <span style="color:{s_color};">●</span><br>
                    <small>분류: {raw_label} | 확신도: {score:.1f}%</small><br>
                    <div style="margin-top:5px; font-weight:bold; color:{s_color};">{detail}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("사진을 업로드하면 실시간 리포트가 생성됩니다.")

elif mode == "📹 실시간 영상":
    st.write("### 📹 실시간 통합 모니터링")
    run_cam = st.toggle("카메라 가동")
    
    col_vid, col_rep = st.columns([1.5, 1])
    with col_vid:
        st_frame = st.empty() 
    with col_rep:
        st.subheader("📋 실시간 분석 리포트")
        report_area = st.empty() 

    if run_cam:
        cap = cv2.VideoCapture(0)
        while cap.isOpened() and run_cam:
            ret, frame = cap.read()
            if not ret:
                st.error("카메라 연결에 실패했습니다.")
                break
            
            res = model.predict(frame, conf=conf_v, iou=iou_v, verbose=False)
            all_cards_html = ""
            
            if len(res[0].boxes) > 0:
                for i, box in enumerate(res[0].boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2-x1, y2-y1
                    label_name = model.names[int(box.cls[0])].lower()
                    proc_key, proc_name = classify_process_simple(w, h)
                    is_dis = any(k in label_name for k in ['dis', 'mold', 'spot', 'wilt'])
                    
                    if is_dis:
                        status_title, status_color = "⚠️ 질병 의심", "#FF5252"
                        detail_text = "즉시 격리 필요"
                    else:
                        status_title, status_color = "✅ 정상 생육", "#4CAF50"
                        size = np.sqrt(w**2 + h**2)
                        days = int(7 + (size / 60))
                        detail_text = f"생육 {min(days, 14)}일차"

                    all_cards_html += f"""
                    <div class="result-card" style="border-left: 6px solid {status_color};">
                        <b style="color: {status_color}; font-size: 1.1rem;">#{i+1} {status_title}</b><br>
                        <div style="margin-top: 5px; font-size: 0.8rem;">
                            <b>구분:</b> {proc_name} | <b>상태:</b> {detail_text}
                        </div>
                    </div>
                    """
            else:
                all_cards_html = f"<p style='color:{txt};'>현재 탐지된 객체가 없습니다.</p>"

            st_frame.image(res[0].plot(), channels="BGR", use_container_width=True)
            report_area.markdown(all_cards_html, unsafe_allow_html=True)
        cap.release()