import streamlit as st
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
from io import BytesIO
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import plotly.graph_objects as go

# إعداد صفحة Streamlit
st.set_page_config(page_title="تصنيف الصور", page_icon="🌍", layout="wide")

# تأثيرات CSS مخصصة
st.markdown("""
<style>
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .earth-background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        opacity: 0.1;
        animation: rotate 120s linear infinite;
    }
    
    .upload-box {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
        transition: all 0.3s;
    }
    
    .upload-box:hover {
        border-color: #4CAF50;
        background-color: #f9f9f9;
    }
    
    .prediction-card {
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .loading {
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 5px solid #f3f3f3;
        border-radius: 50%;
        border-top: 5px solid #4CAF50;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# خلفية الكرة الأرضية
st.markdown("""
<div class="earth-background">
    <img src="https://cdn.pixabay.com/photo/2012/04/14/16/26/world-34414_960_720.png" 
         style="width: 100%; height: 100%; object-fit: contain;">
</div>
""", unsafe_allow_html=True)

# عنوان الصفحة
st.title("🌍 نظام تصنيف الصور الذكي")
st.markdown("قم بتحميل صورة وسنقوم بتصنيفها باستخدام الذكاء الاصطناعي")

# نموذج تصنيف الصور (يمكن استبداله بنموذجك الخاص)
def load_model():
    # هنا يمكنك تحميل نموذجك المخصص
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    return model

model = load_model()

# تحميل فئات التصنيف
def load_labels():
    labels_path = tf.keras.utils.get_file(
        'ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    )
    with open(labels_path) as f:
        labels = f.read().splitlines()
    return labels[1:]  # إزالة الفئة الأولى الفارغة

labels = load_labels()

# معالجة الصورة
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# تصنيف الصورة
def classify_image(image, model):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)
    return decoded_predictions[0]

# تحميل الصورة
def load_image(uploaded_file):
    image = Image.open(uploaded_file)
    return image

# عرض النتائج مع تأثيرات
def display_results(image, predictions):
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="الصورة المرفوعة", use_column_width=True)
        
        # رسم بياني متحرك للنتائج
        fig, ax = plt.subplots(figsize=(8, 4))
        
        classes = [p[1] for p in predictions]
        probs = [p[2] for p in predictions]
        
        def animate(i):
            ax.clear()
            ax.barh(classes[:i+1], probs[:i+1], color='#4CAF50')
            ax.set_xlim(0, 1)
            ax.set_title('نتائج التصنيف')
            ax.set_xlabel('الاحتمالية')
            
        ani = animation.FuncAnimation(fig, animate, frames=len(classes), interval=500, repeat=False)
        st.pyplot(fig)
    
    with col2:
        st.subheader("النتائج:")
        
        for i, (_, label, prob) in enumerate(predictions):
            prob_percent = prob * 100
            color = "#4CAF50" if i == 0 else "#2196F3"
            
            st.markdown(f"""
            <div class="prediction-card" style="border-left: 5px solid {color};">
                <h4>{label.replace('_', ' ').title()}</h4>
                <div style="background: #eee; border-radius: 5px; height: 20px; margin: 5px 0;">
                    <div style="background: {color}; width: {prob_percent}%; height: 100%; border-radius: 5px; 
                                text-align: right; padding-right: 5px; color: white; font-weight: bold;">
                        {prob_percent:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# واجهة تحميل الصور
uploaded_file = st.file_uploader("اختر صورة لتصنيفها...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner('جاري معالجة الصورة...'):
        # عرض مؤشر التحميل
        loading_placeholder = st.empty()
        loading_placeholder.markdown("""
        <div style="text-align: center; margin: 20px 0;">
            <div class="loading"></div>
            <p>جاري تحليل الصورة، الرجاء الانتظار...</p>
        </div>
        """, unsafe_allow_html=True)
        
        # محاكاة وقت المعالجة للتأثير البصري
        time.sleep(1)
        
        # تحميل وتصنيف الصورة
        image = load_image(uploaded_file)
        predictions = classify_image(image, model)
        
        # إزالة مؤشر التحميل
        loading_placeholder.empty()
        
        # عرض النتائج
        display_results(image, predictions)
        
        # تأثير نجاح
        st.balloons()
else:
    st.info("الرجاء تحميل صورة لتصنيفها")

# تذييل الصفحة
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 50px;">
    <p>تم تطوير هذا النظام باستخدام تقنيات الذكاء الاصطناعي و Streamlit</p>
    <p>🌍 نسعى لجعل العالم أكثر ذكاءً</p>
</div>
""", unsafe_allow_html=True)
