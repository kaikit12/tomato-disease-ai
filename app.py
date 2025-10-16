import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import io
from streamlit_image_comparison import image_comparison
from groq import Groq
from datetime import datetime
import google.generativeai as genai

st.set_page_config(page_title="Trợ lý Cà chua AI", page_icon="🍅", layout="wide")

# API Keys từ Streamlit Secrets với error handling
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
except Exception as e:
    st.error(f"❌ Lỗi đọc secrets: {e}")
    GROQ_API_KEY = ""
    GOOGLE_API_KEY = ""

# Initialize Groq client với error handling
groq_client = None
if GROQ_API_KEY and GROQ_API_KEY.strip():
    try:
        groq_client = Groq(api_key=GROQ_API_KEY.strip())
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi Groq: {e}")
        groq_client = None
else:
    st.sidebar.warning("⚠️ Groq API chưa cấu hình")

# Initialize Google AI với error handling
if GOOGLE_API_KEY and GOOGLE_API_KEY.strip():
    try:
        genai.configure(api_key=GOOGLE_API_KEY.strip())
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi Google AI: {e}")
else:
    st.sidebar.warning("⚠️ Google API chưa cấu hình")

DISEASE_LIBRARY = {
    'Bệnh: Nhện đỏ (spider mites)': {
        'description': 'Nhện đỏ rất nhỏ, sống ở mặt dưới lá, hút nhựa cây, làm lá có đốm vàng, bạc.',
        'symptoms': '- Lá có chấm li ti vàng/trắng.\n- Có thể có mạng nhện mịn mặt dưới lá.'
    },
    'Bệnh: Sương mai (late blight)': {
        'description': 'Bệnh nguy hiểm do nấm, phát triển mạnh khi ẩm, mát.',
        'symptoms': '- Đốm xanh xám, úng nước trên lá.\n- Vết bệnh lan nhanh, chuyển màu nâu đen.'
    },
    'Bệnh: Đốm mục tiêu (target spot)': {
        'description': 'Bệnh do nấm gây ra, tạo thành các đốm tròn có vòng tròn đồng tâm.',
        'symptoms': '- Đốm tròn màu nâu có vòng tròn.\n- Lá vàng và rụng sớm.'
    },
    'Lá khỏe mạnh': {
        'description': 'Lá không có dấu hiệu bệnh, màu xanh đều.',
        'symptoms': '- Bề mặt lá nhẵn, không đốm, mốc.\n- Màu sắc xanh tươi.'
    },
    'Bệnh: Quăn lá (leaf curl)': {
        'description': 'Bệnh do virus gây ra, làm lá quăn cong và biến dạng.',
        'symptoms': '- Lá quăn cong bất thường.\n- Màu lá nhạt, vàng úa.'
    },
    'Bệnh: Đốm septoria': {
        'description': 'Bệnh do nấm Septoria lycopersici, thường xuất hiện ở lá già.',
        'symptoms': '- Đốm tròn nhỏ màu xám với viền đen.\n- Lá vàng và rụng từ dưới lên.'
    },
    'Bệnh: Mốc lá (leaf mold)': {
        'description': 'Bệnh do nấm, phát triển trong điều kiện ẩm ướt.',
        'symptoms': '- Lớp mốc vàng ở mặt dưới lá.\n- Lá héo và chết dần.'
    },
    'Bệnh: Sương mai sớm (early blight)': {
        'description': 'Bệnh do nấm Alternaria solani, ảnh hưởng đến lá và quả.',
        'symptoms': '- Đốm nâu có vòng tròn đồng tâm.\n- Lá vàng và rụng sớm.'
    }
}

if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_keras_model():
    """Load model từ GitHub hoặc local"""
    # Thay đổi path model cho production
    model_path = "models/keras_model.h5"
    
    # Check if model exists before loading
    if not os.path.exists(model_path):
        st.error(f"⚠️ Không tìm thấy file mô hình tại {model_path}")
        st.info("💡 Vui lòng upload file model vào thư mục 'models/' với tên 'keras_model.h5'")
        st.stop()
    
    def custom_depthwise_conv2d(*args, **kwargs):
        kwargs.pop('groups', None)
        return tf.keras.layers.DepthwiseConv2D(*args, **kwargs)
    
    try:
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d}, 
            compile=False
        )
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"❌ Lỗi khi tải mô hình: {e}")
        st.stop()

# Chỉ load model khi có file
if os.path.exists("models/keras_model.h5"):
    model = load_keras_model()
    class_names = [
        'Bệnh: Nhện đỏ (spider mites)', 
        'Bệnh: Sương mai (late blight)', 
        'Bệnh: Đốm mục tiêu (target spot)', 
        'Lá khỏe mạnh', 
        'Bệnh: Quăn lá (leaf curl)', 
        'Bệnh: Đốm septoria', 
        'Bệnh: Mốc lá (leaf mold)', 
        'Bệnh: Sương mai sớm (early blight)'
    ]
else:
    model = None
    st.error("❌ Model chưa được upload. Vui lòng thêm file 'keras_model.h5' vào thư mục 'models/'")

def predict_image(image):
    if model is None:
        return "Model chưa được tải", 0, []
    
    img = load_img(image, target_size=(224, 224))
    img_array = img_to_array(img) / 127.5 - 1
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    return predicted_class, confidence, predictions[0]

def draw_result(image_file, result_text, confidence):
    image = Image.open(image_file).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()
    
    text = f"{result_text} ({confidence:.2f}%)"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    draw.rectangle([(10, 10), (10 + text_width + 20, 10 + text_height + 20)], fill="rgba(0,0,0,128)")
    draw.text((20, 20), text, fill="white", font=font)
    return image

@st.cache_data
def get_treatment_suggestion(disease_name: str) -> str:
    if not groq_client:
        return "⚠️ Groq API chưa được cấu hình hoặc có lỗi kết nối."
    
    if disease_name == 'Lá khỏe mạnh':
        return "✅ Tuyệt vời! Lá cây của bạn khỏe mạnh."
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia nông nghiệp Việt Nam, tư vấn cách trị bệnh cà chua ngắn gọn cho nông dân."},
                {"role": "user", "content": f"Cây cà chua của tôi bị '{disease_name}'. Đề xuất phương pháp trị và phòng bệnh."}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Lỗi API tư vấn: {str(e)}"

def get_vision_ai_check(image_bytes: bytes) -> str:
    if not GOOGLE_API_KEY or not GOOGLE_API_KEY.strip():
        return "⚠️ Google API Key chưa được cấu hình."
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
        model_ai = genai.GenerativeModel('gemini-1.5-pro-latest')
        
        prompt_parts = [
            "Bạn là một chuyên gia chẩn đoán bệnh cây trồng qua hình ảnh. Hãy phân tích kỹ lưỡng ảnh lá cà chua này.\n",
            "1. Chẩn đoán xem lá cây bị bệnh gì hoặc là lá khỏe mạnh.\n",
            "2. Liệt kê các triệu chứng cụ thể bạn nhìn thấy trên lá (ví dụ: đốm vàng, viền nâu, lá quăn...).\n",
            "3. Đưa ra kết luận một cách ngắn gọn, súc tích.\n\n",
            img,
        ]
        
        response = model_ai.generate_content(prompt_parts)
        return f"**Đánh giá từ Google Gemini Vision:**\n\n" + response.text
    except Exception as e:
        return f"❌ Lỗi khi gọi API Google Gemini Vision: {str(e)}"

# UI
st.title("🍅 Trợ lý Cà chua AI")
st.markdown("*Phân loại bệnh cà chua bằng AI - Phiên bản Production*")

with st.sidebar:
    st.header("📖 Thư viện bệnh học")
    selected_disease = st.selectbox("Tra cứu thông tin bệnh:", list(DISEASE_LIBRARY.keys()))
    
    if selected_disease:
        info = DISEASE_LIBRARY[selected_disease]
        st.subheader(selected_disease)
        st.markdown(f"**Mô tả:** {info['description']}")
        st.markdown(f"**Triệu chứng:**\n{info['symptoms']}")
    
    st.markdown("---")
    st.header("📜 Lịch sử nhận diện")
    
    if not st.session_state.history:
        st.info("Chưa có lịch sử nào.")
    else:
        for i in range(len(st.session_state.history) - 1, -1, -1):
            record = st.session_state.history[i]
            col1, col2 = st.columns([4, 1])
            
            with col1:
                with st.expander(f"{record['prediction']} ({record['time']})"):
                    st.image(record['image'], width=100)
                    st.write(f"Độ tin cậy: {record['confidence']:.2f}%")
            
            with col2:
                if st.button("🗑️", key=f"delete_{i}_{record['time']}", help="Xóa mục này"):
                    del st.session_state.history[i]
                    st.rerun()

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("① Tải hoặc Chụp ảnh")
    uploaded_file = st.file_uploader("Tải ảnh từ thiết bị:", type=["jpg", "jpeg", "png"])
    camera_file = st.camera_input("Chụp ảnh từ camera:")
    
    image_to_process = camera_file or uploaded_file
    if image_to_process:
        st.image(image_to_process, caption="Ảnh được chọn", use_container_width=True)

with col2:
    st.subheader("② Xem kết quả phân tích")
    
    if image_to_process and model is not None:
        with st.spinner("⏳ AI đang phân tích, vui lòng chờ..."):
            predicted_class, confidence, probabilities = predict_image(image_to_process)
            result_image = draw_result(image_to_process, predicted_class, confidence)
        
        current_time = datetime.now().strftime("%H:%M:%S")
        new_record = {
            "image": image_to_process.getvalue(),
            "prediction": predicted_class,
            "confidence": confidence,
            "time": current_time
        }
        
        if not st.session_state.history or st.session_state.history[-1]['prediction'] != new_record['prediction']:
            st.session_state.history.append(new_record)
        
        if len(st.session_state.history) > 10:
            st.session_state.history.pop(0)
        
        tabs = st.tabs(["📊 Kết quả chính", "↔️ So sánh ảnh", "🧑‍🌾 Tư vấn AI (Groq)", "🔍 Kiểm tra chéo (Gemini)"])
        
        with tabs[0]:
            st.metric("Chẩn đoán (Model Tự huấn luyện)", predicted_class)
            st.metric("Độ tin cậy", f"{confidence:.2f}%")
            st.markdown("---")
            st.markdown("##### Phân bổ xác suất:")
            
            for cls, prob in zip(class_names, probabilities):
                st.write(f"{cls}: {prob*100:.2f}%")
                st.progress(float(prob))
            
            buf = io.BytesIO()
            result_image.save(buf, format="PNG")
            st.download_button("⬇️ Tải ảnh kết quả", buf.getvalue(), "ket_qua.png", "image/png")
        
        with tabs[1]:
            st.markdown("Kéo thanh trượt để so sánh.")
            image_comparison(Image.open(image_to_process), result_image, "Ảnh gốc", "Ảnh có chẩn đoán")
        
        with tabs[2]:
            st.info("Nhận gợi ý chi tiết từ AI LLaMA 3.1 qua Groq.")
            if st.button("💡 Nhận gợi ý trị bệnh"):
                with st.spinner("🤖 Groq AI đang soạn thảo..."):
                    st.markdown(get_treatment_suggestion(predicted_class))
        
        with tabs[3]:
            st.info("Sử dụng Google Gemini Vision để có thêm góc nhìn thứ hai.")
            if st.button("🔬 Bắt đầu kiểm tra chéo với Gemini"):
                with st.spinner("🛰️ Gemini Vision đang phân tích ảnh..."):
                    st.markdown(get_vision_ai_check(image_to_process.getvalue()))
    
    elif image_to_process and model is None:
        st.error("❌ Model chưa được tải. Vui lòng kiểm tra file model.")
    else:
        st.info("👈 Vui lòng tải ảnh lên hoặc sử dụng camera để bắt đầu.")

# Footer
st.markdown("---")
st.markdown("**🍅 Tomato Disease AI Assistant** - Powered by TensorFlow & Streamlit")