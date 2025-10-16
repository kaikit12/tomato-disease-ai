# 🍅 Tomato Disease AI Assistant

Hệ thống phân loại bệnh cà chua bằng TensorFlow và Streamlit, tích hợp AI multi-model.

## ✨ Tính năng

- 🔍 **Phân loại 8 loại bệnh** cà chua bằng Deep Learning
- 📸 **Upload ảnh** hoặc **chụp trực tiếp** từ camera
- 🤖 **AI tư vấn** điều trị bằng Groq LLaMA 3.1
- 👁️ **Kiểm tra chéo** với Google Gemini Vision
- 📊 **Phân tích xác suất** chi tiết cho từng bệnh
- 📜 **Lịch sử** nhận diện và quản lý kết quả

## 🎯 Các loại bệnh được nhận diện

1. Bệnh: Nhện đỏ (spider mites)
2. Bệnh: Sương mai (late blight)  
3. Bệnh: Đốm mục tiêu (target spot)
4. Lá khỏe mạnh
5. Bệnh: Quăn lá (leaf curl)
6. Bệnh: Đốm septoria
7. Bệnh: Mốc lá (leaf mold)
8. Bệnh: Sương mai sớm (early blight)

## 🚀 Live Demo

**Streamlit Cloud:** [https://your-app.streamlit.app](https://your-app.streamlit.app)

## 🛠️ Công nghệ sử dụng

- **Frontend:** Streamlit
- **AI Model:** TensorFlow/Keras (MobileNet-based)
- **LLM:** Groq LLaMA 3.1-8B-Instant
- **Vision AI:** Google Gemini 1.5 Pro
- **Image Processing:** PIL, OpenCV
- **Deployment:** Streamlit Cloud

## 📦 Cài đặt & Chạy Local

### 1. Clone repository
```bash
git clone https://github.com/your-username/tomato-disease-ai.git
cd tomato-disease-ai
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Cấu hình API Keys
Tạo file `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your_groq_api_key"
GOOGLE_API_KEY = "your_google_api_key"
```

### 4. Thêm model file
- Download file `keras_model.h5` từ [Google Drive](link-to-your-model)
- Đặt vào thư mục `models/keras_model.h5`

### 5. Chạy ứng dụng
```bash
streamlit run app.py
```

## 🌐 Deploy lên Production

### Streamlit Community Cloud (Recommended)
1. Push code lên GitHub
2. Truy cập [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repository
4. Cấu hình secrets trong dashboard
5. Deploy!

### Railway
```bash
npm install -g @railway/cli
railway login
railway link your-project-id
railway up
```

### Render
- Connect GitHub repo
- Set build command: `pip install -r requirements.txt`
- Set start command: `streamlit run app.py --server.port $PORT`

## 📁 Cấu trúc thư mục

```
tomato-disease-ai/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── models/
│   └── keras_model.h5    # TensorFlow model (download separately)
├── .streamlit/
│   └── secrets.toml      # API keys (not in git)
├── deployment_guide.md   # Chi tiết deployment
└── README.md            # Documentation
```

## 🔑 API Keys cần thiết

1. **Groq API Key** (cho tư vấn điều trị):
   - Đăng ký tại: [console.groq.com](https://console.groq.com)
   - Free tier: 30 requests/minute

2. **Google AI API Key** (cho Gemini Vision):
   - Đăng ký tại: [makersuite.google.com](https://makersuite.google.com)
   - Free tier: Generous limits

## 📊 Model Performance

- **Accuracy:** ~95% trên test set
- **Input size:** 224x224 RGB
- **Model size:** ~80MB
- **Inference time:** <1s trên CPU

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Mở Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Liên hệ

- **Email:** your-email@example.com
- **GitHub:** [@your-username](https://github.com/your-username)
- **LinkedIn:** [Your Name](https://linkedin.com/in/your-profile)

## 🙏 Acknowledgments

- Dataset từ PlantVillage
- Streamlit community
- TensorFlow team
- Google AI & Groq AI