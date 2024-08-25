import streamlit as st
from PIL import Image
from processing import detect_objects, save_results_as_zip

# Подключение CSS-стилей из внешнего файла
def load_css(css_file_path):
    with open(css_file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Загружаем стили
load_css("styles.css")

st.title("Обнаружение объектов на изображении")

# Настройки для выбора порога уверенности и количества изображений
with st.sidebar:
    st.header("Настройки")
    
    # Вставляем стили для ползунка
    st.markdown("""
    <style>
        .stSlider>div>div>div>div {
            font-size: 24px; /* Увеличиваем размер шрифта ползунка */
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Порог уверенности
    confidence_threshold = st.slider("Порог уверенности", 0.0, 1.0, 0.5)
    
    # Вставляем стили для текстовых полей и ползунков
    st.markdown("""
    <style>
        .stNumberInput input,
        .stTextInput input {
            font-size: 24px; /* Увеличиваем размер шрифта */
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Количество изображений для загрузки
    num_images = st.number_input("Количество изображений для загрузки", min_value=1, max_value=100, value=3)

# Поле для ввода имени файла архива
st.markdown("""
<style>
    .stTextInput>div>input {
        font-size: 24px; /* Увеличиваем размер шрифта */
    }
</style>
""", unsafe_allow_html=True)
zip_filename = st.text_input("Введите название для архива", value="results")

# Автоматическое открытие загрузки изображений
st.markdown("""
<style>
    .stFileUploader>div>input {
        font-size: 24px; /* Увеличиваем размер шрифта */
    }
</style>
""", unsafe_allow_html=True)
uploaded_files = st.file_uploader("Загрузите изображения", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Ограничиваем количество загружаемых изображений
if uploaded_files:
    if len(uploaded_files) > num_images:
        st.warning(f"Загружено {len(uploaded_files)} изображений, но будет обрабатываться только первые {num_images}.")
        uploaded_files = uploaded_files[:num_images]

    # Загружаем изображения и обрабатываем их
    images = [Image.open(uploaded_file).convert("RGB") for uploaded_file in uploaded_files]

    if st.button("Обработать изображения"):
        results = []
        for image in images:
            annotated_image, detections = detect_objects(image, confidence_threshold)
            results.append((annotated_image, detections))
        
        # Сохраняем результаты и индекс в st.session_state
        st.session_state.results = results
        st.session_state.image_index = 0

        st.write("Результаты детекции:")

    # Если результаты уже были сгенерированы, отображаем их
    if "results" in st.session_state:
        num_results = len(st.session_state.results)
        if num_results > 0:
            # Функции для навигации между изображениями
            def go_next():
                if st.session_state.image_index < num_results - 1:
                    st.session_state.image_index += 1

            def go_prev():
                if st.session_state.image_index > 0:
                    st.session_state.image_index -= 1

            # Кнопки для переключения изображений
            col1, col2, col3 = st.columns([1, 8, 1])
            with col1:
                if st.button("◀️", key="prev", help="Перейти к предыдущему изображению", use_container_width=True):
                    go_prev()
            with col3:
                if st.button("▶️", key="next", help="Перейти к следующему изображению", use_container_width=True):
                    go_next()
            with col2:
                # Отображаем текущее изображение и его результаты
                image_index = st.session_state.image_index
                annotated_image, detections = st.session_state.results[image_index]
                st.image(annotated_image, caption=f"Аннотированное изображение {image_index + 1}", use_column_width=True)

                # Отображение результатов
                if len(detections) > 0:
                    for detection in detections:
                        class_name = detection["class"]
                        confidence = detection["confidence"]
                        st.markdown(
                            f'<p style="font-size:16px;">Объект: {class_name} Уверенность: {confidence:.2f}</p>', 
                            unsafe_allow_html=True
                        )

            # Отображаем текущий индекс изображения
            st.markdown(
                f'<p style="font-size:20px; font-weight:bold;">Изображение {st.session_state.image_index + 1} из {num_results}</p>', 
                unsafe_allow_html=True
            )
        
        # Сохранение результатов
        if st.button("Сохранить результаты"):
            zip_path, final_zip_filename = save_results_as_zip(st.session_state.results, zip_filename)
            
            # Предоставление возможности загрузить ZIP-файл
            with open(zip_path, "rb") as f:
                st.download_button(
                    label="Скачать результаты",
                    data=f,
                    file_name=final_zip_filename,
                    mime="application/zip"
                )
