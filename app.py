import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Face Blur App", layout="centered")

st.title("얼굴 자동 블러 처리기")
st.write("사진을 업로드하면 얼굴을 인식해서 자동으로 블러 처리해 줍니다.")

mp_face_detection = mp.solutions.face_detection


def blur_faces_bgr(image_bgr: np.ndarray, blur_ksize: int = 31):
    """
    image_bgr: OpenCV BGR 이미지
    blur_ksize: 가우시안 블러 커널 크기(홀수)
    return: (블러 처리된 BGR 이미지, 감지된 얼굴 개수)
    """
    if blur_ksize % 2 == 0:
        blur_ksize += 1  # 가우시안 커널은 홀수여야 함

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    detections_count = 0

    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    ) as face_detection:
        results = face_detection.process(img_rgb)

        if not results.detections:
            return image_bgr, 0

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            x_min = int(max(bbox.xmin * w, 0))
            y_min = int(max(bbox.ymin * h, 0))
            x_max = int(min((bbox.xmin + bbox.width) * w, w))
            y_max = int(min((bbox.ymin + bbox.height) * h, h))

            # ROI 범위 체크
            if x_max <= x_min or y_max <= y_min:
                continue

            face_roi = image_bgr[y_min:y_max, x_min:x_max]
            if face_roi.size == 0:
                continue

            blurred_roi = cv2.GaussianBlur(
                face_roi, (blur_ksize, blur_ksize), 0
            )
            image_bgr[y_min:y_max, x_min:x_max] = blurred_roi
            detections_count += 1

    return image_bgr, detections_count


uploaded_file = st.file_uploader(
    "이미지 파일을 업로드하세요 (JPG, PNG)", type=["jpg", "jpeg", "png"]
)

blur_strength = st.slider(
    "블러 강도 (커널 크기)", min_value=11, max_value=81, value=31, step=2
)

if uploaded_file is not None:
    # 파일을 OpenCV 이미지(BGR)로 읽기
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("이미지를 읽을 수 없습니다. 다른 파일을 시도해 주세요.")
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        st.subheader("원본 이미지")
        st.image(img_rgb, use_column_width=True)

        if st.button("얼굴 블러 처리 실행"):
            with st.spinner("얼굴을 감지하고 블러 처리 중입니다..."):
                output_bgr, face_count = blur_faces_bgr(
                    img_bgr.copy(), blur_ksize=blur_strength
                )
                output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

            st.subheader("블러 처리 결과")
            st.image(
                output_rgb,
                use_column_width=True,
                caption=f"감지된 얼굴 수: {face_count}개",
            )

            # 다운로드용 버퍼 생성
            pil_img = Image.fromarray(output_rgb)
            buf = BytesIO()
            pil_img.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="결과 이미지 다운로드 (PNG)",
                data=byte_im,
                file_name="blurred_faces.png",
                mime="image/png",
            )

elif uploaded_file is None:
    st.info("위의 버튼을 눌러 이미지를 업로드하세요.")
