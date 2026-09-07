import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="جدول المقارنة التفصيلي للتوقيعات", layout="wide"
)

st.title("✍️ جدول المقارنة والتكبير التفصيلي (Side-by-Side Analysis)")

col1, col2 = st.columns(2)
with col1:
    orig_file = st.file_uploader(
        "رفع التوقيع الأصلي", type=["jpg", "png", "jpeg"]
    )
with col2:
    susp_file = st.file_uploader(
        "رفع التوقيع المشتبه به", type=["jpg", "png", "jpeg"]
    )

if st.button("🚀 GENERATE / إنشاء جدول المقارنة") and orig_file and susp_file:
    img1 = np.array(Image.open(orig_file).convert("RGB"))
    img2 = np.array(Image.open(susp_file).convert("RGB"))

    # توحيد أبعاد الصورتين
    h, w = 250, 500
    img1_res = cv2.resize(img1, (w, h))
    img2_res = cv2.resize(img2, (w, h))

    # --- 1. الصف الأول: وضع دوائر حمراء على الأجزاء العلوية/المختلفة ---
    r1_left = img1_res.copy()
    r1_right = img2_res.copy()
    cv2.circle(r1_left, (int(w * 0.65), int(h * 0.25)), 35, (255, 0, 0), 2)
    cv2.circle(r1_right, (int(w * 0.85), int(h * 0.25)), 35, (255, 0, 0), 2)

    # --- 2. الصف الثاني: تحديد مقطع محدد بمربع أحمر ---
    r2_left = img1_res.copy()
    r2_right = img2_res.copy()
    # مربعات تركيز
    cv2.rectangle(
        r2_left,
        (int(w * 0.4), int(h * 0.5)),
        (int(w * 0.55), int(h * 0.8)),
        (255, 0, 0),
        2,
    )
    cv2.rectangle(
        r2_right,
        (int(w * 0.15), int(h * 0.45)),
        (int(w * 0.3), int(h * 0.75)),
        (255, 0, 0),
        2,
    )

    # --- 3. الصف الثالث: اقتصاص المقاطع وتكبيرها مع أسهم الإشارة ---
    r3_left = img1_res.copy()
    r3_right = img2_res.copy()

    # اقتصاص الجزء المرتكز
    crop1 = img1_res[
        int(h * 0.5) : int(h * 0.8), int(w * 0.4) : int(w * 0.55)
    ]
    crop1_zoom = cv2.resize(crop1, (130, 80))

    # دمج التكبير المباشر على الصور
    cv2.rectangle(
        r3_left,
        (int(w * 0.4), int(h * 0.5)),
        (int(w * 0.55), int(h * 0.8)),
        (255, 0, 0),
        2,
    )
    cv2.rectangle(
        r3_right,
        (int(w * 0.15), int(h * 0.45)),
        (int(w * 0.3), int(h * 0.75)),
        (255, 0, 0),
        2,
    )

    # رسم إطار واسهم على الصورة الأولى
    r3_left[
        int(h * 0.4) : int(h * 0.4) + 80, int(w * 0.6) : int(w * 0.6) + 130
    ] = crop1_zoom
    cv2.rectangle(
        r3_left,
        (int(w * 0.6), int(h * 0.4)),
        (int(w * 0.6) + 130, int(h * 0.4) + 80),
        (255, 0, 0),
        2,
    )
    cv2.arrowedLine(
        r3_left,
        (int(w * 0.47), int(h * 0.5)),
        (int(w * 0.6), int(h * 0.45)),
        (255, 0, 0),
        2,
    )

    # --- 4. الصف الرابع: تحديد الامتداد السفلي لشكل بيضاوي أحمر ---
    r4_left = img1_res.copy()
    r4_right = img2_res.copy()
    cv2.ellipse(
        r4_left,
        (int(w * 0.5), int(h * 0.75)),
        (int(w * 0.45), int(h * 0.15)),
        0,
        0,
        360,
        (255, 0, 0),
        2,
    )
    cv2.ellipse(
        r4_right,
        (int(w * 0.7), int(h * 0.65)),
        (int(w * 0.25), int(h * 0.15)),
        -15,
        0,
        360,
        (255, 0, 0),
        2,
    )

    # --- 5. تجميع الجدول النهائي وتوليده ---
    grid_rows = []
    pairs = [
        (r1_left, r1_right),
        (r2_left, r2_right),
        (r3_left, r3_right),
        (r4_left, r4_right),
    ]

    for left, right in pairs:
        # إضافة فاصل أبيض بين العمودين وفي أسفل كل صف
        divider_v = np.ones((h, 10, 3), dtype=np.uint8) * 200
        row = np.hstack((left, divider_v, right))
        divider_h = np.ones((10, row.shape[1], 3), dtype=np.uint8) * 200
        grid_rows.append(row)
        grid_rows.append(divider_h)

    final_table = np.vstack(grid_rows[:-1])

    st.markdown("### 📋 النتيجة النهائية (Grid Verification Table):")
    st.image(final_table, use_container_width=True)

    # زر تنزيل النتيجة
    _, encoded_img = cv2.imencode(
        ".png", cv2.cvtColor(final_table, cv2.COLOR_RGB2BGR)
    )
    st.download_button(
        "📥 تنزيل صورة الجدول (PNG)",
        data=encoded_img.tobytes(),
        file_name="Signature_Comparison_Table.png",
        mime="image/png",
    )
