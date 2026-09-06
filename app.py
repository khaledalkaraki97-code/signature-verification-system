import cv2
import numpy as np
import streamlit as st
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# إعدادات الصفحة المتقدمة
st.set_page_config(
    page_title="Forensic Signature Verification System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# تخصيص واجهة المستخدم
st.title("🔬 نظام التحليل الجنائي والتحقق من صحة التوقيعات")
st.caption(
    "Automated Forensic Signature Verification & Structural Difference Mapping System"
)
st.markdown("---")

# --- الشريط الجانبي لوضع الإعدادات المتقدمة ---
st.sidebar.header("⚙️ إعدادات المعالجة الجنائية")
sensitivity = st.sidebar.slider(
    "حساسية كشف الفروقات (SSIM Threshold)", 0.1, 1.0, 0.5, 0.05
)
min_area = st.sidebar.slider(
    "تصفية التشويش (Min Contour Area)", 10, 200, 40, 10
)
show_heatmap = st.sidebar.checkbox("عرض خريطة الحرارة (Heatmap)", value=True)

# --- 1. رفع الصور ---
st.subheader("📁 1. مدخلات العينات (Signature Samples)")
col_up1, col_up2 = st.columns(2)

with col_up1:
    orig_file = st.file_uploader(
        "رفع التوقيع الأصلي المرجعي (Reference Signature)",
        type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"],
        key="orig_file_input",
    )
with col_up2:
    susp_file = st.file_uploader(
        "رفع التوقيع المشتبه به (Questioned Signature)",
        type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"],
        key="susp_file_input",
    )

if orig_file and susp_file:
    img_orig_raw = Image.open(orig_file).convert("RGB")
    img_susp_raw = Image.open(susp_file).convert("RGB")

    with st.expander("👁️ عرض العينات الحالية قبل المقارنة", expanded=True):
        c1, c2 = st.columns(2)
        c1.image(
            img_orig_raw,
            caption="التوقيع الأصلي (Reference)",
            use_container_width=True,
        )
        c2.image(
            img_susp_raw,
            caption="التوقيع المشتبه به (Questioned)",
            use_container_width=True,
        )


def process_signature_mask(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return gray, mask


# --- 2. تنفيذ التحليل الجنائي ---
if st.button("🚀 GENERATE / إجراء التحليل الشامل") and orig_file and susp_file:
    with st.spinner("جاري استخراج الخصائص البنيوية والحسابات الجنائية..."):
        np_orig = np.array(img_orig_raw)
        np_susp = np.array(img_susp_raw)

        gray_orig, mask_orig = process_signature_mask(np_orig)
        gray_susp, mask_susp = process_signature_mask(np_susp)

        # محاذاة الأبعاد
        h, w = gray_orig.shape
        gray_susp_resized = cv2.resize(gray_susp, (w, h))
        mask_susp_resized = cv2.resize(mask_susp, (w, h))

        # حساب SSIM
        score, diff = ssim(gray_orig, gray_susp_resized, full=True)
        similarity_pct = round(score * 100, 2)
        difference_pct = round(100.0 - similarity_pct, 2)

        # حساب المقاييس الجنائية الإضافية (Forensic Metrics)
        pixels_orig = cv2.countNonZero(mask_orig)
        pixels_susp = cv2.countNonZero(mask_susp_resized)
        density_ratio = round((pixels_susp / max(pixels_orig, 1)) * 100, 2)

        # تلوين التوقيعين (أخضر للأصلي / أحمر للمشتبه)
        colored_overlay = np.ones((h, w, 3), dtype=np.uint8) * 255
        colored_overlay[mask_orig > 0] = [0, 180, 0]  # أخضر
        colored_overlay[mask_susp_resized > 0] = [255, 0, 0]  # أحمر

        # الأجزاء المفقودة والإضافية
        missing_parts = cv2.bitwise_and(
            mask_orig, cv2.bitwise_not(mask_susp_resized)
        )
        added_parts = cv2.bitwise_and(
            mask_susp_resized, cv2.bitwise_not(mask_orig)
        )

        annotated_result = colored_overlay.copy()

        # تعليم الأجزاء المفقودة (أزرق)
        cnts_missing, _ = cv2.findContours(
            missing_parts, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in cnts_missing:
            if cv2.contourArea(c) > min_area:
                x, y, bw, bh = cv2.boundingRect(c)
                cv2.rectangle(
                    annotated_result, (x, y), (x + bw, y + bh), (0, 0, 255), 2
                )
                cv2.putText(
                    annotated_result,
                    "Missing",
                    (x, max(y - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 0, 255),
                    1,
                )

        # تعليم الأجزاء الإضافية/الانحراف (برتقالي)
        cnts_added, _ = cv2.findContours(
            added_parts, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in cnts_added:
            if cv2.contourArea(c) > min_area:
                x, y, bw, bh = cv2.boundingRect(c)
                cv2.rectangle(
                    annotated_result, (x, y), (x + bw, y + bh), (255, 140, 0), 2
                )
                cv2.putText(
                    annotated_result,
                    "Deviation",
                    (x, max(y - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 140, 0),
                    1,
                )

        # --- 3. عرض المؤشرات والنتائج ---
        st.subheader("📊 2. نتائج التحليل والمؤشرات الهيكلية")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("نسبة التطابق (SSIM)", f"{similarity_pct}%")
        m2.metric("نسبة الاختلاف (Diff)", f"{difference_pct}%")
        m3.metric("نسبة كثافة الحبر (Stroke Ratio)", f"{density_ratio}%")

        status = (
            "توقيع مطابق / موثوق"
            if similarity_pct >= 75
            else "توقيع مشتبه به / مزور"
        )
        m4.metric(
            "التقييم التلقائي",
            status,
            delta="PASS" if similarity_pct >= 75 else "ALERT",
        )

        st.markdown("---")

        # --- 4. العرض البصري المتقدم ---
        st.subheader("🖼️ 3. التراكب البصري وتحليل خرائط الاختلاف")

        st.info("""
        **دليل تحليل الواجهة الجنائية:**
        * 🟢 **الخط الأخضر:** مسار التوقيع الأصلي (Reference).
        * 🔴 **الخط الأحمر:** مسار التوقيع المشتبه به (Questioned).
        * 🟦 **مستطيل أزرق (Missing):** مقطع أصلي غير موجود في التوقيع الثاني.
        * 🟧 **مستطيل برتقالي (Deviation):** انحراف في اتجاه الخط أو إضافة غير موجودة في الأصلي.
        """)

        if show_heatmap:
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.image(
                    annotated_result,
                    caption="تراكب التوقيعين وتحديد مواضع الاختلاف",
                    use_container_width=True,
                )
            with col_res2:
                # توليد خريطة الحرارة للإحداثيات
                diff_heatmap = cv2.applyColorMap(
                    ((1 - diff) * 255).astype(np.uint8), cv2.COLORMAP_JET
                )
                st.image(
                    diff_heatmap,
                    caption="خريطة الكثافة الحرارية للاختلافات (Heatmap)",
                    use_container_width=True,
                )
        else:
            st.image(
                annotated_result,
                caption="تراكب التوقيعين وتحديد مواضع الاختلاف",
                use_container_width=True,
            )

        # --- 5. خيارات التصدير ---
        st.markdown("---")
        st.subheader("💾 4. تصدير النتائج والتقارير")

        res_bgr = cv2.cvtColor(annotated_result, cv2.COLOR_RGB2BGR)
        _, encoded_img = cv2.imencode(".png", res_bgr)

        st.download_button(
            label="📥 تحميل صورة التحليل عالية الدقة (PNG)",
            data=encoded_img.tobytes(),
            file_name="Forensic_Signature_Analysis.png",
            mime="image/png",
        )
