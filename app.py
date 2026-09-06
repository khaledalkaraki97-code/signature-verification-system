from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="Forensic Signature Analysis Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 المنظومة الجنائية المتقدمة للتحقق من التوقيعات")
st.caption(
    "Automated Multi-Sample Forensic Signature Verification & Structural Dynamics Engine"
)
st.markdown("---")

# --- الشريط الجانبي ---
st.sidebar.header("⚙️ الإعدادات والخيارات الجنائية")
min_area = st.sidebar.slider(
    "تصفية التشويش (Min Contour Area)", 10, 200, 40, 10
)
show_heatmap = st.sidebar.checkbox("عرض الخريطة الحرارية (Heatmap)", value=True)

# وضع المقارنة المتعددة
st.sidebar.markdown("---")
multi_mode = st.sidebar.checkbox(
    "🔄 تفعيل خدمة التحليل المتعدد (4 مقابل 4)", value=False
)


# --- دوال المساعدة الجنائية ---
def process_mask(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return gray, mask


def extract_forensic_features(mask):
    """استخراج الخصائص الهندسية: مركز الثقل، نسبة الأبعاد، وسمك الخط"""
    # 1. مركز الثقل (Center of Mass)
    M = cv2.moments(mask)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

    # 2. نسبة الأبعاد (Aspect Ratio)
    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if cnts:
        all_cnt = np.vstack(cnts)
        x, y, w, h = cv2.boundingRect(all_cnt)
        aspect_ratio = round(float(w) / max(h, 1), 2)
    else:
        aspect_ratio = 1.0

    # 3. متوسط سُمك الخط (Stroke Thickness via Distance Transform)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    non_zero_dist = dist_transform[dist_transform > 0]
    avg_thickness = (
        round(float(np.mean(non_zero_dist) * 2), 2)
        if len(non_zero_dist) > 0
        else 0.0
    )

    return (cx, cy), aspect_ratio, avg_thickness


def analyze_pair(orig_np, susp_np, min_area_val):
    gray_orig, mask_orig = process_mask(orig_np)
    gray_susp, mask_susp = process_mask(susp_np)

    h, w = gray_orig.shape
    gray_susp_res = cv2.resize(gray_susp, (w, h))
    mask_susp_res = cv2.resize(mask_susp, (w, h))

    # حساب SSIM
    score, diff = ssim(gray_orig, gray_susp_res, full=True)
    ssim_pct = round(score * 100, 2)
    diff_pct = round(100.0 - ssim_pct, 2)

    # الخصائص الجنائية
    (cx1, cy1), ar1, thick1 = extract_forensic_features(mask_orig)
    (cx2, cy2), ar2, thick2 = extract_forensic_features(mask_susp_res)

    centroid_shift = round(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2), 2)
    ar_dev = round(abs(ar1 - ar2), 2)
    thick_diff = round(thick2 - thick1, 2)

    # التراكب الملون
    overlay = np.ones((h, w, 3), dtype=np.uint8) * 255
    overlay[mask_orig > 0] = [0, 180, 0]  # أخضر للأصلي
    overlay[mask_susp_res > 0] = [255, 0, 0]  # أحمر للمشتبه

    # تعليم الفروقات
    missing = cv2.bitwise_and(mask_orig, cv2.bitwise_not(mask_susp_res))
    added = cv2.bitwise_and(mask_susp_res, cv2.bitwise_not(mask_orig))

    annotated = overlay.copy()
    cnts_m, _ = cv2.findContours(
        missing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in cnts_m:
        if cv2.contourArea(c) > min_area_val:
            x, y, bw, bh = cv2.boundingRect(c)
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

    cnts_a, _ = cv2.findContours(
        added, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in cnts_a:
        if cv2.contourArea(c) > min_area_val:
            x, y, bw, bh = cv2.boundingRect(c)
            cv2.rectangle(
                annotated, (x, y), (x + bw, y + bh), (255, 140, 0), 2
            )

    diff_heatmap = cv2.applyColorMap(
        ((1 - diff) * 255).astype(np.uint8), cv2.COLORMAP_JET
    )

    return {
        "ssim_pct": ssim_pct,
        "diff_pct": diff_pct,
        "ar_orig": ar1,
        "ar_susp": ar2,
        "ar_dev": ar_dev,
        "centroid_shift": centroid_shift,
        "thick_orig": thick1,
        "thick_susp": thick2,
        "thick_diff": thick_diff,
        "annotated": annotated,
        "heatmap": diff_heatmap,
    }


# =========================================================
# 1. الوضع العادي (مقارنة توقيع واحد بآخر)
# =========================================================
if not multi_mode:
    st.subheader("📁 1. رفع عينات التوقيع الفردية (Single Comparison)")
    col_up1, col_up2 = st.columns(2)

    with col_up1:
        orig_file = st.file_uploader(
            "رفع التوقيع الأصلي المرجعي (Reference)",
            type=["jpg", "jpeg", "png"],
            key="orig_single",
        )
    with col_up2:
        susp_file = st.file_uploader(
            "رفع التوقيع المشتبه به (Questioned)",
            type=["jpg", "jpeg", "png"],
            key="susp_single",
        )

    if orig_file and susp_file:
        img_orig = Image.open(orig_file).convert("RGB")
        img_susp = Image.open(susp_file).convert("RGB")

        with st.expander("👁️ المعاينة القبلية للعينات", expanded=True):
            c1, c2 = st.columns(2)
            c1.image(
                img_orig,
                caption="التوقيع الأصلي",
                use_container_width=True,
            )
            c2.image(
                img_susp,
                caption="التوقيع المشتبه به",
                use_container_width=True,
            )

        if st.button("🚀 GENERATE / إجراء التحليل الجنائي الشامل"):
            res = analyze_pair(
                np.array(img_orig), np.array(img_susp), min_area
            )

            # القرار التلقائي
            if res["ssim_pct"] >= 75 and res["thick_diff"] < 1.5:
                decision = "✅ توقيع مطاق وموثوق (Authentic Signature)"
            elif res["ssim_pct"] < 60 or res["thick_diff"] >= 2.5:
                decision = (
                    "⚠️ توقيع مزور / مشتبه به بشدة (Highly Suspicious/Forged)"
                )
            else:
                decision = "🔍 يتطلب مراجعة خبير جنائي (Inconclusive / Manual Review)"

            # المؤشرات الرئيسية
            st.subheader("📊 2. المؤشرات الجنائية الأساسية والهندسية")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("نسبة التطابق SSIM", f"{res['ssim_pct']}%")
            m2.metric("تزحزح مركز الثقل (Centroid)", f"{res['centroid_shift']} px")
            m3.metric("انحراف الأبعاد (Aspect Ratio)", f"{res['ar_dev']}")
            m4.metric("فرق سُمك الخط (Thickness)", f"{res['thick_diff']} px")

            # جدول التقرير التفصيلي
            analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report_df = pd.DataFrame(
                [
                    {
                        "تاريخ/وقت التحليل": analysis_time,
                        "نسبة التطابق SSIM": f"{res['ssim_pct']}%",
                        "نسبة الاختلاف": f"{res['diff_pct']}%",
                        "أبعاد الأصلي (W/H)": res["ar_orig"],
                        "أبعاد المشتبه به (W/H)": res["ar_susp"],
                        "انحراف الأبعاد": res["ar_dev"],
                        "تزحزح مركز الثقل (px)": res["centroid_shift"],
                        "سُمك الخط الأصلي (px)": res["thick_orig"],
                        "سُمك الخط المشتبه به (px)": res["thick_susp"],
                        "التوصية الجنائية التلقائية": decision,
                    }
                ]
            )

            st.markdown("#### 📋 تقرير التحليل الجنائي المولد تلقائياً")
            st.dataframe(report_df, use_container_width=True)

            # العرض البصري
            st.subheader("🖼️ 3. التراكب البصري وخرائط الفروقات")
            st.info("""
            * 🟢 **أخضر:** الأصلي | 🔴 **أحمر:** المشتبه به.
            * 🟦 **مستطيل أزرق:** مقطع مفقود من التوقيع الأصلي.
            * 🟧 **مستطيل برتقالي:** إضافة جديدة أو انحراف في انحناء وسمك الخط.
            """)

            if show_heatmap:
                col1, col2 = st.columns(2)
                col1.image(
                    res["annotated"],
                    caption="التراكب البصري وإعادة التوجيه",
                    use_container_width=True,
                )
                col2.image(
                    res["heatmap"],
                    caption="خريطة الكثافة الحرارية للاختلافات (Heatmap)",
                    use_container_width=True,
                )
            else:
                st.image(
                    res["annotated"],
                    caption="التراكب البصري وإعادة التوجيه",
                    use_container_width=True,
                )

# =========================================================
# 2. وضع المقارنة المتعددة (4 تواقيع أصلية مقابل 4 مشتبه بها)
# =========================================================
else:
    st.subheader("🔄 2. خدمة التحليل الجنائي المتعدد (4 vs 4 Multi-Analysis)")
    st.write(
        "قم برفع حتى 4 عينات أصلية لحساب متوسط التباين الطبيعي، ومقارنتها بـ 4 عينات مشتبه بها."
    )

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        orig_files = st.file_uploader(
            "رفع التواقيع الصحيحة/الأصلية (حتى 4 صور)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="multi_orig",
        )
    with col_m2:
        susp_files = st.file_uploader(
            "رفع التواقيع المشتبه بها (حتى 4 صور)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="multi_susp",
        )

    if orig_files and susp_files:
        if st.button("⚡ بدء تحليل وتدقيق العينات المتعددة"):
            with st.spinner("جاري حساب متوسط التباين الجنائي والمقارنة..."):
                orig_imgs = [
                    np.array(Image.open(f).convert("RGB")) for f in orig_files
                ]
                susp_imgs = [
                    np.array(Image.open(f).convert("RGB")) for f in susp_files
                ]

                # حساب متوسطات التواقيع الأصلية
                orig_ar_list = []
                orig_thick_list = []
                for o_img in orig_imgs:
                    _, mask_o = process_mask(o_img)
                    _, ar_o, thick_o = extract_forensic_features(mask_o)
                    orig_ar_list.append(ar_o)
                    orig_thick_list.append(thick_o)

                avg_orig_ar = round(float(np.mean(orig_ar_list)), 2)
                avg_orig_thick = round(float(np.mean(orig_thick_list)), 2)

                st.success(
                    f"✅ تم حساب المعايير الطبيعية للتواقيع الأصلية — متوسط الأبعاد: {avg_orig_ar} | متوسط سُمك الخط: {avg_orig_thick} px"
                )

                # مقارنة كل توقيع مشتبه به مع التوقيع الأصلي الأول كمرجع للـ SSIM
                results_list = []
                analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                ref_orig_np = orig_imgs[0]

                for idx, s_img in enumerate(susp_imgs):
                    res = analyze_pair(ref_orig_np, s_img, min_area)

                    # تقييم مقابل المتوسط العام للأصلي
                    ar_dev_avg = round(abs(res["ar_susp"] - avg_orig_ar), 2)
                    thick_dev_avg = round(
                        abs(res["thick_susp"] - avg_orig_thick), 2
                    )

                    if res["ssim_pct"] >= 75 and thick_dev_avg < 1.5:
                        dec = "✅ مطابق للنمط الطبيعي"
                    else:
                        dec = "🚨 مشتبه به / انحراف عن النمط"

                    results_list.append(
                        {
                            "تاريخ التحليل": analysis_time,
                            "عينة المشتبه به": f"Sample #{idx+1}",
                            "نسبة التطابق SSIM": f"{res['ssim_pct']}%",
                            "سُمك الخط (px)": res["thick_susp"],
                            "انحراف السُمك عن المتوسط": thick_dev_avg,
                            "انحراف الأبعاد عن المتوسط": ar_dev_avg,
                            "تزحزح مركز الثقل (px)": res["centroid_shift"],
                            "التوصية الجنائية": dec,
                        }
                    )

                st.markdown("### 📊 جدول نتائج التحليل الشامل للعينات")
                df_multi = pd.DataFrame(results_list)
                st.dataframe(df_multi, use_container_width=True)

                st.markdown("### 🖼️ خرائط التراكب الفردية للعينات")
                cols_vis = st.columns(len(susp_imgs))
                for i, s_img in enumerate(susp_imgs):
                    res = analyze_pair(ref_orig_np, s_img, min_area)
                    with cols_vis[i]:
                        st.image(
                            res["annotated"],
                            caption=f"عينة مشتبه به #{i+1}",
                            use_container_width=True,
                        )
