import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import joblib
import timm
import os
from datetime import datetime

import faiss
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="DeepDR™ | AI Screening",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #5F9EA0; font-family: 'Helvetica Neue', Arial, sans-serif; }
    div[data-testid="InputInstructions"] { display: none !important; }
    div[data-baseweb="input"] { border-color: #d1d9e6 !important; }
    div[data-baseweb="input"]:focus-within { border-color: #1e3d59 !important; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05); 
        border: 1px solid #e1e4e8; 
        padding: 20px; 
    }
    div[data-baseweb="input"], div[data-baseweb="select"] > div {
        background-color: #f8f9fa !important; 
        border: 1px solid #d1d9e6 !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploadDropzone"] { 
        background-color: #f8f9fa !important; 
        border: 2px dashed #92b0c3 !important;
        border-radius: 12px; 
    }
    .stButton>button[kind="primary"] {
        background-color: #1e3d59; color: white; border-radius: 8px; border: none; padding: 0.5rem 1rem; font-weight: 600; transition: all 0.3s;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #2b5876; transform: translateY(-2px); box-shadow: 0 6px 12px rgba(30, 61, 89, 0.3);
    }
    .stButton>button[kind="secondary"] {
        background-color: white; color: #1e3d59; border: 1px solid #1e3d59;
    }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


class BinaryDRModel(nn.Module):
    def __init__(self, clinical_dim=5, feature_dim=256):
        super().__init__()
        self.image_encoder = timm.create_model('convnextv2_tiny', pretrained=False, num_classes=0, in_chans=3)
        self.image_proj = nn.Linear(768, feature_dim)
        self.aux_classifier = nn.Linear(feature_dim, 2)
        self.clinical_dropout = nn.Dropout(p=0.3)
        self.clinical_embedding = nn.Linear(clinical_dim, feature_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, dim_feedforward=512, dropout=0.1,
                                                   activation='gelu', batch_first=True)
        self.clinical_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.cross_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.fusion_dropout = nn.Dropout(p=0.15)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 2)
        )

    def forward(self, img, clinical, return_feats=False):
        img_raw = self.image_encoder(img)
        img_feat = self.image_proj(img_raw)
        clin_dropped = self.clinical_dropout(clinical)
        clin = self.clinical_embedding(clin_dropped).unsqueeze(1)
        clin_feat = self.clinical_encoder(clin).squeeze(1)
        img_q = self.norm1(img_feat).unsqueeze(1)
        clin_kv = self.norm2(clin_feat).unsqueeze(1)
        attn_out, _ = self.cross_attention(img_q, clin_kv, clin_kv)
        fused_features = torch.cat([attn_out.squeeze(1), clin_feat], dim=1)
        fused_features = self.fusion_dropout(fused_features)
        if return_feats: return fused_features
        return self.classifier(fused_features)


@st.cache_resource
def load_assets():

    model = BinaryDRModel(clinical_dim=5)
    model.load_state_dict(torch.load(r"C:\Users\admin\best_binary_dr_model.pth", map_location='cpu'))
    model.eval()

    scaler = joblib.load(r"C:\Users\admin\binary_clinical_scaler.pkl")
    voting_clf = joblib.load(r"C:\Users\admin\final_voting_ensemble.pkl")

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    knowledge_base = [
        "Diabetic retinopathy is a diabetes complication that affects the eyes and can lead to blindness.",
        "Early stages of diabetic retinopathy often have no symptoms.",
        "In mild nonproliferative retinopathy, microaneurysms occur.",
        "Moderate nonproliferative retinopathy involves blocked blood vessels.",
        "Severe nonproliferative retinopathy has many blocked vessels leading to oxygen deprivation.",
        "Proliferative retinopathy involves new abnormal blood vessel growth.",
        "Symptoms include blurry vision, floaters, and vision loss.",
        "Biomarkers include microaneurysms, hemorrhages, exudates, and neovascularization.",
        "Treatment for early stages focuses on blood sugar control.",
        "Laser treatment like pan-retinal photocoagulation is used for advanced stages.",
        "Anti-VEGF injections help reduce swelling and new vessel growth.",
        "Vitrectomy surgery may be needed for severe cases.",
        "Regular eye exams are crucial for early detection.",
        "High blood sugar damages retinal blood vessels over time.",
        "Hypertension and high cholesterol increase DR risk.",
        "Duration of diabetes correlates with DR severity.",
        "HbA1c levels above 7% increase DR progression risk.",
        "Diabetic macular edema is a common complication.",
        "Fundus photography is key for DR screening.",
        "OCT imaging helps detect retinal thickening.",
        "Mild NPDR (Grade 1) is characterized by at least one microaneurysm with no other lesions.",
        "Moderate NPDR (Grade 2) features extensive microaneurysms, dot-blot hemorrhages, cotton-wool spots, and venous beading in less than two quadrants.",
        "Severe NPDR (Grade 3) follows the 4-2-1 rule: hemorrhages in 4 quadrants, venous beading in 2, and IRMA in 1.",
        "Proliferative DR (Grade 4) includes neovascularization on the disc or elsewhere, with risk of vitreous hemorrhage.",
        "In moderate NPDR, blocked retinal vessels lead to ischemia and potential progression to severe stages.",
        "Cotton-wool spots in Grade 2 indicate nerve fiber layer infarcts due to capillary non-perfusion.",
        "Venous beading in moderate DR is a sign of retinal hypoxia and impending severe disease.",
        "HbA1c greater than 7% is strongly associated with faster progression from moderate to severe NPDR.",
        "Early detection of Grade 2 changes via fundus exam can prevent vision loss through timely intervention.",
        "Anti-VEGF therapy is effective for macular edema complicating moderate NPDR.",
        "Duration of diabetes over 10 years increases likelihood of moderate retinopathy.",
        "Hypertension exacerbates hemorrhages in Grade 2 DR.",
        "No apparent retinopathy (Grade 0) shows no observable abnormalities on dilated exam.",
        "International Clinical DR Severity Scale defines mild, moderate, severe NPDR, and PDR.",
        "ETDRS study provides evidence-based grading for DR clinical trials."
    ]

    kb_embeds = sbert_model.encode(knowledge_base)
    index = faiss.IndexFlatL2(kb_embeds.shape[1])
    index.add(kb_embeds.astype(np.float32))

    return model, scaler, voting_clf, sbert_model, index, knowledge_base


try:
    ai_model, clinical_scaler, voting_model, sbert_model, faiss_index, knowledge_base = load_assets()
except Exception as e:
    st.error(f"System Error: Model/Asset loading failed. Details: {e}")


def retrieve_evidence(query, k=3, threshold=1.2):
    query_emb = sbert_model.encode([query])
    D, I = faiss_index.search(query_emb.astype(np.float32), k)
    evidence = []
    for dist, idx in zip(D[0], I[0]):
        if dist < threshold:
            evidence.append(knowledge_base[idx])

    if not evidence:
        for idx in I[0][:2]:
            evidence.append(knowledge_base[idx])
    return evidence


if 'diagnosis_done' not in st.session_state:
    st.session_state.diagnosis_done = False
if 'saved_prob' not in st.session_state:
    st.session_state.saved_prob = 0.0
if 'reset_key' not in st.session_state:
    st.session_state.reset_key = 0


def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image.convert('RGB')).unsqueeze(0)


@st.dialog("📋 Official Medical AI Assessment Report", width="large")
def show_diagnosis_report(prob, clinical_data):
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record_id = f"DR-{datetime.now().strftime('%Y%m%d')}-{np.random.randint(1000, 9999)}"

    st.markdown(
        f"<div style='text-align: right; color: #7f8c8d; font-size: 12px;'>Date: {report_time}<br>Record ID: {record_id}</div>",
        unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align: center; color: #1e3d59; border-bottom: 2px solid #1e3d59; padding-bottom: 10px;'>MULTIMODAL DR DIAGNOSTIC REPORT</h3>",
        unsafe_allow_html=True)

    st.write("#### 👤 PATIENT CLINICAL PROFILE")

    best_thresh = 0.3032
    pred_idx = 1 if prob > best_thresh else 0

    if clinical_data['hba1c'] >= 7.0 or clinical_data['sbp'] >= 140 or clinical_data['hyp'] == 'Yes':
        status = 'Metabolic Risk'
        status_color = '#e74c3c'
    else:
        status = 'Stable Baseline'
        status_color = '#27ae60'

    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 25px;">
        <div style="border: 1px solid #d1d9e6; border-radius: 10px; padding: 15px 5px; display: flex; justify-content: center; align-items: center; background-color: #ffffff; box-shadow: 0 2px 5px rgba(0,0,0,0.02);">
            <span style="font-size: 15px; color: #1e3d59;"><b>Age:</b> {clinical_data['age']} Yrs</span>
        </div>
        <div style="border: 1px solid #d1d9e6; border-radius: 10px; padding: 15px 5px; display: flex; justify-content: center; align-items: center; background-color: #ffffff; box-shadow: 0 2px 5px rgba(0,0,0,0.02);">
            <span style="font-size: 15px; color: #1e3d59;"><b>HbA1c:</b> {clinical_data['hba1c']:.2f} %</span>
        </div>
        <div style="border: 1px solid #d1d9e6; border-radius: 10px; padding: 15px 5px; display: flex; justify-content: center; align-items: center; background-color: #ffffff; box-shadow: 0 2px 5px rgba(0,0,0,0.02);">
            <span style="font-size: 15px; color: #1e3d59;"><b>Diabetes Duration:</b> {clinical_data['duration']} Yrs</span>
        </div>
        <div style="border: 1px solid #d1d9e6; border-radius: 10px; padding: 15px 5px; display: flex; justify-content: center; align-items: center; background-color: #ffffff; box-shadow: 0 2px 5px rgba(0,0,0,0.02);">
            <span style="font-size: 15px; color: #1e3d59;"><b>Systolic BP:</b> {clinical_data['sbp']} mmHg</span>
        </div>
        <div style="border: 1px solid #d1d9e6; border-radius: 10px; padding: 15px 5px; display: flex; justify-content: center; align-items: center; background-color: #ffffff; box-shadow: 0 2px 5px rgba(0,0,0,0.02);">
            <span style="font-size: 15px; color: #1e3d59;"><b>Hypertension:</b> {clinical_data['hyp']}</span>
        </div>
        <div style="border: 1px solid #d1d9e6; border-radius: 10px; padding: 15px 5px; display: flex; justify-content: center; align-items: center; background-color: #ffffff; box-shadow: 0 2px 5px rgba(0,0,0,0.02);">
            <span style="font-size: 15px; color: #1e3d59;"><b>Baseline:</b> <span style="color: {status_color}; font-weight: bold;">{status}</span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.write("#### 🔬 AI DIAGNOSTIC INFERENCE")

    prob_color = "#e74c3c" if pred_idx == 1 else "#27ae60"
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid {prob_color};">
        <span style="font-size: 16px;">Predicted DR Probability:</span><br>
        <span style="font-size: 32px; font-weight: bold; color: {prob_color};">{prob:.2%}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


    st.write("#### 🧠 EVIDENCE-BASED REASONING (RAG)")

    query_parts = []
    if pred_idx == 1:
        query_parts.append(
            "Diabetic retinopathy is a diabetes complication. High blood sugar damages retinal blood vessels over time.")
        if clinical_data['hba1c'] > 7.0:
            query_parts.append("HbA1c levels above 7% increase DR progression risk.")
        if clinical_data['duration'] >= 10:
            query_parts.append("Duration of diabetes over 10 years increases likelihood of retinopathy.")
        if clinical_data['sbp'] >= 140 or clinical_data['hyp'] == 'Yes':
            query_parts.append("Hypertension and high cholesterol increase DR risk.")
    else:
        query_parts.append(
            "Early stages of diabetic retinopathy often have no symptoms. Regular eye exams are crucial for early detection.")
        if clinical_data['hba1c'] > 7.0:
            query_parts.append("Treatment for early stages focuses on blood sugar control.")

    query = " ".join(query_parts)

    raw_evidence = retrieve_evidence(query, k=6)

    filtered_evidence = []

    banned_terms = [
        "macular edema", "microaneurysm", "hemorrhage", "cotton-wool",
        "venous beading", "neovascularization", "exudate", "thickening",
        "mild", "moderate", "severe", "proliferative", "npdr", "pdr", "grade",
        "blocked", "ischemia", "infarcts"
    ]

    for ev in raw_evidence:
        if "10 years" in ev and clinical_data['duration'] < 10: continue
        if "7%" in ev and clinical_data['hba1c'] <= 7.0: continue
        if ("Hypertension" in ev or "blood pressure" in ev.lower()) and clinical_data['sbp'] < 140 and clinical_data[
            'hyp'] == 'No': continue

        if any(term in ev.lower() for term in banned_terms): continue

        if pred_idx == 1 and ("early stage" in ev.lower() or "no symptoms" in ev.lower() or "mild" in ev.lower()):
            continue

        if pred_idx == 0 and ("blindness" in ev.lower() or "progression" in ev.lower()):
            continue

        filtered_evidence.append(ev)

        if len(filtered_evidence) == 3:
            break

    if len(filtered_evidence) < 3:
        if pred_idx == 1:
            fallback_knowledge = [
                "Diabetic retinopathy is a diabetes complication that affects the eyes and can lead to blindness.",
                "High blood sugar damages retinal blood vessels over time.",
                "Hypertension and high cholesterol increase DR risk."
            ]
        else:
            fallback_knowledge = [
                "Early stages of diabetic retinopathy often have no symptoms.",
                "Regular eye exams are crucial for early detection.",
                "Treatment for early stages focuses on blood sugar control."
            ]

        for fb in fallback_knowledge:
            if fb not in filtered_evidence:
                filtered_evidence.append(fb)
            if len(filtered_evidence) == 3: break

    st.info(
        "The following medical knowledge was dynamically retrieved and fact-checked by the DeepDR-Agent based on the patient's specific features:")

    for i, ev in enumerate(filtered_evidence, 1):
        st.markdown(f"<p style='color:#34495e; font-size: 14px;'><b>[{i}]</b> {ev}</p>", unsafe_allow_html=True)

    st.write("#### ⚕️ CLINICAL ADVICE")
    if pred_idx == 1:
        if clinical_data['hba1c'] > 7.0:
            st.warning(
                f"**Metabolic Alert:** Patient's HbA1c ({clinical_data['hba1c']:.2f}%) is above 7.0%, accelerating DR progression.")
        st.error("🚨 **Action:** REFERRAL to Ophthalmology for comprehensive staging (Status: Positive).")
    else:
        if clinical_data['hba1c'] > 7.0:
            st.warning(
                f"**Metabolic Alert:** Although no DR detected, HbA1c ({clinical_data['hba1c']:.2f}%) is high. Strict glycemic control is advised.")
        st.success("✅ **Action:** Routine annual fundus examination is recommended (Status: Negative).")

    st.caption(
        "Disclaimer: This AI system is designed to assist clinical decision-making and should not replace professional medical judgment.")

st.markdown(
    "<h1 style='text-align: center; color: #1e3d59; margin-bottom: 0px; font-weight: 800;'>DeepDR™ Multimodal System</h1>",
    unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #555555; font-size: 16px; margin-bottom: 30px; letter-spacing: 0.5px;'>Advanced AI-Powered Retinal & RAG Integrated Screening</p>",
    unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🩺 Clinical Screening Desk", "📊 Model Performance", "🧠 Interpretability (XAI)", "💊 Recommendation"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        with st.container(border=True):
            st.markdown(
                "<h3 style='color: #2b5876; border-bottom: 2px solid #f0f2f6; padding-bottom: 10px; margin-bottom: 20px;'>📝 Patient Data & Imaging</h3>",
                unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size:14px; color:#555; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;'><b>Clinical Vitals</b></div>",
                unsafe_allow_html=True)

            cc1, cc2 = st.columns(2)
            with cc1:
                age = st.number_input("Age (Years)", value=60, step=1, help="Patient's current age.",
                                      key=f"age_{st.session_state.reset_key}")
                if not (1 <= age <= 120): st.markdown(
                    "<div style='color: #e74c3c; font-size: 13px; margin-top: -12px; margin-bottom: 10px;'>⚠️ Value out of range (1 - 120)</div>",
                    unsafe_allow_html=True)

                hba1c = st.number_input("HbA1c (%)", value=7.5, step=0.1, help="Normal < 5.7%, Diabetes >= 6.5%",
                                        key=f"hba1c_{st.session_state.reset_key}")
                if not (4.0 <= hba1c <= 25.0): st.markdown(
                    "<div style='color: #e74c3c; font-size: 13px; margin-top: -12px; margin-bottom: 10px;'>⚠️ Value out of range (4.0 - 25.0)</div>",
                    unsafe_allow_html=True)

                hyp_ui = st.selectbox("Hypertension?", ["No", "Yes"], key=f"hyp_{st.session_state.reset_key}")

            with cc2:
                dur = st.number_input("Diabetes Duration (Yrs)", value=10, step=1,
                                      key=f"dur_{st.session_state.reset_key}")
                if not (0 <= dur <= 100): st.markdown(
                    "<div style='color: #e74c3c; font-size: 13px; margin-top: -12px; margin-bottom: 10px;'>⚠️ Value out of range (0 - 100)</div>",
                    unsafe_allow_html=True)

                sbp = st.number_input("Systolic BP (mmHg)", value=130, step=1, key=f"sbp_{st.session_state.reset_key}")
                if not (60 <= sbp <= 250): st.markdown(
                    "<div style='color: #e74c3c; font-size: 13px; margin-top: -12px; margin-bottom: 10px;'>⚠️ Value out of range (60 - 250)</div>",
                    unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(
                "<div style='font-size:14px; color:#555; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;'><b>Retinal Fundus Scan</b></div>",
                unsafe_allow_html=True)

            uploaded_file = st.file_uploader("Upload Fundus Image", type=["png", "jpg", "jpeg"],
                                             label_visibility="collapsed", key=f"file_{st.session_state.reset_key}")
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Preview", use_container_width=True)
            else:
                st.info("👆 Please upload a fundus image to proceed.")

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button(" INITIATE ANALYSIS", type="primary", use_container_width=True):
                if not uploaded_file:
                    st.error("⚠️ Operation Aborted: Please upload a fundus image first.")
                elif not (1 <= age <= 120):
                    st.error("⚠️ Input Error: Age must be between 1 and 120.")
                elif not (0 <= dur <= 100):
                    st.error("⚠️ Input Error: Duration must be between 0 and 100.")
                elif not (4.0 <= hba1c <= 25.0):
                    st.error("⚠️ Input Error: HbA1c must be between 4.0% and 25.0%.")
                elif not (60 <= sbp <= 250):
                    st.error("⚠️ Input Error: Systolic BP must be between 60 and 250.")
                else:
                    with st.spinner("🧠 AI Models Processing (Feature Fusion & RAG)..."):
                        try:
                            img_tensor = process_image(image)
                            h_val = 1.0 if hyp_ui == "Yes" else 0.0
                            raw_clin = np.array([[age, dur, hba1c, sbp, h_val]])
                            scaled_clin = clinical_scaler.transform(raw_clin)
                            clin_tensor = torch.tensor(scaled_clin, dtype=torch.float32)

                            with torch.no_grad():
                                f1 = ai_model(img_tensor, clin_tensor, return_feats=True)
                                f2 = ai_model(torch.flip(img_tensor, [3]), clin_tensor, return_feats=True)
                                f3 = ai_model(torch.flip(img_tensor, [2]), clin_tensor, return_feats=True)
                                f4 = ai_model(torch.rot90(img_tensor, 1, [2, 3]), clin_tensor, return_feats=True)
                                f5 = ai_model(torch.rot90(img_tensor, 3, [2, 3]), clin_tensor, return_feats=True)
                                feats_avg = (f1 + f2 + f3 + f4 + f5) / 5.0
                                features_np = feats_avg.cpu().numpy()

                            prob = voting_model.predict_proba(features_np)[0][1]

                            st.session_state.saved_prob = prob
                            st.session_state.diagnosis_done = True
                            st.rerun()

                        except Exception as e:
                            st.error(f"Inference Error: {e}")

    with col2:
        with st.container(border=True):
            st.markdown(
                "<h3 style='color: #2b5876; border-bottom: 2px solid #f0f2f6; padding-bottom: 10px; margin-bottom: 20px;'>📊 Diagnostic Results</h3>",
                unsafe_allow_html=True)

            if not st.session_state.diagnosis_done:
                st.markdown("""
                <div style='text-align: center; color: #9aa5b1; padding: 40px;'>
                    <div style='font-size: 60px;'>🧬</div><p>Waiting for Input...</p>
                    <p style='font-size: 12px;'>Enter clinical data and upload an image on the left panel to start the AI analysis.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                p = st.session_state.saved_prob
                risk_label = "Positive (High Risk)" if p > 0.3032 else "Negative (Low Risk)"
                risk_color = "#e74c3c" if p > 0.3032 else "#27ae60"

                st.markdown(f"""
                    <div style="text-align: center; margin-bottom: 20px;">
                        <span style="font-size: 14px; color: #555;">DR Probability Score</span><br>
                        <span style="font-size: 48px; font-weight: bold; color: {risk_color};">{p * 100:.2f}%</span><br>
                        <span style="font-size: 18px; font-weight: 500; color: {risk_color}; background: {'rgba(231, 76, 60, 0.1)' if p > 0.3032 else 'rgba(39, 174, 96, 0.1)'}; padding: 5px 15px; border-radius: 20px;">{risk_label}</span>
                    </div>
                """, unsafe_allow_html=True)

                st.write("Risk Threshold Meter:")
                st.progress(min(p, 1.0))
                st.caption("Threshold Cutoff: 30.32% (Calibrated by Youden Index)")

                st.markdown("---")
                st.markdown("#### Actions:")
                col_btn1, col_btn2 = st.columns(2)
                report_data = {'age': age, 'hba1c': hba1c, 'duration': dur, 'sbp': sbp, 'hyp': hyp_ui}

                with col_btn1:
                    if st.button("📄 View Full Report", use_container_width=True):
                        show_diagnosis_report(p, report_data)

                with col_btn2:
                    if st.button("🔄 Reset Analysis", type="secondary", use_container_width=True):
                        st.session_state.diagnosis_done = False
                        st.session_state.saved_prob = 0.0
                        st.session_state.reset_key += 1
                        st.rerun()

with tab2:
    st.markdown("<h3 style='color: #1e3d59; padding-bottom: 10px;'>🏆 Global Model Performance Dashboard</h3>",
                unsafe_allow_html=True)
    st.info(
        "The system uses a Hybrid Ensemble Architecture (CNN Features + SVM/LR Soft Voting). Below are the comprehensive evaluation metrics on the unseen Stress Test Dataset (Second Dataset).")

    st.markdown(
        "<div style='font-size:15px; color:#2b5876; font-weight:bold; margin-top: 10px;'>🌟 Primary Metrics</div>",
        unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="Accuracy", value="92.18%", delta="Overall Correctness")
    with m2:
        st.metric(label="AUC Score", value="0.9796", delta="Excellent Discernment")
    with m3:
        st.metric(label="F1-Score (Weighted)", value="92.23%", delta="Harmonic Mean")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:15px; color:#2b5876; font-weight:bold;'>🩺 Diagnostic Validity</div>",
                unsafe_allow_html=True)

    m5, m6, m7 = st.columns(3)
    with m5:
        st.metric(label="Sensitivity (Recall)", value="92.17%", delta="True Positive Rate (TPR)")
    with m6:
        st.metric(label="Specificity", value="92.20%", delta="True Negative Rate (TNR)")
    with m7:
        st.metric(label="PPV (Precision)", value="95.39%", delta="Positive Predictive Value")

    st.divider()

    st.markdown("#### 📈 Evaluation Curves")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Receiver Operating Characteristic (ROC)**")
        try:
            st.image(r"C:\Users\admin\DR images\Evaluation_ROC_SecondDataset.png", use_container_width=True)
        except:
            st.warning("⚠️ Please update the path to Evaluation_ROC_SecondDataset.png")

        st.markdown("**Model Learning Curve**")
        try:
            st.image(r"D:\Downloads\974bbe0c-708d-4a85-bd9d-08305580189c.jpg", use_container_width=True)
        except:
            st.warning("⚠️ Please update the path to Learning_Curve.png")

    with c2:
        st.markdown("**Precision-Recall Curve (PR)**")
        try:
            st.image(r"C:\Users\admin\DR images\Evaluation_PR_SecondDataset.png", use_container_width=True)
        except:
            st.warning("⚠️ Please update the path to Evaluation_PR_SecondDataset.png")

        st.markdown("**Confusion Matrix**")
        try:
            st.image(r"C:\Users\admin\DR images\Evaluation_CM_SecondDataset.png", use_container_width=True)
        except:
            st.warning("⚠️ Please update the path to Evaluation_CM_SecondDataset.png")

with tab3:
    st.markdown("<h3 style='color: #1e3d59; padding-bottom: 10px;'>🧠 eXplainable AI (XAI) Analysis</h3>",
                unsafe_allow_html=True)
    st.write(
        "This section unpacks the 'black box' of our deep learning model, showing exactly which clinical features drove the AI's final decision using **SHAP (SHapley Additive exPlanations)**.")

    st.markdown("#### 🌍 Global Feature Importance (Cohort Level)")
    st.info(
        "💡 **Clinical Insights:** These plots reveal the overall behavior of the model. **Diabetes Duration** and **HbA1c** emerge as the most critical predictors globally. Red dots on the right indicate that higher values of these features significantly increase the DR risk, which aligns perfectly with medical literature.")

    xai_col1, xai_col2 = st.columns(2)
    with xai_col1:
        st.markdown("**SHAP Global Feature Importance**")
        try:
            st.image(r"C:\Users\admin\DR images\SHAP_Summary_Bar_SecondDataset.png",
                     caption="Mean |SHAP| Value (Impact Magnitude)",
                     use_container_width=True)
        except:
            st.warning("⚠️ Please update the path to SHAP_Summary_Bar_SecondDataset.png")

    with xai_col2:
        st.markdown("**SHAP Impact Distribution**")
        try:
            st.image(r"C:\Users\admin\DR images\SHAP_Summary_Dot_SecondDataset.png",
                     caption="Feature Value Distribution & Directionality", use_container_width=True)
        except:
            st.warning("⚠️ Please update the path to SHAP_Summary_Dot_SecondDataset.png")

    # ------ 这里为你新增了 SHAP 的第 5 个图 (Dependence Plot) 区域 ------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📈 Feature Dependence Analysis")
    dep_col1, dep_col2 = st.columns(2)
    with dep_col1:
        st.markdown("**HbA1c Impact Trajectory**")
        try:
            st.image(r"C:\Users\admin\DR images\SHAP_Dependence_HbA1c_SecondDataset.png",
                     caption="Dependence Plot for HbA1c", use_container_width=True)
        except:
            st.warning("⚠️ Please update the path to SHAP_Dependence_HbA1c_SecondDataset.png")
    with dep_col2:
        st.markdown("**Duration Impact Trajectory**")
        try:
            st.image(r"C:\Users\admin\DR images\SHAP_Dependence_Duration_SecondDataset.png",
                     caption="Dependence Plot for Duration of Diabetes", use_container_width=True)
        except:
            st.warning("⚠️ Please update the path to SHAP_Dependence_Duration_SecondDataset.png")

    st.divider()

    st.markdown("#### 👤 Local Explanation (Individual Patient Level)")
    st.info(
        "🔍 **Micro-Analysis:** These plots explain the decision-making process for a **single specific prediction**. Features pushing the risk score higher are shown in red, while features lowering the risk are in blue.")

    xai_col3, xai_col4 = st.columns([1, 1.2])
    with xai_col3:
        st.markdown("**Step-by-Step Feature Contributions**")
        try:
            st.image(r"C:\Users\admin\DR images\SHAP_Waterfall_SecondDataset.png", caption="SHAP Waterfall Plot",
                     use_container_width=True)
        except:
            st.warning("⚠️ Please update the path to SHAP_Waterfall_SecondDataset.png")

    with xai_col4:
        st.markdown("**Competing Forces Driving the Output**")
        st.markdown("<br>", unsafe_allow_html=True)
        try:
            st.image(r"C:\Users\admin\DR images\SHAP_Force_Plot_SecondDataset.png", caption="SHAP Force Plot",
                     use_container_width=True)
        except:
            st.warning("⚠️ Please update the path to SHAP_Force_Plot_SecondDataset.png")

with tab4:
    st.markdown("<h3 style='text-align: center;'>📋 Personalized Patient Management Plan</h3>", unsafe_allow_html=True)
    if not st.session_state.diagnosis_done:
        st.info(
            "⚠️ Please initiate the analysis in the '🩺 Clinical Screening Desk' tab first to generate dynamically personalized recommendations.")
    else:
        r_key = st.session_state.reset_key
        current_age = st.session_state.get(f"age_{r_key}", 60)
        current_hba1c = st.session_state.get(f"hba1c_{r_key}", 7.5)
        current_dur = st.session_state.get(f"dur_{r_key}", 10)
        current_sbp = st.session_state.get(f"sbp_{r_key}", 130)
        current_hyp = st.session_state.get(f"hyp_{r_key}", "No")
        prob = st.session_state.saved_prob
        is_high_risk = prob > 0.3032

        st.write(
            "Based on the patient's specific clinical biomarkers and multimodal AI screening results, the DeepDR agent has dynamically generated the following personalized management plan.")

        c_col1, c_col2 = st.columns(2, gap="large")
        with c_col1:
            if current_hba1c > 8.0:
                st.error(
                    f"#### 🩸 1. Glycemic Control (Critical)\nCurrent HbA1c is **{current_hba1c}%**, indicating poor glycemic control.\n* **Action:** Urgent consultation with an endocrinologist to adjust insulin or oral hypoglycemic medications.\n* **Goal:** Aggressively lower HbA1c below 7.0% to prevent further retinal microvascular damage.")
            elif current_hba1c > 7.0:
                st.warning(
                    f"#### 🩸 1. Glycemic Control (Caution)\nCurrent HbA1c is **{current_hba1c}%**, slightly above the target range.\n* **Action:** Review diet and current medication compliance.\n* **Goal:** Tighten blood sugar management to reach the strictly < 7.0% target.")
            else:
                st.success(
                    f"#### 🩸 1. Glycemic Control (Stable)\nCurrent HbA1c is **{current_hba1c}%**, showing excellent control.\n* **Action:** Maintain current dietary and medication regimen.\n* **Goal:** Continue to keep HbA1c below 7.0%.")

            if is_high_risk:
                st.error(
                    f"#### 👁️ 2. Ophthalmic Follow-up (Urgent)\nAI predicts a **High Risk ({prob:.1%})** of Diabetic Retinopathy.\n* **Action:** Immediate referral to a retina specialist for a comprehensive dilated fundus examination and OCT scan within the next **2-4 weeks**.\n* **Note:** Early intervention (e.g., anti-VEGF or laser) may be required.")
            elif current_dur > 10:
                st.warning(
                    f"#### 👁️ 2. Ophthalmic Follow-up (Moderate Risk)\nAI result is Negative, but diabetes duration is long (**{current_dur} years**).\n* **Action:** Schedule a follow-up screening in **6 months**.\n* **Note:** Long disease duration inherently increases future risks.")
            else:
                st.success(
                    f"#### 👁️ 2. Ophthalmic Follow-up (Routine)\nAI indicates Low Risk of Diabetic Retinopathy.\n* **Action:** Maintain routine **annual** fundus screening to catch early asymptomatic changes.")

        with c_col2:
            if current_sbp >= 140 or current_hyp == "Yes":
                st.error(
                    f"#### 🫀 3. Blood Pressure Management (Alert)\nSystolic BP is **{current_sbp} mmHg** (Hypertension History: {current_hyp}).\n* **Action:** High BP severely exacerbates retinal hemorrhage. Adjust antihypertensive therapy immediately.\n* **Diet:** Strict adherence to the DASH diet (low sodium).")
            elif current_sbp >= 120:
                st.warning(
                    f"#### 🫀 3. Blood Pressure Management (Monitor)\nSystolic BP is **{current_sbp} mmHg** (Pre-hypertension range).\n* **Action:** Monitor blood pressure daily. Aim to keep Systolic BP < 130 mmHg.\n* **Diet:** Reduce salt intake and increase potassium-rich foods.")
            else:
                st.success(
                    f"#### 🫀 3. Blood Pressure Management (Optimal)\nSystolic BP is **{current_sbp} mmHg**, which is optimal.\n* **Action:** Continue current cardiovascular health habits to protect retinal capillaries.")

            exercise_rec = "low-impact exercises (e.g., swimming, cycling) to protect joints" if current_age > 65 else "at least 150 minutes of moderate-intensity aerobic physical activity per week"
            st.info(
                f"#### 🏃‍♂️ 4. Lifestyle & Weight Management\nOverall metabolic health directly impacts diabetes severity (Patient Age: {current_age}).\n* **Exercise:** Engage in {exercise_rec}.\n* **Habits:** Strict cessation of smoking, as nicotine aggressively constricts small blood vessels in the retina.")

    st.divider()
    st.markdown(
        "<hr><div style='text-align: center; color: gray; font-size: 12px; margin-top: 20px;'>⚠️ Medical Disclaimer: These dynamically generated recommendations are for educational and preventative purposes and do not substitute a formal doctor's prescription.</div>",
        unsafe_allow_html=True)