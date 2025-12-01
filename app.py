# -*- coding: utf-8 -*-
import io
import os
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import streamlit as st

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay, log_loss,
    f1_score, classification_report, roc_curve, auc, roc_auc_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch
from sklearn.preprocessing import label_binarize

from scipy.stats import chi2

np.random.seed(42)

# ======================= KSU logo & basic page setup =======================
ksu_logo_path = "KSU_MasterLogo_Colour_RGB.png"  # Make sure the logo is in the same folder

st.set_page_config(
    page_title="مشروع التخرج - تصنيف تغريدات البنوك السعودية",
    page_icon=ksu_logo_path,
    layout="wide"
)

# ======================= KSU theme CSS =======================
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }

    .ksu-header {
        padding: 1.5rem 1rem 1.2rem 1rem;
        background-color: #0076B6;
        border-radius: 0 0 18px 18px;
        color: #ffffff;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }

    .ksu-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .ksu-subtitle {
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 0.2rem;
    }

    .ksu-project {
        font-size: 0.95rem;
        font-weight: 500;
        line-height: 1.6;
        margin-top: 0.4rem;
    }

    .ksu-student {
        font-size: 0.9rem;
        font-weight: 400;
        margin-top: 0.3rem;
        opacity: 0.9;
    }

    .stButton>button {
        background-color: #0076B6;
        color: white;
        border-radius: 999px;
        padding: 0.45rem 1.2rem;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #005c8b;
        color: #ffffff;
    }

    h2, h3 {
        color: #005c8b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================= KSU HEADER (ENGLISH ONLY) =======================
with st.container():
    st.markdown('<div class="ksu-header">', unsafe_allow_html=True)
    col_logo, col_text = st.columns([1, 3])

    with col_logo:
        if os.path.exists(ksu_logo_path):
            st.image(ksu_logo_path, use_container_width=True)
        else:
            st.write("**[KSU Logo not found: Place 'KSU_MasterLogo_Colour_RGB.png' in the same folder]**")

    with col_text:
        st.markdown(
            """
            <div class="ksu-title">King Saud University</div>
            <div class="ksu-subtitle">College of Science – Department of Statistics & Operations Research</div>
            <div class="ksu-subtitle">Graduation Project</div>
            <div class="ksu-project">
            Statistics Is All You Need: Comparing Large Language Models and Traditional Statistical Models<br>
            for Sentiment Classification of Saudi Banks' Tweets
            </div>
            <div class="ksu-student">
            Student: <b>Abdulrahman Al-Kurayshan</b>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

# App title below header
st.markdown(
    "###  تصنيف تغريدات البنوك السعودية باستخدام BERT و Logistic Regression (Shared Split)"
)

# ======================= BERT: Data & Model =======================
@st.cache_data(show_spinner=True)
def load_data(path):
    df = pd.read_excel(path)
    df = df.dropna(subset=['Tweet', 'Final annotation'])
    # Use the processed text from "Tokens without stop words" (list -> string)
    df['text'] = df['Tokens without stop words'].apply(lambda x: " ".join(eval(x)))
    label_map = {'NEG': 0, 'POS': 1, 'NEU': 2}
    id2label = {v: k for k, v in label_map.items()}
    df['label'] = df['Final annotation'].map(label_map)
    return df, label_map, id2label

@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
    mdl = AutoModelForSequenceClassification.from_pretrained(
        "asafaya/bert-base-arabic", num_labels=3
    )
    return tok, mdl

class TextDataset(torch.utils.data.Dataset):
    """Simple dataset wrapper for HuggingFace Trainer."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# ======================= Common helper functions =======================
def softmax_np(logits):
    """2D softmax for shape (N, C)."""
    z = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)

def softmax_1d(z):
    """1D softmax for a vector (C,)."""
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()

def brier_multiclass(y_true, probs, C):
    """Multiclass Brier score."""
    one_hot = np.eye(C)[y_true]
    return np.mean(np.sum((probs - one_hot) ** 2, axis=1))

def ece(probs, y_true, n_bins=15):
    """Expected Calibration Error (ECE)."""
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    acc = (preds == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_sum = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i > 0:
            m = (conf > lo) & (conf <= hi)
        else:
            m = (conf >= lo) & (conf <= hi)
        if not np.any(m):
            continue
        ece_sum += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return ece_sum

def reliability_diagram(probs, y_true, n_bins=15):
    """Plot reliability diagram for predicted probabilities."""
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    acc = (preds == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, accs, confs, counts = [], [], [], []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i > 0:
            m = (conf > lo) & (conf <= hi)
        else:
            m = (conf >= lo) & (conf <= hi)
        if not np.any(m):
            continue
        bin_centers.append((lo + hi) / 2.0)
        accs.append(acc[m].mean())
        confs.append(conf[m].mean())
        counts.append(m.sum())

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect", alpha=0.8)
    ax.plot(confs, accs, marker="o", linewidth=2, label="Model")
    ax.set_xlabel("Average confidence")
    ax.set_ylabel("Accuracy in bin")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    return fig

def bootstrap_macro_f1(y_true, y_pred, B=1000, seed=42):
    """Bootstrap CI for macro-F1."""
    rng = np.random.default_rng(seed)
    N = len(y_true)
    vals = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for _ in range(B):
        idx = rng.integers(0, N, size=N)
        f1 = precision_recall_fscore_support(
            y_true[idx], y_pred[idx], average="macro", zero_division=0
        )[2]
        vals.append(f1)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)

def mcnemar_test(y_true, pred_A, pred_B):
    """
    McNemar test.
    A = model to evaluate, B = reference/baseline model.
    """
    y_true = np.asarray(y_true)
    A = np.asarray(pred_A)
    B = np.asarray(pred_B)
    a_correct = A == y_true
    b_correct = B == y_true
    b = int(np.sum(a_correct & ~b_correct))  # A correct / B wrong
    c = int(np.sum(~a_correct & b_correct))  # A wrong / B correct
    denom = b + c
    if denom == 0:
        return b, c, 0.0, 1.0
    chi2_stat = (abs(b - c) - 1) ** 2 / denom  # continuity correction
    p = chi2.sf(chi2_stat, df=1)
    return b, c, float(chi2_stat), float(p)

def mcnemar_bar_plot(b, c, title="McNemar Contingency"):
    """Simple bar plot for McNemar b/c counts."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["Model correct / Ref wrong", "Model wrong / Ref correct"], [b, c])
    ax.set_ylabel("Count")
    ax.set_title(title)
    for i, v in enumerate([b, c]):
        ax.text(i, v + 0.5, str(v), ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.2)
    return fig

def plot_multiclass_roc(y_true, probs, class_labels, model_name="Model"):
    """
    Multiclass ROC (one-vs-rest) with macro AUC.
    y_true: (N,)
    probs: (N, C)
    class_labels: list of class names in order
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    classes = np.arange(probs.shape[1])

    y_bin = label_binarize(y_true, classes=classes)

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= len(classes)
    macro_auc = auc(all_fpr, mean_tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(len(classes)):
        ax.plot(
            fpr[i], tpr[i],
            label=f"Class {class_labels[i]} (AUC = {roc_auc[i]:.3f})",
            linewidth=1.5
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves ({model_name}) - Macro AUC = {macro_auc:.3f}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)

    return fig, macro_auc, roc_auc

def compute_metrics_trainer(p):
    """Metric function for HuggingFace Trainer."""
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    f1_macro = precision_recall_fscore_support(
        p.label_ids, preds, average='macro', zero_division=0
    )[2]
    return {"accuracy": acc, "f1_macro": f1_macro}

# ======================= Shared Stratified Split (70/10/20) =======================
def stratified_70_10_20(X, y):
    """
    Shared split for both models:
    - 70% train
    - 10% validation
    - 20% shared test
    """
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    trval_idx, test_idx = next(sss1.split(X, y))
    X_trval, X_test = X[trval_idx], X[test_idx]
    y_trval, y_test = y[trval_idx], y[test_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
    tr_idx, val_idx = next(sss2.split(X_trval, y_trval))
    X_train, X_val = X_trval[tr_idx], X_trval[val_idx]
    y_train, y_val = y_trval[tr_idx], y_trval[val_idx]

    return Bunch(X_train=X_train, y_train=y_train,
                 X_val=X_val, y_val=y_val,
                 X_test=X_test, y_test=y_test)

# ======================= BERT: Train & Predict (Shared Split) =======================
def train_bert_model_with_split(split):
    """Train BERT on (train+val) and evaluate on shared test set."""
    tok = st.session_state['tokenizer']
    mdl = st.session_state['model']

    X_train_full = np.concatenate([split.X_train, split.X_val])
    y_train_full = np.concatenate([split.y_train, split.y_val])

    X_tr = list(X_train_full)
    y_tr = list(y_train_full)
    X_te = list(split.X_test)
    y_te = list(split.y_test)

    enc_tr = tok(X_tr, truncation=True, padding=True)
    enc_te = tok(X_te, truncation=True, padding=True)
    ds_tr = TextDataset(enc_tr, y_tr)
    ds_te = TextDataset(enc_te, y_te)

    args = TrainingArguments(
        output_dir='./bert_output',
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        logging_dir='./logs',
        disable_tqdm=True
    )

    trainer = Trainer(
        model=mdl,
        args=args,
        train_dataset=ds_tr,
        eval_dataset=ds_te,
        compute_metrics=compute_metrics_trainer
    )

    with st.spinner("يتم تدريب نموذج BERT..."):
        trainer.train()
        quick_eval = trainer.evaluate()

    st.session_state['model'] = trainer.model
    return trainer, quick_eval, ds_te, X_te, y_te

def predict_texts_bert(texts):
    """Run BERT inference on raw texts."""
    if isinstance(texts, str):
        texts = [texts]
    tok = st.session_state['tokenizer']
    mdl = st.session_state['model']
    enc = tok(texts, truncation=True, padding=True, return_tensors="pt")
    enc = {k: v.to(mdl.device) for k, v in enc.items()}
    mdl.eval()
    with torch.no_grad():
        out = mdl(**enc)
        probs = torch.softmax(out.logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    id2label_local = {0: 'NEG', 1: 'POS', 2: 'NEU'}
    labels = [id2label_local[int(p)] for p in preds]
    return labels, probs

# ======================= BERT: Introspection =======================
def bert_introspection(sample_text: str, id2label: dict):
    """Inspect BERT classifier head: W, b, h[CLS], logits and probs."""
    if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
        raise RuntimeError("نموذج BERT غير محمّل في session_state.")

    tok = st.session_state['tokenizer']
    mdl = st.session_state['model']
    mdl.eval()

    enc = tok(sample_text, truncation=True, padding=True, return_tensors="pt")
    enc = {k: v.to(mdl.device) for k, v in enc.items()}

    with torch.no_grad():
        out = mdl(**enc, output_hidden_states=True, return_dict=True)

    logits_model = out.logits[0].cpu().numpy()
    hidden_last = out.hidden_states[-1]
    cls_vec = hidden_last[0, 0, :].cpu().numpy()

    classifier = mdl.classifier
    W = classifier.weight.detach().cpu().numpy()
    b = classifier.bias.detach().cpu().numpy()

    logits_manual = W @ cls_vec + b

    probs_model = softmax_1d(logits_model)
    probs_manual = softmax_1d(logits_manual)

    rows = []
    num_labels = W.shape[0]
    for i in range(num_labels):
        rows.append({
            "class_id": i,
            "class_label": id2label[i],
            "logit_from_model": float(logits_model[i]),
            "logit_manual_W_hCLS_plus_b": float(logits_manual[i]),
            "prob_from_model_softmax": float(probs_model[i]),
            "prob_from_manual_softmax": float(probs_manual[i]),
        })

    proba_df = pd.DataFrame(rows).sort_values(
        "prob_from_model_softmax", ascending=False
    ).reset_index(drop=True)

    return W, b, cls_vec, proba_df

# ======================= Logistic Regression (TF-IDF) =======================
ARABIC_DIACRITICS = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
TATWEEL = '\u0640'

def normalize_ar(text: str) -> str:
    """Basic Arabic normalization used before TF-IDF."""
    if not isinstance(text, str):
        return ""
    text = ARABIC_DIACRITICS.sub('', text).replace(TATWEEL, '')
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[^0-9a-zA-Z\u0621-\u064A\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def make_vectorizer(min_df=2, max_df=0.95):
    """Create TF-IDF vectorizer for Arabic unigrams/bigrams."""
    token_pattern = r'(?u)\b[\u0621-\u064A]+\b'
    return TfidfVectorizer(
        preprocessor=normalize_ar,
        token_pattern=token_pattern,
        ngram_range=(1, 2),
        min_df=min_df, max_df=max_df,
        sublinear_tf=True, lowercase=False
    )

def evaluate_logreg(pipe, X, y, name="model"):
    """Evaluate Logistic Regression on given X, y."""
    y_pred = pipe.predict(X)
    probs = pipe.predict_proba(X)
    ll = log_loss(y, probs, labels=np.unique(y))
    acc = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average="macro")
    weighted = precision_recall_fscore_support(y, y_pred, average="weighted", zero_division=0)
    prec_w, rec_w, f1_w = weighted[:3]
    return {
        "name": name, "accuracy": acc, "macro_f1": macro_f1,
        "precision_weighted": prec_w, "recall_weighted": rec_w,
        "f1_weighted": f1_w, "log_loss": ll, "y_pred": y_pred,
        "report": classification_report(y, y_pred)
    }

def train_logreg_model_with_split(split, id2label):
    """Train Logistic Regression using the same shared split."""
    vect = make_vectorizer()

    Cs = [0.01, 0.1, 0.5, 1, 2, 5, 10]
    best_f1, best_C = -1, None

    for c in Cs:
        pipe = Pipeline([
            ("tfidf", vect),
            ("clf", LogisticRegression(C=c, max_iter=1000, solver="lbfgs", multi_class="auto"))
        ])
        pipe.fit(split.X_train, split.y_train)
        metrics = evaluate_logreg(pipe, split.X_val, split.y_val, name=f"logreg_C{c}")
        if metrics["macro_f1"] > best_f1:
            best_f1, best_C = metrics["macro_f1"], c

    X_tr_full = np.concatenate([split.X_train, split.X_val])
    y_tr_full = np.concatenate([split.y_train, split.y_val])

    best_pipe = Pipeline([
        ("tfidf", vect),
        ("clf", LogisticRegression(C=best_C, max_iter=1000, solver="lbfgs", multi_class="auto"))
    ])
    best_pipe.fit(X_tr_full, y_tr_full)

    test_metrics = evaluate_logreg(best_pipe, split.X_test, split.y_test, name=f"logreg_C{best_C}")

    cm = confusion_matrix(split.y_test, test_metrics["y_pred"])
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(
        cm, display_labels=[id2label[i] for i in range(len(id2label))]
    ).plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    ax.set_title("Logistic Regression Confusion Matrix (Shared Test)")

    probs_test = best_pipe.predict_proba(split.X_test)

    preds_df = pd.DataFrame({
        "text": split.X_test,
        "true_label_id": split.y_test,
        "true_label": [id2label[i] for i in split.y_test],
        "pred_label_id": test_metrics["y_pred"],
        "pred_label": [id2label[i] for i in test_metrics["y_pred"]],
    })

    return best_pipe, test_metrics, preds_df, fig, probs_test

# ======================= Logistic Regression Introspection =======================
def logreg_introspection(pipe: Pipeline, id2label: dict, sample_text: str):
    """Inspect Logistic Regression: W, b, TF-IDF vector x, logits and probs."""
    vect_fitted: TfidfVectorizer = pipe.named_steps["tfidf"]
    clf: LogisticRegression = pipe.named_steps["clf"]

    feature_names = np.array(vect_fitted.get_feature_names_out())
    classes = clf.classes_

    X_samp = vect_fitted.transform([sample_text])
    x_dense = X_samp.toarray()[0]

    W = clf.coef_
    b = clf.intercept_

    logits = (X_samp @ W.T).toarray()[0] + b
    probs_manual = softmax_1d(logits)
    probs_model = clf.predict_proba(X_samp)[0]

    rows = []
    for i, cid in enumerate(classes):
        rows.append({
            "class_id": int(cid),
            "class_label": id2label[int(cid)],
            "logit_xWT_plus_b": float(logits[i]),
            "prob_from_model_predict_proba": float(probs_model[i]),
            "prob_from_manual_softmax": float(probs_manual[i]),
        })

    proba_df = pd.DataFrame(rows).sort_values(
        "prob_from_model_predict_proba", ascending=False
    ).reset_index(drop=True)

    return W, b, x_dense, feature_names, proba_df

# ======================= UI: Excel Upload & Main Flow =======================
uploaded_file = st.file_uploader(
    "📥 ارفع ملف Excel (يحتوي الأعمدة: Tweet, Final annotation, Tokens without stop words)",
    type=["xlsx"]
)

if uploaded_file is not None:
    df, label_map, id2label = load_data(uploaded_file)
    st.write("### معاينة البيانات:")
    st.dataframe(df[['text', 'Final annotation']].head(10))

    # Build shared X, y and split
    X_all = df['text'].astype(str).values
    y_all = df['label'].values
    split = stratified_70_10_20(X_all, y_all)

    # Load BERT model & tokenizer if not in session_state
    if 'tokenizer' not in st.session_state or 'model' not in st.session_state:
        tok, mdl = load_model_and_tokenizer()
        st.session_state['tokenizer'] = tok
        st.session_state['model'] = mdl

    # -------------------- BERT Training (Shared Split) --------------------
    st.markdown("## 🤖 تدريب نموذج BERT (باستخدام نفس مجموعة الاختبار المشتركة)")

    if st.button("▶️ ابدأ التدريب بـ BERT (Shared Split)"):
        trainer, quick_eval, test_ds, X_test_bert, y_test_bert = train_bert_model_with_split(split)
        st.success("تم تدريب نموذج BERT بنجاح!")
        st.write("**مقاييس سريعة (Trainer):**", quick_eval)

        pred_out = trainer.predict(test_ds)
        y_true = pred_out.label_ids
        logits = pred_out.predictions
        probs = softmax_np(logits)
        y_pred = probs.argmax(axis=1)
        C = probs.shape[1]

        st.write("### مصفوفة الالتباس (BERT - Shared Test):")
        cm = confusion_matrix(y_true, y_pred, labels=list(range(C)))
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(cm, display_labels=[id2label[i] for i in range(C)])
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)
        ax.set_title("Confusion Matrix (BERT Test - Shared Split)")
        st.pyplot(fig)

        st.write("### ROC Curves + AUC (BERT)")
        class_labels = [id2label[i] for i in range(C)]
        fig_roc_bert, macro_auc_bert, roc_auc_bert = plot_multiclass_roc(
            y_true, probs, class_labels, model_name="BERT"
        )
        st.pyplot(fig_roc_bert)
        st.write(f"**Macro AUC (OVR) لـ BERT = {macro_auc_bert:.3f}**")

        st.markdown("## 📊 مقاييس مفصلة (BERT)")

        acc = accuracy_score(y_true, y_pred)
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        prec_c, rec_c, f1_c, support_c = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=list(range(C)), zero_division=0
        )

        ll = log_loss(y_true, probs, labels=list(range(C)))
        brier = brier_multiclass(y_true, probs, C)
        ece_val = ece(probs, y_true, n_bins=15)
        ci_lo, ci_hi = bootstrap_macro_f1(y_true, y_pred, B=1000, seed=42)

        detailed_metrics_df = pd.DataFrame({
            "metric": [
                "accuracy", "precision_weighted", "recall_weighted", "f1_weighted",
                "precision_macro", "recall_macro", "f1_macro",
                "log_loss", "brier_score", "ece_15bins",
                "f1_macro_ci_lo", "f1_macro_ci_hi"
            ],
            "value": [
                acc, prec_w, rec_w, f1_w,
                prec_m, rec_m, f1_m,
                ll, brier, ece_val,
                ci_lo, ci_hi
            ]
        })
        st.dataframe(detailed_metrics_df, use_container_width=True)

        per_class_df = pd.DataFrame({
            "class_id": list(range(C)),
            "class_label": [id2label[i] for i in range(C)],
            "precision": prec_c,
            "recall": rec_c,
            "f1": f1_c,
            "support": support_c
        })
        st.write("### تقرير حسب الفئة (BERT Per-class):")
        st.dataframe(per_class_df, use_container_width=True)

        st.write("### Reliability Diagram (ECE) - BERT")
        fig_rel = reliability_diagram(probs, y_true, n_bins=15)
        st.pyplot(fig_rel)

        st.write("### 💾 تنزيل نتائج BERT كملفات CSV")
        det_buf = io.StringIO()
        detailed_metrics_df.to_csv(det_buf, index=False)
        st.download_button(
            "⬇️ تنزيل Detailed Metrics (BERT)",
            data=det_buf.getvalue().encode(),
            file_name="bert_detailed_metrics_shared.csv",
            mime="text/csv"
        )

        per_buf = io.StringIO()
        per_class_df.to_csv(per_buf, index=False)
        st.download_button(
            "⬇️ تنزيل Per-class Report (BERT)",
            data=per_buf.getvalue().encode(),
            file_name="bert_per_class_report_shared.csv",
            mime="text/csv"
        )

        pred_buf = io.StringIO()
        audit_df = pd.DataFrame({
            "text": X_test_bert,
            "true_label_id": y_true,
            "true_label": [id2label[i] for i in y_true],
            "pred_label_id": y_pred,
            "pred_label": [id2label[i] for i in y_pred],
            "prob_NEG": probs[:, 0],
            "prob_POS": probs[:, 1],
            "prob_NEU": probs[:, 2],
        })
        audit_df.to_csv(pred_buf, index=False)
        st.download_button(
            "⬇️ تنزيل تنبؤات BERT للاختبار مع الاحتمالات",
            data=pred_buf.getvalue().encode(),
            file_name="bert_test_predictions_with_probs_shared.csv",
            mime="text/csv"
        )

        st.markdown("### ⚖️ اختبار ماكنيمار (BERT vs Majority Baseline)")

        maj = pd.Series(y_true).mode().iloc[0]
        baseline = np.full_like(y_true, maj)
        b_val, c_val, chi2_stat, p_val = mcnemar_test(y_true, y_pred, baseline)
        st.write(f"Baseline (الأكثر شيوعًا): **{id2label[maj]}**")
        st.write(f"b (BERT صحيح / المرجع خطأ) = **{b_val}**,  c (BERT خطأ / المرجع صحيح) = **{c_val}**")
        st.write(f"χ² = **{chi2_stat:.4f}**,  p = **{p_val:.4g}**")

        fig_mcn_baseline = mcnemar_bar_plot(
            b_val, c_val,
            title="McNemar: BERT vs Majority Baseline (Shared Test)"
        )
        st.pyplot(fig_mcn_baseline)

        st.session_state["bert_y_true"] = y_true
        st.session_state["bert_y_pred"] = y_pred
        st.session_state["bert_probs"] = probs

    # -------------------- Logistic Regression Training (Shared Split) --------------------
    st.markdown("---")
    st.markdown("## 📈 تدريب Logistic Regression (TF-IDF) على نفس الـ split")

    if st.button("▶️ درّب Logistic Regression (Shared Split)"):
        with st.spinner("يتم تدريب Logistic Regression..."):
            best_pipe, test_metrics_lr, preds_df_lr, fig_lr_cm, probs_lr_test = \
                train_logreg_model_with_split(split, id2label)

        st.success("تم تدريب Logistic Regression بنجاح!")
        st.write("### مصفوفة الالتباس (Logistic Regression - Shared Test):")
        st.pyplot(fig_lr_cm)

        st.write("### مقاييس الاختبار (Logistic Regression):")
        metrics_view = {
            k: v for k, v in test_metrics_lr.items()
            if k not in ("y_pred", "report")
        }
        st.json(metrics_view)

        st.write("### Classification Report (Logistic Regression):")
        st.text(test_metrics_lr["report"])

        buf_lr = io.StringIO()
        preds_df_lr.to_csv(buf_lr, index=False)
        st.download_button(
            "⬇️ تنزيل تنبؤات الاختبار (Logistic Regression)",
            data=buf_lr.getvalue().encode(),
            file_name="logreg_test_predictions_shared.csv",
            mime="text/csv"
        )

        st.write("### ROC Curves + AUC (Logistic Regression)")
        class_labels_lr = [id2label[i] for i in range(len(id2label))]
        fig_roc_lr, macro_auc_lr, roc_auc_lr = plot_multiclass_roc(
            split.y_test, probs_lr_test, class_labels_lr, model_name="Logistic Regression"
        )
        st.pyplot(fig_roc_lr)
        st.write(f"**Macro AUC (OVR) لـ Logistic Regression = {macro_auc_lr:.3f}**")

        st.write("### ⚖️ اختبار ماكنيمار (LogReg vs Majority Baseline)")

        maj_lr = pd.Series(split.y_test).mode().iloc[0]
        baseline_lr = np.full_like(split.y_test, maj_lr)
        b_lr_val, c_lr_val, chi2_lr, p_lr = mcnemar_test(
            split.y_test, test_metrics_lr["y_pred"], baseline_lr
        )

        st.write(f"Baseline (الأكثر شيوعًا في مجموعة الاختبار): **{id2label[maj_lr]}**")
        st.write(f"b = **{b_lr_val}**, c = **{c_lr_val}**, χ² = **{chi2_lr:.4f}**, p = **{p_lr:.4g}**")

        fig_mcn_lr = mcnemar_bar_plot(
            b_lr_val, c_lr_val,
            title="McNemar: LogReg vs Majority Baseline (Shared Test)"
        )
        st.pyplot(fig_mcn_lr)

        st.session_state["logreg_y_true"] = split.y_test
        st.session_state["logreg_y_pred"] = test_metrics_lr["y_pred"]
        st.session_state["logreg_pipe"] = best_pipe
        st.session_state["logreg_probs"] = probs_lr_test

    # -------------------- AUC comparison on shared test --------------------
    st.markdown("---")
    st.header("📈 مقارنة Macro AUC بين BERT و Logistic Regression (نفس مجموعة الاختبار)")

    if ("bert_probs" in st.session_state and "bert_y_true" in st.session_state and
        "logreg_probs" in st.session_state and "logreg_y_true" in st.session_state):
        try:
            auc_bert = roc_auc_score(
                st.session_state["bert_y_true"],
                st.session_state["bert_probs"],
                multi_class="ovr",
                average="macro"
            )
            auc_lr = roc_auc_score(
                st.session_state["logreg_y_true"],
                st.session_state["logreg_probs"],
                multi_class="ovr",
                average="macro"
            )
            st.write(f"**Macro AUC (OVR) لـ BERT = {auc_bert:.3f}**")
            st.write(f"**Macro AUC (OVR) لـ Logistic Regression = {auc_lr:.3f}**")
            st.success("هذه مقارنة عادلة 100٪ لأن كلا النموذجين تم تقييمهما على نفس مجموعة الاختبار بالضبط. ✅")
        except Exception as e:
            st.error(f"تعذر حساب AUC المقارن: {e}")
    else:
        st.info("درّب النموذجين (BERT و Logistic Regression) أولاً لعرض مقارنة AUC على نفس مجموعة الاختبار.")

    # ======================= Manual inference (BERT) =======================
    st.markdown("---")
    st.header("⚡ تصنيف نصوص جديدة باستخدام BERT")

    input_text = st.text_area("أدخل نصًا أو عدة نصوص (كل سطر نص منفصل):", height=150)

    if st.button("تصنيف النصوص بـ BERT"):
        if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
            st.error("فضلاً درّب نموذج BERT أولاً.")
        else:
            lines = [t.strip() for t in input_text.split('\n') if t.strip()]
            if not lines:
                st.warning("أدخل على الأقل نصًا واحدًا.")
            else:
                labels, probs = predict_texts_bert(lines)
                for t, l, p in zip(lines, labels, probs):
                    st.write(f"**النص:** {t}")
                    st.write(f"↳ التنبؤ: **{l}**")
                    st.write(f"↳ الاحتمالات: NEG={p[0]:.3f} | POS={p[1]:.3f} | NEU={p[2]:.3f}")
                    st.write("---")

    # ======================= CSV inference (BERT) =======================
    st.markdown("---")
    st.header("📂 تصنيف نصوص من ملف CSV باستخدام BERT")

    uploaded_csv = st.file_uploader("ارفع ملف CSV يحوي عمود نصوص (مثلاً: text)", type=["csv"], key="csv_uploader")

    if uploaded_csv is not None:
        if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
            st.error("فضلاً درّب نموذج BERT أولاً.")
        else:
            try:
                df_new = pd.read_csv(uploaded_csv)
            except Exception as e:
                st.error(f"خطأ قراءة CSV: {e}")
                df_new = None

            if df_new is not None:
                text_column = st.text_input("اسم عمود النصوص داخل CSV:", "text")
                if text_column not in df_new.columns:
                    st.warning(f"العمود '{text_column}' غير موجود.")
                else:
                    if st.button("▶️ تصنيف ملف CSV بـ BERT"):
                        texts = df_new[text_column].astype(str).tolist()
                        labels, probs = predict_texts_bert(texts)
                        df_new['Pred_Label'] = labels
                        df_new['Prob_NEG'] = probs[:, 0]
                        df_new['Prob_POS'] = probs[:, 1]
                        df_new['Prob_NEU'] = probs[:, 2]

                        st.success("تم التصنيف!")
                        st.dataframe(df_new.head(10))

                        buf = io.StringIO()
                        df_new.to_csv(buf, index=False)
                        st.download_button(
                            "⬇️ تنزيل CSV بعد التصنيف (BERT)",
                            data=buf.getvalue().encode(),
                            file_name="bert_classified_texts.csv",
                            mime="text/csv"
                        )

    # ======================= BERT Head Introspection =======================
    st.markdown("---")
    st.header("🔬 تحليل رأس التصنيف في BERT (W و h(CLS)+b مع softmax)")

    if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
        st.info("درّب نموذج BERT أولاً قبل التحليل.")
    else:
        sample_text_bert = st.text_area(
            "أدخل تغريدة/نص لتحليلها من منظور رأس التصنيف في BERT:",
            value="للأسف تم احتسابها لي بطريقة خاطئه كيف يتم الاعتراض على الاحتساب",
            height=100,
            key="bert_introspection_text"
        )

        if st.button("عرض مصفوفة W و h(CLS)+b و softmax من BERT"):
            try:
                W_bert, b_bert, cls_vec, proba_df_bert = bert_introspection(
                    sample_text_bert, id2label
                )

                st.subheader("1️⃣ مصفوفة الأوزان W لرأس التصنيف في BERT")
                st.write(f"شكل W: {W_bert.shape}  (عدد الكلاسات × حجم التمثيل المخفي)")
                st.write("مثال على أول 5 أسطر و أول 10 أعمدة من W:")
                st.dataframe(
                    pd.DataFrame(
                        W_bert[:5, :10]
                    )
                )

                st.subheader("2️⃣ متجه التحيّز b")
                st.write(b_bert)

                st.subheader("3️⃣ تمثيل [CLS] (من آخر طبقة مخفية)")
                st.write(f"شكل h(CLS): {cls_vec.shape}")
                st.write("أول 10 قيم من h(CLS):")
                st.write(cls_vec[:10])

                st.subheader("4️⃣ اللوجِت و softmax لكل كلاس")
                st.dataframe(proba_df_bert)

            except NameError:
                st.error("المتغير id2label غير متوفر. تأكد أنك رفعت ملف الإكسل (load_data) قبل استخدام هذا الجزء.")

    # ======================= Logistic Regression Introspection UI =======================
    st.markdown("---")
    st.header("🔎 تحليل Logistic Regression (W و xWᵀ + b مع softmax)")

    if "logreg_pipe" not in st.session_state:
        st.info("درّب Logistic Regression أولاً حتى يمكن عرض W و xWᵀ + b.")
    else:
        sample_text_lr = st.text_area(
            "أدخل تغريدة/نص لتحليلها باستخدام Logistic Regression:",
            value="للأسف تم احتسابها لي بطريقة خاطئه كيف يتم الاعتراض على الاحتساب",
            height=100,
            key="logreg_introspection_text"
        )

        if st.button("عرض مصفوفة W و xWᵀ + b و softmax لـ Logistic Regression"):
            pipe = st.session_state["logreg_pipe"]

            try:
                W_lr, b_lr, x_vec_lr, feature_names_lr, proba_df_lr = logreg_introspection(
                    pipe, id2label, sample_text_lr
                )

                st.subheader("1️⃣ مصفوفة الأوزان W لـ Logistic Regression")
                st.write(f"شكل W: {W_lr.shape}  (عدد الكلاسات × عدد الخصائص TF-IDF)")
                st.write("مثال على أول 5 أسطر و أول 10 أعمدة من W:")
                st.dataframe(
                    pd.DataFrame(
                        W_lr[:5, :10],
                        columns=feature_names_lr[:10]
                    )
                )

                st.subheader("2️⃣ متجه التحيّز b")
                st.write(b_lr)

                st.subheader("3️⃣ متجه x (TF-IDF) للنص المدخل")
                st.write(f"طول x: {len(x_vec_lr)} (يساوي عدد الخصائص في TF-IDF)")
                st.write("أكبر 10 قيم في x (أكثر n-grams نشاطًا):")
                top_idx = np.argsort(-np.abs(x_vec_lr))[:10]
                top_x_df = pd.DataFrame({
                    "feature_index": top_idx,
                    "feature_name": feature_names_lr[top_idx],
                    "x_value": x_vec_lr[top_idx]
                })
                st.dataframe(top_x_df)

                st.subheader("4️⃣ اللوجِت و softmax لكل كلاس (logreg)")
                st.dataframe(proba_df_lr)

            except Exception as e:
                st.error(f"خطأ أثناء تحليل Logistic Regression: {e}")

else:
    st.info("📄 فضلاً ارفع ملف Excel لبداية العمل.")
