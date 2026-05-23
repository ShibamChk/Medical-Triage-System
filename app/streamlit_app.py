from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import PROJECT_ROOT, CLASS_NAMES
from src.inference.predict import (
    load_trained_model,
    predict_image,
    load_image_as_pil,
    get_inference_transform,
)
from src.explainability.gradcam import (
    GradCAM,
    get_target_layer,
    overlay_heatmap_on_image,
    pil_to_numpy_rgb,
)


DEFAULT_MODEL_NAME = "convnext_tiny"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_convnext_tiny.pth"
DEFAULT_THRESHOLD = 0.29


st.set_page_config(
    page_title="MedTriage-CXR",
    page_icon="🫁",
    layout="wide",
)


@st.cache_resource
def load_model_cached(model_name: str, checkpoint_path: str):
    """
    Cache model loading so Streamlit does not reload the model every interaction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_trained_model(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    return model, device


def save_uploaded_file(uploaded_file) -> Path:
    """
    Save Streamlit uploaded file to a temporary file and return its path.

    We use delete=False because Windows can lock NamedTemporaryFile while another
    library tries to read it.
    """
    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def make_probability_dataframe(probabilities: dict) -> pd.DataFrame:
    rows = []

    for class_name, probability in probabilities.items():
        rows.append(
            {
                "Class": class_name,
                "Probability": probability,
            }
        )

    return pd.DataFrame(rows)


def get_priority_style(priority: str) -> str:
    if priority == "High Priority":
        return "🔴 High Priority"

    if priority == "Medium Priority":
        return "🟠 Medium Priority"

    return "🟢 Low Priority"


def generate_gradcam_overlay(
    model: torch.nn.Module,
    model_name: str,
    image_path: Path,
    target_class_index: int,
    device: torch.device,
):
    """
    Generate Grad-CAM overlay for one uploaded image.
    """
    pil_image = load_image_as_pil(image_path)

    transform = get_inference_transform()
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    target_layer = get_target_layer(
        model=model,
        model_name=model_name,
    )

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    cam, logits = gradcam.generate(
        input_tensor=input_tensor,
        target_class_index=target_class_index,
    )

    gradcam.remove_hooks()

    probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    image_np = pil_to_numpy_rgb(pil_image)
    overlay = overlay_heatmap_on_image(
        image_np=image_np,
        cam=cam,
        alpha=0.45,
    )

    return pil_image, cam, overlay, probabilities


def render_gradcam(
    model: torch.nn.Module,
    model_name: str,
    image_path: Path,
    target_class_index: int,
    device: torch.device,
):
    pil_image, cam, overlay, probabilities = generate_gradcam_overlay(
        model=model,
        model_name=model_name,
        image_path=image_path,
        target_class_index=target_class_index,
        device=device,
    )

    image_np = pil_to_numpy_rgb(pil_image)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title(f"Grad-CAM Target: {CLASS_NAMES[target_class_index]}")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay\nP(Lung Opacity): {probabilities[2]:.3f}")
    axes[2].axis("off")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


def main():
    st.title("MedTriage-CXR")
    st.caption(
        "AI-assisted chest X-ray triage demo using ConvNeXt-Tiny, "
        "threshold tuning, and Grad-CAM explainability."
    )

    st.warning(
        "This demo is for educational and portfolio purposes only. "
        "It is not a medical device and must not be used for clinical diagnosis."
    )

    with st.sidebar:
        st.header("Model Settings")

        model_name = st.selectbox(
            "Model architecture",
            options=["convnext_tiny", "resnet50"],
            index=0,
        )

        checkpoint_path = st.text_input(
            "Checkpoint path",
            value=str(DEFAULT_CHECKPOINT_PATH),
        )

        st.divider()

        st.header("Decision Mode")

        decision_mode = st.radio(
            "Prediction mode",
            options=[
                "Balanced classification",
                "High-sensitivity triage",
            ],
            index=1,
        )

        if decision_mode == "High-sensitivity triage":
            lung_opacity_threshold = st.slider(
                "Lung Opacity threshold",
                min_value=0.05,
                max_value=0.95,
                value=DEFAULT_THRESHOLD,
                step=0.01,
                help=(
                    "Lower threshold increases Lung Opacity recall but may "
                    "increase false positives."
                ),
            )
        else:
            lung_opacity_threshold = None

        st.divider()

        show_gradcam = st.checkbox(
            "Generate Grad-CAM explanation",
            value=True,
        )

        gradcam_target_mode = st.selectbox(
            "Grad-CAM target",
            options=[
                "Predicted class",
                "Lung Opacity class",
            ],
            index=1,
        )

    checkpoint_path_obj = Path(checkpoint_path)

    if not checkpoint_path_obj.exists():
        st.error(
            f"Checkpoint not found: {checkpoint_path_obj}\n\n"
            "Train the model first or place the checkpoint in the models folder."
        )
        st.stop()

    model, device = load_model_cached(
        model_name=model_name,
        checkpoint_path=str(checkpoint_path_obj),
    )

    st.success(f"Loaded model: {model_name} on {device}")

    uploaded_file = st.file_uploader(
        "Upload a chest X-ray file",
        type=["dcm", "png", "jpg", "jpeg"],
    )

    if uploaded_file is None:
        st.info("Upload a DICOM, PNG, JPG, or JPEG chest X-ray image to run inference.")
        return

    temp_image_path = save_uploaded_file(uploaded_file)

    try:
        pil_image = load_image_as_pil(temp_image_path)
    except Exception as error:
        st.error(f"Failed to load uploaded image: {error}")
        return

    col_image, col_result = st.columns([1, 1])

    with col_image:
        st.subheader("Input Image")
        st.image(
            pil_image,
            caption=uploaded_file.name,
            use_container_width=True,
        )

    with col_result:
        st.subheader("Prediction")

        try:
            result = predict_image(
                image_path=temp_image_path,
                model=model,
                device=device,
                lung_opacity_threshold=lung_opacity_threshold,
            )
        except Exception as error:
            st.error(f"Prediction failed: {error}")
            return

        priority_text = get_priority_style(result["triage_priority"])

        st.metric(
            label="Predicted Class",
            value=result["predicted_class"],
        )

        st.metric(
            label="Triage Priority",
            value=priority_text,
        )

        st.metric(
            label="Confidence",
            value=f"{result['confidence']:.3f}",
        )

        st.write("**Decision mode:**", result["decision_mode"])

        if lung_opacity_threshold is not None:
            st.write("**Lung Opacity threshold:**", lung_opacity_threshold)

        probabilities_df = make_probability_dataframe(result["probabilities"])

        st.subheader("Class Probabilities")
        st.dataframe(probabilities_df, use_container_width=True)
        st.bar_chart(
            probabilities_df.set_index("Class"),
            y="Probability",
        )

    if show_gradcam:
        st.divider()
        st.subheader("Grad-CAM Explainability")

        if gradcam_target_mode == "Predicted class":
            target_class_index = int(result["predicted_label"])
        else:
            target_class_index = 2

        try:
            render_gradcam(
                model=model,
                model_name=model_name,
                image_path=temp_image_path,
                target_class_index=target_class_index,
                device=device,
            )
        except Exception as error:
            st.error(f"Grad-CAM generation failed: {error}")

    with st.expander("Raw Prediction Output"):
        st.json(result)


if __name__ == "__main__":
    main()