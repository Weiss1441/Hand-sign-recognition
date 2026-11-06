"""
Hand Gesture Recognition demo (ResNet18 + real Grad-CAM).

Run:
    pip install streamlit torch torchvision opencv-python pillow numpy
    pip install mediapipe        # optional, enables automatic hand cropping
    streamlit run app.py

Put gesture_model_sota.pth next to this file.
"""
import time

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torchvision import models, transforms

WEIGHTS_PATH = "gesture_model_sota.pth"
# ImageFolder sorts class folders alphabetically. This MUST match training.
DEFAULT_CLASSES = ["peace", "stop", "thumbsup"]
IMG_SIZE = 128  # same as training in the notebook
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

st.set_page_config(page_title="Hand Gesture Demo", page_icon="🖐️", layout="wide")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Returns (model, classes). Supports a plain state_dict or a dict
    {"state_dict": ..., "classes": [...]}."""
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")
    classes = DEFAULT_CLASSES
    state = ckpt
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        classes = list(ckpt.get("classes", DEFAULT_CLASSES))

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(state)  # raises on mismatch instead of failing silently
    model.eval()
    return model, classes


try:
    model, CLASSES = load_model()
except Exception as e:
    st.error(f"Could not load `{WEIGHTS_PATH}`: {e}")
    st.stop()

preprocess = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)


def predict(img: Image.Image, use_flip_tta: bool):
    """Returns probabilities (np array), logits (np array), latency in ms."""
    x = preprocess(img).unsqueeze(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        if use_flip_tta:  # training used horizontal flips, so this is safe
            logits_f = model(torch.flip(x, dims=[3]))
            probs = (probs + F.softmax(logits_f, dim=1)) / 2
            logits = (logits + logits_f) / 2
    latency = (time.perf_counter() - t0) * 1000
    return probs[0].numpy(), logits[0].numpy(), latency


# --------------------------------------------------------------------------
# Real Grad-CAM on the last conv block (layer4)
# --------------------------------------------------------------------------
def gradcam(img: Image.Image, class_idx: int) -> np.ndarray:
    """Returns the CAM in [0, 1] with the size of the model input."""
    x = preprocess(img).unsqueeze(0)
    store = {}

    def fwd_hook(_, __, output):
        store["act"] = output
        output.register_hook(lambda g: store.__setitem__("grad", g))

    handle = model.layer4.register_forward_hook(fwd_hook)
    try:
        with torch.enable_grad():
            model.zero_grad()
            out = model(x)
            out[0, class_idx].backward()
    finally:
        handle.remove()

    weights = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store["act"]).sum(dim=1))[0].detach().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam


def overlay_cam(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb = np.array(img)
    h, w = rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(cam_resized, 0, 1)), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, heat, alpha, 0)


# --------------------------------------------------------------------------
# Cropping: the hand must be big in the frame, like in the training photos
# --------------------------------------------------------------------------
def center_square_box(w, h):
    s = min(w, h)
    l, t = (w - s) // 2, (h - s) // 2
    return l, t, l + s, t + s


def clamp_box(l, t, r, b, w, h):
    return max(0, int(l)), max(0, int(t)), min(w, int(r)), min(h, int(b))


@st.cache_resource
def get_hands_detector():
    try:
        import mediapipe as mp

        return mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=1, min_detection_confidence=0.4
        )
    except Exception:
        return None


def auto_hand_box(img: Image.Image, padding: float = 0.35):
    """Square box around the detected hand, or None if not found."""
    detector = get_hands_detector()
    if detector is None:
        return None
    w, h = img.size
    res = detector.process(np.array(img))
    if not res.multi_hand_landmarks:
        return None
    pts = res.multi_hand_landmarks[0].landmark
    xs = [p.x * w for p in pts]
    ys = [p.y * h for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + 2 * padding)
    return clamp_box(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2, w, h)


def draw_box(img: Image.Image, box):
    out = img.copy()
    ImageDraw.Draw(out).rectangle(box, outline=(0, 255, 0), width=max(3, img.width // 200))
    return out


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🖐️ Hand Gesture Recognition")
st.caption(f"ResNet18 (transfer learning + ArtAug) · classes: {', '.join(CLASSES)}")

with st.sidebar:
    st.header("Settings")
    source = st.radio("Input source", ["Upload photo", "Webcam"])

    st.subheader("Crop")
    crop_modes = ["Center square", "Manual", "Full frame (squashed)"]
    if get_hands_detector() is not None:
        crop_modes.insert(0, "Auto (MediaPipe hand)")
    crop_mode = st.selectbox(
        "Crop mode",
        crop_modes,
        help="The model was trained on close-ups of a hand. "
        "On a wide webcam frame the hand is tiny, so crop around it.",
    )
    if get_hands_detector() is None:
        st.caption("Install `mediapipe` for automatic hand cropping.")

    st.subheader("Prediction")
    threshold = st.slider("Min. confidence", 0, 100, 60, help="Below this the result is marked uncertain.")
    use_tta = st.checkbox("Flip test-time augmentation", value=True)
    show_cam = st.checkbox("Show Grad-CAM", value=True)

if source == "Upload photo":
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
else:
    file = st.camera_input("Take a snapshot")

if file is None:
    st.info("Upload a photo or take a snapshot. Hold your hand close to the camera.")
    st.stop()

image = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
W, H = image.size

# ---- choose crop box ----
box = (0, 0, W, H)
if crop_mode == "Auto (MediaPipe hand)":
    found = auto_hand_box(image)
    if found is None:
        st.warning("No hand detected, falling back to center square crop.")
        box = center_square_box(W, H)
    else:
        box = found
elif crop_mode == "Center square":
    box = center_square_box(W, H)
elif crop_mode == "Manual":
    with st.sidebar:
        x_rng = st.slider("Horizontal range (%)", 0, 100, (30, 100))
        y_rng = st.slider("Vertical range (%)", 0, 100, (0, 100))
    box = clamp_box(W * x_rng[0] / 100, H * y_rng[0] / 100, W * x_rng[1] / 100, H * y_rng[1] / 100, W, H)
    if box[2] - box[0] < 10 or box[3] - box[1] < 10:
        st.error("Crop area is too small.")
        st.stop()

cropped = image.crop(box)

# ---- inference ----
probs, logits, latency_ms = predict(cropped, use_tta)
pred_idx = int(np.argmax(probs))
pred_class = CLASSES[pred_idx]
confidence = float(probs[pred_idx]) * 100

col1, col2, col3 = st.columns([1.2, 1, 1.2])

with col1:
    st.subheader("Input")
    st.image(draw_box(image, box), caption="Green box = area given to the model", use_container_width=True)

with col2:
    st.subheader("Model input")
    st.image(cropped.resize((IMG_SIZE * 2, IMG_SIZE * 2)), caption=f"{IMG_SIZE}×{IMG_SIZE} after resize", use_container_width=True)

with col3:
    st.subheader("Prediction")
    if confidence >= threshold:
        st.metric("Recognized gesture", pred_class.upper(), f"{confidence:.1f}% confidence")
    else:
        st.metric("Recognized gesture", "UNCERTAIN", f"best guess: {pred_class} ({confidence:.1f}%)", delta_color="off")
        st.warning("Low confidence. Move the hand closer, improve lighting, or adjust the crop.")

    for name, p in sorted(zip(CLASSES, probs), key=lambda t: -t[1]):
        st.progress(int(p * 100), text=f"{name}: {p * 100:.1f}%")

    st.caption(f"Latency: {latency_ms:.1f} ms · device: CPU · TTA: {'on' if use_tta else 'off'}")

with st.expander("Raw logits (debug)"):
    st.write({c: round(float(v), 3) for c, v in zip(CLASSES, logits)})
    st.caption("If logits are almost identical for different images, the model is not seeing the hand.")

if show_cam:
    st.markdown("---")
    st.subheader("Explainability: Grad-CAM (layer4)")
    cam = gradcam(cropped, pred_idx)
    c1, c2 = st.columns(2)
    with c1:
        st.image(cropped, caption="Model input (crop)", use_container_width=True)
    with c2:
        st.image(overlay_cam(cropped, cam), caption=f"Where the model looks for '{pred_class}'", use_container_width=True)
    st.caption("Red = regions that increased the score of the predicted class. "
               "If it lights up the face or background instead of the hand, the crop is wrong or the model learned shortcuts.")