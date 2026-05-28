"""
Prosthetic Hand Visualization Component
SVG-based gesture visualization with smooth animations.
"""

GESTURE_LABELS = {
    "0": {"thumb": 0, "fingers": 0, "wrist": 0, "label": "Rest"},
    "1": {"thumb": 45, "fingers": 45, "wrist": -15, "label": "Fist"},
    "2": {"thumb": -30, "fingers": 80, "wrist": 0, "label": "Open Palm"},
    "3": {"thumb": 20, "fingers": 30, "wrist": 15, "label": "Pinch"},
    "4": {"thumb": 10, "fingers": 60, "wrist": -20, "label": "Point"},
    "5": {"thumb": -20, "fingers": 25, "wrist": 25, "label": "Grasp"},
    "6": {"thumb": 40, "fingers": 70, "wrist": -10, "label": "Thumbs Up"},
    "7": {"thumb": -25, "fingers": 15, "wrist": -30, "label": "Wave"},
}

def normalize_gesture(gesture: str) -> str:
    """Extract gesture number from various formats."""
    gesture_str = str(gesture).lower().strip()
    gesture_str = gesture_str.replace("gesture ", "").replace("g", "").strip()
    try:
        gesture_num = int(float(gesture_str))
        if 0 <= gesture_num <= 7:
            return str(gesture_num)
    except (ValueError, TypeError):
        pass
    return "0"

def get_hand_html(gesture: str, confidence: float = 0.0) -> str:
    """
    Return HTML/SVG prosthetic hand visualization for a gesture.
    Gesture: "Gesture 0" through "Gesture 7" or variations.
    Confidence: 0.0 to 1.0 (will be displayed as %).
    Always uses real data — never hardcoded.
    """

    gesture_num = normalize_gesture(gesture)
    pos = GESTURE_LABELS[gesture_num]
    conf_pct = int(confidence * 100) if isinstance(confidence, float) else 0
    conf_pct = max(0, min(100, conf_pct))

    html = f"""
    <style>
        .hand-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            background: linear-gradient(135deg, rgba(13,148,136,0.08) 0%, rgba(129,140,248,0.06) 100%);
            border: 1px solid rgba(45,212,191,0.2);
            border-radius: 16px;
            margin: 1.5rem 0;
        }}

        .hand-viz {{
            position: relative;
            width: 200px;
            height: 280px;
            margin-bottom: 1rem;
        }}

        .hand-palm {{
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%) rotateZ({pos['wrist']}deg);
            width: 80px;
            height: 100px;
            background: linear-gradient(135deg, #2dd4bf 0%, #0d9488 100%);
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(45,212,191,0.3);
            transition: all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
        }}

        .hand-palm::before {{
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 100%);
            border-radius: 12px;
        }}

        .finger {{
            position: absolute;
            background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
            border-radius: 20px;
            transition: all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
            box-shadow: 0 4px 16px rgba(6,182,212,0.2);
        }}

        .finger-index {{
            width: 18px;
            height: 70px;
            left: 20px;
            top: -40px;
            transform: rotateZ({pos['fingers'] * 0.8}deg);
            transform-origin: center bottom;
        }}

        .finger-middle {{
            width: 18px;
            height: 80px;
            left: 41px;
            top: -50px;
            transform: rotateZ({pos['fingers']}deg);
            transform-origin: center bottom;
        }}

        .finger-ring {{
            width: 18px;
            height: 75px;
            right: 20px;
            top: -45px;
            transform: rotateZ({pos['fingers'] * 0.9}deg);
            transform-origin: center bottom;
        }}

        .finger-pinky {{
            width: 16px;
            height: 65px;
            right: -5px;
            top: -35px;
            transform: rotateZ({pos['fingers'] * 1.1}deg);
            transform-origin: center bottom;
        }}

        .thumb {{
            position: absolute;
            width: 16px;
            height: 60px;
            background: linear-gradient(135deg, #f97316 0%, #d97706 100%);
            border-radius: 20px;
            left: -20px;
            top: 20px;
            transform: rotateZ({pos['thumb']}deg);
            transform-origin: center bottom;
            transition: all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
            box-shadow: 0 4px 12px rgba(249,115,22,0.25);
        }}

        .gesture-label {{
            font-family: 'Syne', sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
            color: #2dd4bf;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 1rem;
            text-align: center;
        }}

        .confidence-bar {{
            width: 100%;
            height: 4px;
            background: rgba(255,255,255,0.1);
            border-radius: 2px;
            margin-top: 0.8rem;
            overflow: hidden;
        }}

        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, #06b6d4 0%, #2dd4bf 100%);
            width: {conf_pct}%;
            transition: width 0.4s ease;
            box-shadow: 0 0 12px rgba(45,212,191,0.5);
        }}

        .confidence-text {{
            font-family: 'DM Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.4rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
    </style>

    <div class="hand-container">
        <div class="hand-viz">
            <div class="hand-palm">
                <div class="finger finger-index"></div>
                <div class="finger finger-middle"></div>
                <div class="finger finger-ring"></div>
                <div class="finger finger-pinky"></div>
                <div class="thumb"></div>
            </div>
        </div>
        <div class="gesture-label">🦾 {pos['label']}</div>
        <div class="confidence-bar">
            <div class="confidence-fill"></div>
        </div>
        <div class="confidence-text">Confidence: {conf_pct}%</div>
    </div>
    """

    return html

def get_hand_component(gesture: str, confidence: float = 0.0):
    """Render hand visualization in Streamlit."""
    import streamlit as st
    html = get_hand_html(gesture, confidence)
    st.markdown(html, unsafe_allow_html=True)
