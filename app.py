import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests
import xml.etree.ElementTree as ET
from svgpathtools import parse_path
from streamlit_drawable_canvas import st_canvas

# --- 1. CONFIGURATION ---

KANJI_VG_SIZE = 109 
CANVAS_SIZE = 400

# --- 2. KANJIVG INTEGRATION ---

def fetch_kanji_strokes(char: str):
    """Fetches SVG data from KanjiVG and returns stroke coordinates."""
    if not char:
        return []
        
    code = f"{ord(char):05x}"
    url = f"https://raw.githubusercontent.com/KanjiVG/kanjivg/master/kanji/{code}.svg"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"Could not find data for '{char}'.")
            return []
            
        root = ET.fromstring(response.content)
        strokes = []
        
        for path_elem in root.iter():
            if path_elem.tag.endswith('path'):
                d_str = path_elem.get('d')
                if d_str:
                    path_obj = parse_path(d_str)
                    points = []
                    # Sample 20 points per stroke
                    for i in range(21):
                        complex_pt = path_obj.point(i / 20.0)
                        points.append((complex_pt.real, complex_pt.imag))
                    strokes.append(np.array(points))
        return strokes
    except Exception as e:
        st.error(f"Error parsing data: {e}")
        return []

# --- 3. HELPER FUNCTIONS ---

def resample_path(path_arr, num_points=20):
    if len(path_arr) < 2:
        return np.array([path_arr[0]] * num_points)
    
    dists = np.sqrt(np.sum(np.diff(path_arr, axis=0)**2, axis=1))
    cum_dists = np.insert(np.cumsum(dists), 0, 0)
    total_len = cum_dists[-1]
    
    if total_len == 0:
        return np.array([path_arr[0]] * num_points)

    new_dists = np.linspace(0, total_len, num_points)
    new_x = np.interp(new_dists, cum_dists, path_arr[:, 0])
    new_y = np.interp(new_dists, cum_dists, path_arr[:, 1])
    
    return np.column_stack((new_x, new_y))

def extract_user_strokes(canvas_result):
    if not canvas_result.json_data:
        return []
    
    objects = canvas_result.json_data["objects"]
    user_strokes = []
    scale_factor = KANJI_VG_SIZE / CANVAS_SIZE
    
    for obj in objects:
        if obj["type"] == "path":
            raw_path = obj["path"]
            points = []
            for cmd in raw_path:
                if len(cmd) >= 3:
                    points.append((cmd[-2], cmd[-1]))
            
            if points:
                norm_points = []
                for x, y in points:
                    norm_points.append((x * scale_factor, y * scale_factor))
                user_strokes.append(np.array(norm_points))
    return user_strokes

def grade_submission(user_strokes, ref_strokes):
    if len(user_strokes) != len(ref_strokes):
        return 0, f"Incorrect stroke count. Expected {len(ref_strokes)}, got {len(user_strokes)}.", None

    total_error = 0
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, KANJI_VG_SIZE)
    ax.set_ylim(KANJI_VG_SIZE, 0) 
    ax.axis('off')
    
    for i, (u_raw, r_pts) in enumerate(zip(user_strokes, ref_strokes)):
        u_pts = resample_path(u_raw)
        dist = np.mean(np.linalg.norm(u_pts - r_pts, axis=1))
        total_error += dist
        
        # Plot Reference (Green Dotted)
        ax.plot(r_pts[:,0], r_pts[:,1], 'g--', linewidth=3, alpha=0.5, label="Correct" if i==0 else "")
        # Plot User
        color = 'black' if dist < 12 else 'red'
        ax.plot(u_pts[:,0], u_pts[:,1], color=color, linewidth=3, label="You" if i==0 else "")

    avg_error = total_error / len(ref_strokes)
    score = max(0, 100 - (avg_error * 5))
    
    feedback = "Perfect!"
    if score < 85: feedback = "Good, but watch the details."
    if score < 50: feedback = "Try again. Follow the green guides."
    
    return int(score), feedback, fig

# --- 4. MAIN APP UI ---

def main():
    st.set_page_config(page_title="Kanji Memory Test", layout="centered")
    
    st.title("🧠 Kanji Memory Test")
    st.caption("Select a word and write it from memory.")

    # Input Selection
    col_sel, col_inp = st.columns([1, 1])
    with col_sel:
        practice_list = ["一", "二", "三", "川", "口", "木", "人", "永"]
        target_char = st.selectbox("Select Target to Practice:", practice_list)
    with col_inp:
        custom = st.text_input("Or type custom Kanji:", max_chars=1)
        if custom: target_char = custom

    # Fetch Data
    if 'current_kanji' not in st.session_state or st.session_state.current_kanji != target_char:
        st.session_state.current_kanji = target_char
        with st.spinner(f"Loading data..."):
            st.session_state.ref_strokes = fetch_kanji_strokes(target_char)

    ref_strokes = st.session_state.ref_strokes

    # --- CANVAS SECTION (Target Display Removed) ---
    st.markdown("### Write below:")
    
    # Canvas is now centered and the main focus
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=6,
        stroke_color="#000000",
        background_color="#ffffff",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Grade My Memory", type="primary"):
        if canvas_result.json_data:
            user_strokes = extract_user_strokes(canvas_result)
            
            if not user_strokes:
                st.warning("Canvas is empty.")
            elif not ref_strokes:
                st.error("Invalid Kanji selected.")
            else:
                score, comment, fig = grade_submission(user_strokes, ref_strokes)
                
                st.divider()
                st.subheader("Results")
                
                r_col1, r_col2 = st.columns([1, 2])
                with r_col1:
                    st.metric("Score", f"{score}/100")
                    if score > 80: st.balloons()
                with r_col2:
                    st.write(f"**Feedback:** {comment}")
                    if fig:
                        st.pyplot(fig)
                        st.caption("Green = Correct Shape | Red = Your Error")

if __name__ == "__main__":
    main()
