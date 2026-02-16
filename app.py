import streamlit as st
import numpy as np
import requests
import xml.etree.ElementTree as ET
from svgpathtools import parse_path
from streamlit_drawable_canvas import st_canvas
from scipy.spatial.distance import cdist

# --- 1. CONFIGURATION ---

KANJI_VG_SIZE = 109 
CANVAS_SIZE = 400
SNAP_THRESHOLD = 30  # How lenient the snapping is (Higher = easier)

# --- 2. KANJIVG INTEGRATION ---

@st.cache_data
def fetch_kanji_data(char: str):
    """Fetches reference strokes and raw SVG paths."""
    if not char: return [], []
    code = f"{ord(char):05x}"
    url = f"https://raw.githubusercontent.com/KanjiVG/kanjivg/master/kanji/{code}.svg"
    try:
        r = requests.get(url)
        if r.status_code != 200: return [], []
        
        root = ET.fromstring(r.content)
        strokes_resampled = []
        strokes_svg_str = [] # We keep the raw SVG string for rendering the "perfect" line
        
        for path_elem in root.iter():
            if path_elem.tag.endswith('path'):
                d = path_elem.get('d')
                strokes_svg_str.append(d)
                # Mathematical resampling for grading
                p_obj = parse_path(d)
                pts = [(p_obj.point(i/20).real, p_obj.point(i/20).imag) for i in range(21)]
                strokes_resampled.append(np.array(pts))
        return strokes_resampled, strokes_svg_str
    except:
        return [], []

# --- 3. MATH & CONVERSION ---

def resample_path(path_arr, num_points=20):
    if len(path_arr) < 2: return np.array([path_arr[0]]*num_points)
    dists = np.sqrt(np.sum(np.diff(path_arr, axis=0)**2, axis=1))
    cum_dists = np.insert(np.cumsum(dists), 0, 0)
    if cum_dists[-1] == 0: return np.array([path_arr[0]]*num_points)
    new_dists = np.linspace(0, cum_dists[-1], num_points)
    return np.column_stack((np.interp(new_dists, cum_dists, path_arr[:,0]), 
                            np.interp(new_dists, cum_dists, path_arr[:,1])))

def svg_to_fabric_json(svg_path_str, color="black"):
    """
    Converts a raw SVG path string into a Fabric.js object dict 
    so st_canvas can render the 'snapped' stroke.
    """
    scale = CANVAS_SIZE / KANJI_VG_SIZE
    return {
        "type": "path",
        "originX": "left", "originY": "top",
        "left": 0, "top": 0,
        "fill": None,
        "stroke": color,
        "strokeWidth": 4,
        "path": parse_path(svg_path_str).d(), # Ensures clean string
        "scaleX": scale,
        "scaleY": scale,
        "selectable": False # User cannot move the snapped lines
    }

def extract_last_stroke(canvas_result):
    """Gets the most recent stroke drawn by the user."""
    if not canvas_result.json_data: return None
    objs = canvas_result.json_data["objects"]
    if not objs: return None
    
    # Get last object
    last_obj = objs[-1]
    path_cmds = last_obj.get("path", [])
    points = []
    for cmd in path_cmds:
        if len(cmd) >= 3: points.append((cmd[-2], cmd[-1]))
    
    if not points: return None
    
    # Normalize to 109 scale
    scale = KANJI_VG_SIZE / CANVAS_SIZE
    return np.array([(x*scale, y*scale) for x,y in points])

# --- 4. MAIN APP ---

def main():
    st.title("⚡ Magic Kanji Snapping")
    st.caption("Draw the stroke. If it's correct, it will 'snap' to the perfect shape.")

    # A. Session State Setup
    if "step" not in st.session_state: st.session_state.step = 0
    if "locked_objects" not in st.session_state: st.session_state.locked_objects = []
    if "canvas_key" not in st.session_state: st.session_state.canvas_key = 0

    # B. Input
    char = st.text_input("Kanji", value="木", max_chars=1)
    ref_strokes, ref_svgs = fetch_kanji_data(char)

    if not ref_strokes:
        st.error("Kanji not found.")
        return

    # C. Logic: Did user just draw a stroke?
    # We use a placeholder to render canvas, then check logic below it
    canvas_container = st.empty()
    
    # Prepare the initial drawing (The strokes we have already snapped)
    initial_drawing = {
        "version": "4.4.0",
        "objects": st.session_state.locked_objects
    }

    with canvas_container:
        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=5,
            stroke_color="black",
            background_color="#fff",
            initial_drawing=initial_drawing,
            update_streamlit=True,
            height=CANVAS_SIZE, 
            width=CANVAS_SIZE,
            drawing_mode="freedraw",
            # We increment key to force a re-render when we snap a line
            key=f"canvas_{st.session_state.canvas_key}" 
        )

    # D. Grading Logic
    if canvas_result.json_data:
        objects = canvas_result.json_data["objects"]
        
        # If there are more objects on screen than we have locked, the user just drew one
        if len(objects) > len(st.session_state.locked_objects):
            
            # 1. Get the user's new stroke
            user_stroke_raw = extract_last_stroke(canvas_result)
            
            if user_stroke_raw is not None and st.session_state.step < len(ref_strokes):
                
                # 2. Compare against the CURRENT target stroke
                current_target_idx = st.session_state.step
                target_stroke = ref_strokes[current_target_idx]
                
                # Math: Resample and Distance
                u_res = resample_path(user_stroke_raw)
                dist = np.mean(np.linalg.norm(u_res - target_stroke, axis=1))
                
                # 3. Decision
                if dist < SNAP_THRESHOLD:
                    # SUCCESS: Snap to grid
                    st.toast(f"Stroke {current_target_idx + 1} Snapped! ✅")
                    
                    # Convert the Perfect Reference Stroke to Fabric JSON
                    perfect_stroke_json = svg_to_fabric_json(ref_svgs[current_target_idx], color="#00aa00")
                    
                    # Add to locked list
                    st.session_state.locked_objects.append(perfect_stroke_json)
                    st.session_state.step += 1
                    
                    # Force canvas reload to remove user's messy line and show the clean one
                    st.session_state.canvas_key += 1
                    st.rerun()
                    
                else:
                    # FAILURE: Wrong shape
                    st.toast(f"Too messy! Try stroke {current_target_idx + 1} again. ❌")
                    # Ideally, we undo their last stroke here.
                    # In this simple version, we just let them hit undo on the toolbar or ignore it.

    # E. Progress Display
    st.progress(st.session_state.step / len(ref_strokes))
    if st.session_state.step == len(ref_strokes):
        st.balloons()
        st.success("Kanji Complete!")
        if st.button("Reset"):
            st.session_state.step = 0
            st.session_state.locked_objects = []
            st.session_state.canvas_key += 1
            st.rerun()

if __name__ == "__main__":
    main()
