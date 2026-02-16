import streamlit as st
import numpy as np
import requests
import xml.etree.ElementTree as ET
from svgpathtools import parse_path
from streamlit_drawable_canvas import st_canvas

# --- 1. CONFIGURATION ---

KANJI_VG_SIZE = 109 
CANVAS_SIZE = 400
SNAP_THRESHOLD = 45  # Increased slightly to make it easier to hit
NUM_POINTS = 20      # CRITICAL: Both user and reference must match this exactly

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
        strokes_svg_str = [] 
        
        for path_elem in root.iter():
            if path_elem.tag.endswith('path'):
                d = path_elem.get('d')
                strokes_svg_str.append(d)
                
                # Mathematical resampling
                p_obj = parse_path(d)
                
                # FIX: We strictly generate NUM_POINTS (20)
                # We divide by (NUM_POINTS - 1) to ensure we get t=0.0 to t=1.0 inclusive
                pts = []
                for i in range(NUM_POINTS):
                    t = i / (NUM_POINTS - 1)
                    complex_pt = p_obj.point(t)
                    pts.append((complex_pt.real, complex_pt.imag))
                    
                strokes_resampled.append(np.array(pts))
                
        return strokes_resampled, strokes_svg_str
    except Exception as e:
        st.error(f"Error parsing Kanji data: {e}")
        return [], []

# --- 3. MATH & CONVERSION ---

def resample_path(path_arr, num_points=NUM_POINTS):
    """Resamples user input to have exactly NUM_POINTS."""
    if len(path_arr) < 2: 
        return np.array([path_arr[0]] * num_points)
    
    # Calculate cumulative distance along the path
    dists = np.sqrt(np.sum(np.diff(path_arr, axis=0)**2, axis=1))
    cum_dists = np.insert(np.cumsum(dists), 0, 0)
    total_len = cum_dists[-1]
    
    if total_len == 0:
        return np.array([path_arr[0]] * num_points)

    # Interpolate to find exactly num_points equidistant spots
    new_dists = np.linspace(0, total_len, num_points)
    new_x = np.interp(new_dists, cum_dists, path_arr[:, 0])
    new_y = np.interp(new_dists, cum_dists, path_arr[:, 1])
    
    return np.column_stack((new_x, new_y))

def svg_to_fabric_json(svg_path_str, color="black"):
    """Converts SVG path to Fabric.js JSON for the canvas background."""
    scale = CANVAS_SIZE / KANJI_VG_SIZE
    return {
        "type": "path",
        "originX": "left", 
        "originY": "top",
        "left": 0, 
        "top": 0,
        "fill": None,
        "stroke": color,
        "strokeWidth": 5,
        "strokeLineCap": "round",
        "strokeLineJoin": "round",
        "path": parse_path(svg_path_str).d(),
        "scaleX": scale,
        "scaleY": scale,
        "selectable": False
    }

def extract_last_stroke(canvas_result):
    """Gets the most recent stroke drawn by the user and scales it."""
    if not canvas_result.json_data: return None
    objs = canvas_result.json_data["objects"]
    if not objs: return None
    
    # Get last object
    last_obj = objs[-1]
    path_cmds = last_obj.get("path", [])
    points = []
    
    # Extract points from SVG commands
    for cmd in path_cmds:
        if len(cmd) >= 3: 
            points.append((cmd[-2], cmd[-1]))
    
    if not points: return None
    
    # Scale from Canvas Size (400) down to KanjiVG Size (109) for comparison
    scale = KANJI_VG_SIZE / CANVAS_SIZE
    return np.array([(x*scale, y*scale) for x,y in points])

# --- 4. MAIN APP ---

def main():
    st.set_page_config(page_title="Kanji Snapper")
    st.title("⚡ Magic Kanji Snapping")
    st.caption("Draw the stroke. If it's correct, it will 'snap' to the perfect shape.")

    # A. Session State Setup
    if "step" not in st.session_state: st.session_state.step = 0
    if "locked_objects" not in st.session_state: st.session_state.locked_objects = []
    if "canvas_key" not in st.session_state: st.session_state.canvas_key = 0

    # B. Input
    col1, col2 = st.columns([3, 1])
    with col1:
        char = st.text_input("Kanji to Practice", value="木", max_chars=1)
    with col2:
        if st.button("Reset"):
            st.session_state.step = 0
            st.session_state.locked_objects = []
            st.session_state.canvas_key += 1
            st.rerun()

    ref_strokes, ref_svgs = fetch_kanji_data(char)

    if not ref_strokes:
        st.error("Kanji not found.")
        return

    # Progress Bar
    progress = st.session_state.step / len(ref_strokes)
    st.progress(progress)

    # C. Logic: Canvas Rendering
    # We use a placeholder to render canvas
    canvas_container = st.container()
    
    # Prepare the initial drawing (The strokes we have already snapped)
    initial_drawing = {
        "version": "4.4.0",
        "objects": st.session_state.locked_objects
    }

    with canvas_container:
        # We use a unique key to force the canvas to "clear" the user's messy stroke
        # when we successfully snap a new one.
        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=5,
            stroke_color="black",
            background_color="#ffffff",
            initial_drawing=initial_drawing,
            update_streamlit=True,
            height=CANVAS_SIZE, 
            width=CANVAS_SIZE,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}" 
        )

    # D. Grading Logic
    if canvas_result.json_data:
        objects = canvas_result.json_data["objects"]
        
        # Check if user added a new stroke (objects count > locked count)
        if len(objects) > len(st.session_state.locked_objects):
            
            # 1. Get the user's new stroke
            user_stroke_raw = extract_last_stroke(canvas_result)
            
            if user_stroke_raw is not None and st.session_state.step < len(ref_strokes):
                
                # 2. Compare against the CURRENT target stroke
                current_target_idx = st.session_state.step
                target_stroke = ref_strokes[current_target_idx]
                
                # Math: Resample to NUM_POINTS and calculate Distance
                u_res = resample_path(user_stroke_raw)
                
                # Debugging info (optional, remove in prod)
                # st.write(f"User points: {u_res.shape}, Target points: {target_stroke.shape}")
                
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
                    st.toast(f"Too messy! (Error: {dist:.1f}) ❌")
                    
    # E. Completion
    if st.session_state.step == len(ref_strokes) and len(ref_strokes) > 0:
        st.balloons()
        st.success("Kanji Complete! Press Reset to try again.")

if __name__ == "__main__":
    main()
