import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import List, Tuple

# --- 1. CONFIGURATION & DATA ---

@dataclass
class KanjiData:
    character: str
    meaning: str
    # Simplified reference strokes: List of [ (x, y), (x, y), ... ] coordinates
    # Coordinates are normalized to a 0-100 scale (top-left is 0,0)
    strokes: List[List[Tuple[float, float]]]

# Hardcoded reference data for demonstration
# (In a production app, you would load these from SVG files like KanjiVG)
KANJI_DB = {
    "One": KanjiData(
        "一", "One",
        [[(10, 50), (90, 50)]] # Horizontal line
    ),
    "Two": KanjiData(
        "二", "Two",
        [
            [(25, 30), (75, 30)], # Top short
            [(10, 70), (90, 70)]  # Bottom long
        ]
    ),
    "River": KanjiData(
        "川", "River",
        [
            [(20, 20), (15, 80)], # Left bent
            [(50, 20), (50, 70)], # Middle straight
            [(80, 20), (85, 80)]  # Right straight
        ]
    ),
    "Tree": KanjiData(
        "木", "Tree", 
        [
             [(15, 45), (85, 45)], # Horizontal
             [(50, 15), (50, 85)], # Vertical center
             [(50, 45), (20, 80)], # Left diag
             [(50, 45), (80, 80)], # Right diag
        ]
    )
}

# --- 2. HELPER FUNCTIONS ---

def resample_path(path, num_points=20):
    """Resamples a list of (x,y) points into a fixed number of equidistant points."""
    if len(path) < 2:
        return np.array(path * num_points)
    
    path = np.array(path)
    # Calculate cumulative distance along the path
    dists = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
    cum_dists = np.insert(np.cumsum(dists), 0, 0)
    total_len = cum_dists[-1]
    
    if total_len == 0:
        return np.array([path[0]] * num_points)

    # Interpolate
    new_dists = np.linspace(0, total_len, num_points)
    new_x = np.interp(new_dists, cum_dists, path[:, 0])
    new_y = np.interp(new_dists, cum_dists, path[:, 1])
    
    return np.column_stack((new_x, new_y))

def extract_user_strokes(canvas_result, canvas_height, canvas_width):
    """Parses the JSON output from streamlit-drawable-canvas."""
    if not canvas_result.json_data:
        return []
    
    objects = canvas_result.json_data["objects"]
    user_strokes = []
    
    for obj in objects:
        if obj["type"] == "path":
            # Fabric.js path format: [['M', x, y], ['Q', x1, y1, x2, y2], ...]
            # We assume a simple drawing mode where we can extract points
            raw_path = obj["path"]
            points = []
            for cmd in raw_path:
                # Extract the last 2 numbers as (x,y)
                # This is a simplification; robust SVG parsing is harder
                if len(cmd) >= 3:
                    points.append((cmd[-2], cmd[-1]))
            
            if points:
                # Normalize to 0-100 scale
                norm_points = []
                for x, y in points:
                    norm_points.append((
                        (x / canvas_width) * 100,
                        (y / canvas_height) * 100
                    ))
                user_strokes.append(norm_points)
                
    return user_strokes

def grade_submission(user_strokes, ref_strokes):
    """
    Grades the user strokes against reference strokes.
    Returns: Score (0-100), Feedback String, Matplotlib Figure
    """
    
    # 1. Stroke Count Check
    if len(user_strokes) != len(ref_strokes):
        return 0, f"Incorrect stroke count. Expected {len(ref_strokes)}, got {len(user_strokes)}.", None

    total_error = 0
    
    # 2. Geometry Check (Greedy matching or Index matching)
    # Assuming user writes in correct order for this simple demo
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, 100)
    ax.set_ylim(100, 0) # Flip Y for image coords
    ax.axis('off')
    
    for i, (u_raw, r_raw) in enumerate(zip(user_strokes, ref_strokes)):
        # Resample both to compare shape
        u_pts = resample_path(u_raw)
        r_pts = resample_path(r_raw)
        
        # Euclidean distance between corresponding points
        dist = np.mean(np.linalg.norm(u_pts - r_pts, axis=1))
        total_error += dist
        
        # Plotting for Feedback
        # Reference in Green (Dotted)
        ax.plot(r_pts[:,0], r_pts[:,1], 'g--', linewidth=3, alpha=0.6, label="Correct" if i==0 else "")
        # User in Black/Red depending on error
        color = 'black' if dist < 15 else 'red'
        ax.plot(u_pts[:,0], u_pts[:,1], color=color, linewidth=3, label="You" if i==0 else "")

    avg_error = total_error / len(ref_strokes)
    
    # Score formula (heuristic)
    # Error of 0 = 100 points. Error of 30+ = 0 points.
    score = max(0, 100 - (avg_error * 3))
    
    feedback = "Great job!"
    if score < 80: feedback = "Watch your stroke placement."
    if score < 50: feedback = "Try again! Follow the green guides."
    
    ax.legend()
    return int(score), feedback, fig

# --- 3. MAIN APP UI ---

def main():
    st.set_page_config(page_title="Kanji Grader", layout="centered")
    
    st.title("🇯🇵 Kanji Practice Dojo")
    st.markdown("Select a word, write it in the box, and get AI feedback on your stroke form.")

    # Sidebar Selection
    word_key = st.selectbox("Choose a word to practice:", list(KANJI_DB.keys()))
    target = KANJI_DB[word_key]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Target")
        # Display the target character large
        st.markdown(f"""
            <div style="font-size: 100px; text-align: center; border: 2px solid #ddd; border-radius: 10px; background-color: #f9f9f9;">
                {target.character}
            </div>
            <p style="text-align: center;">Meaning: {target.meaning}</p>
        """, unsafe_allow_html=True)
        
        # Toggle for "Cheat Mode" (Show stroke order hints)
        show_hint = st.checkbox("Show Stroke Guide (Cheat Mode)")
        if show_hint:
             st.info("Try to mimic the strokes shown in the grading result.")

    with col2:
        st.subheader("Your Canvas")
        
        # Canvas Configuration
        CANVAS_SIZE = 300
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=8,
            stroke_color="#000000",
            background_color="#ffffff",
            background_image=None,
            update_streamlit=True,
            height=CANVAS_SIZE,
            width=CANVAS_SIZE,
            drawing_mode="freedraw",
            key="canvas",
        )

    # Grading Section
    if st.button("Grade My Writing"):
        if canvas_result.json_data is not None:
            user_strokes = extract_user_strokes(canvas_result, CANVAS_SIZE, CANVAS_SIZE)
            
            if not user_strokes:
                st.warning("Please write something on the canvas first!")
            else:
                score, comment, fig = grade_submission(user_strokes, target.strokes)
                
                # Display Results
                st.divider()
                r_col1, r_col2 = st.columns([1, 2])
                
                with r_col1:
                    st.metric(label="Stroke Score", value=f"{score}/100")
                    if score > 80:
                        st.balloons()
                    st.write(f"**Feedback:** {comment}")
                
                with r_col2:
                    st.write("**Stroke Correction:**")
                    st.write("Green = Ideal Path | Black/Red = Your Path")
                    if fig:
                        st.pyplot(fig)
        else:
            st.error("Error reading canvas data.")

if __name__ == "__main__":
    main()
