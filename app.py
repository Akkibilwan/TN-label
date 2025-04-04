import sqlite3
from datetime import datetime

# Connect or create DB
conn = sqlite3.connect("thumbnail_analysis.db")
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS thumbnail_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    video_id TEXT,
    source TEXT,
    openai_description TEXT,
    vision_result TEXT,
    final_analysis TEXT,
    final_prompt TEXT
);
""")
conn.commit()

# Save analysis to DB (insert this inside your processing logic after analysis is complete)
def save_analysis_to_db(video_id, source, openai_description, vision_result, final_analysis, final_prompt):
    timestamp = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO thumbnail_analysis (timestamp, video_id, source, openai_description, vision_result, final_analysis, final_prompt)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp,
        video_id,
        source,
        openai_description,
        json.dumps(vision_result) if vision_result else None,
        final_analysis,
        final_prompt
    ))
    conn.commit()

# Modify your main app where you have all data ready (after `final_analysis` and `prompt_paragraph`):
save_analysis_to_db(
    video_id=video_info.get("id", "uploaded_image"),
    source=input_option,
    openai_description=openai_description,
    vision_result=vision_results,
    final_analysis=analysis,
    final_prompt=prompt_paragraph
)

# -----------------------------
# Optional: Add a tab in the app to browse historical analyses
# -----------------------------
with st.expander("🧠 View Past Analyses"):
    cursor.execute("SELECT * FROM thumbnail_analysis ORDER BY timestamp DESC LIMIT 10")
    records = cursor.fetchall()
    for r in records:
        st.markdown(f"**Date**: {r[1]}")
        st.markdown(f"**Video ID / Source**: {r[2]} ({r[3]})")
        st.markdown(f"**Prompt**: {r[6][:200]}...")
        st.markdown(f"**OpenAI Description**: {r[4][:300]}...")
        st.markdown("---")

# This will give you a complete SQLite-backed, AI-powered, thumbnail analysis tool.
# Let me know if you want filtering/searching in the history tab as well!
