# Dashboard constants, color palette, and indicator mappings

CURRENT_SEASON  = 2025
DATA_CF_FILE_ID = "1G8eA6gX8hdCwWp1YWddAMmwt6R62tcmA"

PA_INDICATORS = [
    "hitting_decisions_score",
    "power_efficiency",
    "woba_residual",
    "launch_angle_stability_50pa",
]
PA_LABELS = {
    "hitting_decisions_score":     "Hitting Decisions",
    "power_efficiency":            "Power Efficiency",
    "woba_residual":               "wOBA Residual",
    "launch_angle_stability_50pa": "Launch Angle Stability",
}
PA_TOOLTIPS = {
    "hitting_decisions_score":     "Plate discipline. Measures swing vs. take quality. Higher is better (Elite: >3.0, League Avg: ~0.3).",
    "power_efficiency":            "Raw power. Effectiveness of converting swing effort to exit velocity. Higher is better (Elite: >0.0100, League Avg: ~0.0040).",
    "woba_residual":               "Luck vs Skill. Difference between actual results and physics-based expectation. Positive (>0.15) means outperforming physics (luck/skill); Negative (<-0.15) means 'unlucky'.",
    "launch_angle_stability_50pa": "Swing consistency. Stability of ball flight path over recent 50 PAs. Higher values indicate a more repeatable, optimized swing path.",
}
PA_COLORS = {
    "hitting_decisions_score":     "#2ca02c",
    "power_efficiency":            "#1f77b4",
    "woba_residual":               "#ff7f0e",
    "launch_angle_stability_50pa": "#d62728",
}

SENSITIVITY_MAP        = {"Low": 8, "Medium": 3, "High": 1}
SENSITIVITY_TO_MIN_SEG = {"Low": 0.10, "Medium": 0.05, "High": 0.02}

# Color palette
DARK       = "#FFFFFF"
PANEL      = "#F0F2F6"
BORDER     = "#D0D7DE"
GOLD       = "#855D00"
GOLD_LT    = "#B08800"
TEAL       = "#0068C9"
TEAL_LT    = "#58A6FF"
RED        = "#D32F2F"
RED_LT     = "#B71C1C"
GREY       = "#586069"
TEXT       = "#111418"
TEXT_MUTED = "#586069"
