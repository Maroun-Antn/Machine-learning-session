# lesson_1_app_m.py — Lesson 1 (Military Edition): What Is Machine Learning?
# Run with:  streamlit run lesson_1_app_m.py
# Install:   pip install streamlit plotly pandas scikit-learn numpy

import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="What Is Machine Learning?",
    page_icon="🪖",
    layout="wide",
)

# ── Colour palette ─────────────────────────────────────────────────────────────
BLUE   = "#2196F3"
ORANGE = "#FF9800"
GREEN  = "#4CAF50"
RED    = "#F44336"

# ── Data generation ────────────────────────────────────────────────────────────
@st.cache_data
def generate_data() -> pd.DataFrame:
    np.random.seed(7)
    vehicle_id     = np.arange(1, 31)
    vehicle_age    = np.random.randint(1, 13, 30)
    operating_hrs  = np.round(np.random.uniform(200, 4800, 30) / 10) * 10
    vehicle_type   = (np.random.rand(30) < 0.35).astype(int)  # 1 = Type-A (heavy), 0 = Type-B (light)

    noise_last = np.random.normal(0, 1.0, 30)
    last_pm = np.round(4.0 + 0.0012 * operating_hrs + 2.0 * vehicle_type + noise_last, 1)
    last_pm = np.clip(last_pm, 3, 15)

    noise_cur = np.random.normal(0, 1.0, 30)
    current_pm = np.round(4.0 + 0.0015 * operating_hrs + 2.5 * vehicle_type + noise_cur, 1)
    current_pm = np.clip(current_pm, 3, 16)

    return pd.DataFrame({
        "vehicle_id":            vehicle_id,
        "vehicle_age_years":     vehicle_age,
        "operating_hours":       operating_hrs,
        "operating_hours_k":     np.round(operating_hrs / 1000, 2),
        "vehicle_type":          vehicle_type,
        "last_pm_duration_hrs":  last_pm,
        "current_pm_duration_hrs": current_pm,
    })


@st.cache_data
def fit_linear_regression():
    """Return (m1, m2, c, predictions, mae) for the full 30-vehicle dataset."""
    df = generate_data()
    X = df[["operating_hours_k", "vehicle_type"]].values
    y = df["current_pm_duration_hrs"].values
    lr = LinearRegression().fit(X, y)
    preds = lr.predict(X)
    return (
        float(lr.coef_[0]),
        float(lr.coef_[1]),
        float(lr.intercept_),
        preds,
        float(mean_absolute_error(y, preds)),
    )


@st.cache_data
def fit_random_forest(n_rows: int = 30):
    """
    Train a RandomForest on the first n_rows vehicles.
    Returns:
        importances  – list[float], length 2
        preds_all    – ndarray of predictions on all 30 vehicles
        mae_train    – float, MAE on training subset
        z_heatmap    – ndarray, grid of predicted PM duration
        oh_grid      – ndarray, operating-hours axis for heatmap
        type_grid    – ndarray, vehicle-type axis for heatmap
        heat_preds   – dict {delta: list}, predictions for +delta*200 extra hours
    """
    df = generate_data()
    subset = df.iloc[:n_rows]
    X_tr = subset[["operating_hours_k", "vehicle_type"]].values
    y_tr = subset["current_pm_duration_hrs"].values

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)

    importances = rf.feature_importances_.tolist()
    preds_all   = rf.predict(df[["operating_hours_k", "vehicle_type"]].values)
    mae_train   = float(mean_absolute_error(y_tr, rf.predict(X_tr)))

    oh_grid   = np.linspace(0.2, 4.8, 80)
    type_grid = np.linspace(0, 1, 50)
    tt, ww    = np.meshgrid(oh_grid, type_grid)
    z_heatmap = rf.predict(
        np.column_stack([tt.ravel(), ww.ravel()])
    ).reshape(tt.shape)

    heat_preds = {}
    oh_base   = df["operating_hours_k"].values
    types_all = df["vehicle_type"].values
    for delta in range(11):
        X_heat = np.column_stack([oh_base + delta * 0.2, types_all])
        heat_preds[delta] = rf.predict(X_heat).tolist()

    return importances, preds_all, mae_train, z_heatmap, oh_grid, type_grid, heat_preds


def officer_rules_vec(
    operating_hrs: np.ndarray,
    vehicle_type: np.ndarray,
    high_heavy: float = 12.0,
    high_light: float = 8.0,
    low_heavy: float = 8.0,
    low_light: float = 5.0,
) -> np.ndarray:
    base = np.where(operating_hrs > 2500,
                    np.where(vehicle_type == 1, high_heavy, high_light),
                    np.where(vehicle_type == 1, low_heavy, low_light))
    return base.astype(float)


def _build_dt1_fig(overall_avg: float) -> go.Figure:
    """Decision tree: single leaf — predict overall average for every vehicle."""
    fig = go.Figure()
    W, H = 12.0, 6.0
    BW, BH = 3.2, 1.2
    fig.update_layout(
        width=800, height=280,
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(visible=False, range=[0, W]),
        yaxis=dict(visible=False, range=[0, H]),
        margin=dict(l=10, r=10, t=50, b=10),
        title=dict(text="Decision Tree 1 — One Rule: Always Predict the Fleet Average",
                   font=dict(family="Arial", size=15)),
        showlegend=False,
    )
    cx, cy, color = 6.0, 4.0, "#1565C0"
    fig.add_shape(type="rect",
        x0=cx - BW/2 + 0.05, y0=cy - BH/2 - 0.05,
        x1=cx + BW/2 + 0.05, y1=cy + BH/2 - 0.05,
        fillcolor="rgba(0,0,0,0.12)", line_color="rgba(0,0,0,0)")
    fig.add_shape(type="rect",
        x0=cx - BW/2, y0=cy - BH/2,
        x1=cx + BW/2, y1=cy + BH/2,
        fillcolor=color, line_color="white", line_width=1.5)
    fig.add_annotation(x=cx, y=cy,
        text="<b>Any vehicle?</b>", showarrow=False, align="center",
        font=dict(color="white", size=13, family="Arial"))
    lx, ly, lcolor = 6.0, 1.8, "#2E7D32"
    fig.add_shape(type="line",
        x0=cx, y0=cy - BH/2, x1=lx, y1=ly + BH/2,
        line=dict(color="#78909C", width=2))
    fig.add_annotation(x=(cx+lx)/2, y=(cy - BH/2 + ly + BH/2)/2,
        text="<b>YES — always</b>", showarrow=False,
        bgcolor="rgba(255,255,255,0.88)", bordercolor="#90A4AE",
        borderwidth=1, borderpad=3, font=dict(size=10, color="#263238", family="Arial"))
    fig.add_shape(type="rect",
        x0=lx - BW/2 + 0.05, y0=ly - BH/2 - 0.05,
        x1=lx + BW/2 + 0.05, y1=ly + BH/2 - 0.05,
        fillcolor="rgba(0,0,0,0.12)", line_color="rgba(0,0,0,0)")
    fig.add_shape(type="rect",
        x0=lx - BW/2, y0=ly - BH/2,
        x1=lx + BW/2, y1=ly + BH/2,
        fillcolor=lcolor, line_color="white", line_width=1.5)
    fig.add_annotation(x=lx, y=ly,
        text=f"<b style='font-size:14px'>{overall_avg:.1f} hrs</b> 🔧<br>Predict fleet average",
        showarrow=False, align="center",
        font=dict(color="white", size=11, family="Arial"))
    return fig


def _build_dt2_fig(light_avg: float, heavy_avg: float) -> go.Figure:
    """Decision tree: split on vehicle type only."""
    fig = go.Figure()
    W, H = 12.0, 8.0
    BW, BH = 2.8, 1.1
    fig.update_layout(
        width=800, height=340,
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(visible=False, range=[0, W]),
        yaxis=dict(visible=False, range=[0, H]),
        margin=dict(l=10, r=10, t=50, b=10),
        title=dict(text="Decision Tree 2 — One Split: Heavy (Type-A) or Light (Type-B)?",
                   font=dict(family="Arial", size=15)),
        showlegend=False,
    )
    nodes = {
        "root": (6.0, 6.5, "#1565C0", "<b>Is it a Type-A (Heavy) vehicle?</b>"),
        "heavy": (3.0, 3.2, "#E65100",
                  f"<b style='font-size:14px'>{heavy_avg:.1f} hrs</b> 🔧🔧<br>Heavy avg"),
        "light": (9.0, 3.2, "#455A64",
                  f"<b style='font-size:14px'>{light_avg:.1f} hrs</b> 🔧<br>Light avg"),
    }
    edges = [("root", "heavy", "YES ✓"), ("root", "light", "NO ✗")]
    for src, dst, label in edges:
        sx, sy = nodes[src][:2];  dx, dy = nodes[dst][:2]
        fig.add_shape(type="line",
            x0=sx, y0=sy - BH/2, x1=dx, y1=dy + BH/2,
            line=dict(color="#78909C", width=2))
        mx, my = (sx+dx)/2, (sy - BH/2 + dy + BH/2)/2
        fig.add_annotation(x=mx, y=my, text=f"<b>{label}</b>", showarrow=False,
            bgcolor="rgba(255,255,255,0.88)", bordercolor="#90A4AE",
            borderwidth=1, borderpad=3, font=dict(size=10, color="#263238", family="Arial"))
    for cx, cy, color, html in nodes.values():
        fig.add_shape(type="rect",
            x0=cx - BW/2 + 0.05, y0=cy - BH/2 - 0.05,
            x1=cx + BW/2 + 0.05, y1=cy + BH/2 - 0.05,
            fillcolor="rgba(0,0,0,0.12)", line_color="rgba(0,0,0,0)")
        fig.add_shape(type="rect",
            x0=cx - BW/2, y0=cy - BH/2,
            x1=cx + BW/2, y1=cy + BH/2,
            fillcolor=color, line_color="white", line_width=1.5)
        fig.add_annotation(x=cx, y=cy, text=html,
            showarrow=False, align="center",
            font=dict(color="white", size=11, family="Arial"))
    return fig


def _build_decision_tree_fig(
    high_heavy: float = 12.0,
    high_light: float = 8.0,
    low_heavy: float = 8.0,
    low_light: float = 5.0,
) -> go.Figure:
    """Return a Plotly figure showing Officer Samer's hand-crafted decision tree."""
    fig = go.Figure()

    W, H   = 12.0, 10.0
    BW, BH = 2.4, 1.1

    fig.update_layout(
        width=800, height=430,
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(visible=False, range=[0, W]),
        yaxis=dict(visible=False, range=[0, H]),
        margin=dict(l=10, r=10, t=50, b=10),
        title=dict(
            text="Officer Samer's Decision Tree",
            font=dict(family="Arial", size=15),
        ),
        showlegend=False,
    )

    nodes = {
        "root": (6.0,  8.5, "#1565C0",
                 "⏱️  <b>Operating Hours &gt; 2,500?</b>"),
        "hot":  (3.0,  5.5, "#BF360C",
                 "⚠️  <b>High-Wear</b><br><i>Type-A (Heavy)?</i>"),
        "cold": (9.0,  5.5, "#0277BD",
                 "✅  <b>Low-Wear</b><br><i>Type-A (Heavy)?</i>"),
        "hw":   (1.2,  2.2, "#B71C1C",
                 f"<b style='font-size:14px'>{high_heavy:.1f} hrs</b> 🔧🔧🔧<br>High-Wear + Heavy"),
        "hwd":  (4.8,  2.2, "#E65100",
                 f"<b style='font-size:14px'>{high_light:.1f} hrs</b> 🔧🔧<br>High-Wear + Light"),
        "cw":   (7.2,  2.2, "#1565C0",
                 f"<b style='font-size:14px'>{low_heavy:.1f} hrs</b> 🔧🔧<br>Low-Wear + Heavy"),
        "cwd":  (10.8, 2.2, "#455A64",
                 f"<b style='font-size:14px'>{low_light:.1f} hrs</b> 🔧<br>Low-Wear + Light"),
    }

    edges = [
        ("root", "hot",  "YES ✓"),
        ("root", "cold", "NO ✗"),
        ("hot",  "hw",   "Heavy"),
        ("hot",  "hwd",  "Light"),
        ("cold", "cw",   "Heavy"),
        ("cold", "cwd",  "Light"),
    ]

    for src, dst, label in edges:
        sx, sy = nodes[src][:2]
        dx, dy = nodes[dst][:2]
        fig.add_shape(
            type="line",
            x0=sx, y0=sy - BH / 2,
            x1=dx, y1=dy + BH / 2,
            line=dict(color="#78909C", width=2),
        )
        mx = (sx + dx) / 2
        my = (sy - BH / 2 + dy + BH / 2) / 2
        fig.add_annotation(
            x=mx, y=my,
            text=f"<b>{label}</b>",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#90A4AE", borderwidth=1, borderpad=3,
            font=dict(size=10, color="#263238", family="Arial"),
        )

    for cx, cy, color, html in nodes.values():
        fig.add_shape(
            type="rect",
            x0=cx - BW / 2 + 0.05, y0=cy - BH / 2 - 0.05,
            x1=cx + BW / 2 + 0.05, y1=cy + BH / 2 - 0.05,
            fillcolor="rgba(0,0,0,0.12)",
            line_color="rgba(0,0,0,0)",
        )
        fig.add_shape(
            type="rect",
            x0=cx - BW / 2, y0=cy - BH / 2,
            x1=cx + BW / 2, y1=cy + BH / 2,
            fillcolor=color,
            line_color="white", line_width=1.5,
        )
        fig.add_annotation(
            x=cx, y=cy,
            text=html,
            showarrow=False, align="center",
            font=dict(color="white", size=11, family="Arial"),
        )

    return fig


# ── Sidebar navigation ─────────────────────────────────────────────────────────
SECTIONS = [
    "📖 The Story",
    "📊 The Data",
    "🧑 The Human Way",
    "📐 The Formula Way",
    "🤖 The ML Way",
    "🧠 ML Defined",
]


# ══════════════════════════════════════════════════════════════════════════════
# Section renderers
# ══════════════════════════════════════════════════════════════════════════════

def render_story():
    st.title("🪖 The Story")
    st.subheader("A story about a vehicle fleet, experience, and pattern-finding machines")

    st.markdown("""
Imagine a fleet of **30 vehicles**. Preventive maintenance (**PM-500**) is mandatory every
500 operating hours. The maintenance officer already knows *which* vehicles need maintenance —
that part is a fixed schedule. The real question is:

**How long will each vehicle occupy the maintenance workshop?**

Historically, the unit has used a fixed estimate:
> *"A PM-500 service normally takes 6 hours."*

But actual maintenance history tells a messier story — some vehicles are in and out in 4 hours,
others tie up the bay for 14.

Now picture the maintenance officer facing this question tomorrow morning:

> *"A six-year-old Type-A vehicle with 4,200 operating hours is entering PM tomorrow.*
> *How many hours should we reserve the maintenance bay?"*

Three people will try to answer this question, each a different way:
- **🧑 Officer Samer** — uses rules of thumb built from experience
- **📐 Lieutenant Samira** — writes a formula based on the numbers
- **🤖 Captain Fares** — lets a machine learn the pattern from history
    """)

    st.markdown("---")
    st.markdown(
        "➡️ Use the **sidebar** on the left to navigate. "
        "Start with **📊 The Data** to explore the maintenance log."
    )


# ──────────────────────────────────────────────────────────────────────────────

def render_data():
    df = generate_data()
    st.title("📊 The Maintenance Log")
    st.markdown(
        "You are looking at the raw data recorded over the last 30 PM jobs. "
        "Each row is one vehicle entering maintenance. The last column — "
        "**`current_pm_duration_hrs`** — is what all three approaches will try to **predict**."
    )

    st.markdown("#### First 10 maintenance jobs")

    def style_data_row(row):
        out = []
        for col in row.index:
            if col == "current_pm_duration_hrs":
                out.append("background-color: #FFE0B2; font-weight: bold")
            elif row.name % 2 == 0:
                out.append("background-color: #E3F2FD")
            else:
                out.append("background-color: white")
        return out

    display_df = df.copy()
    display_df["vehicle_type"] = display_df["vehicle_type"].map({0: "Type-B (Light)", 1: "Type-A (Heavy)"})
    display_cols = [
        "vehicle_id", "vehicle_age_years", "operating_hours", "vehicle_type",
        "last_pm_duration_hrs", "current_pm_duration_hrs",
    ]
    styled = (
        display_df[display_cols].head(10)
        .style
        .apply(style_data_row, axis=1)
        .format({
            "operating_hours": "{:.0f}",
            "last_pm_duration_hrs": "{:.1f}",
            "current_pm_duration_hrs": "{:.1f}",
        })
    )
    st.caption("*The maintenance log — first 10 jobs*")
    st.dataframe(styled, use_container_width=True)

    st.markdown("#### Summary statistics")
    st.dataframe(
        df[[
            "vehicle_age_years", "operating_hours",
            "last_pm_duration_hrs", "current_pm_duration_hrs",
        ]].describe().round(2),
        use_container_width=True,
    )

    st.markdown("#### Quick look: operating hours vs. PM duration")
    fig = px.scatter(
        df,
        x="operating_hours", y="current_pm_duration_hrs",
        color=df["vehicle_type"].map({0: "Type-B (Light)", 1: "Type-A (Heavy)"}),
        symbol=df["vehicle_type"].map({0: "Type-B (Light)", 1: "Type-A (Heavy)"}),
        color_discrete_map={"Type-B (Light)": BLUE, "Type-A (Heavy)": ORANGE},
        template="plotly_white",
        labels={
            "operating_hours": "Operating Hours",
            "current_pm_duration_hrs": "PM Duration (hours)",
            "color": "Vehicle Type", "symbol": "Vehicle Type",
        },
        title="All 30 Jobs: Operating Hours vs. PM Duration",
    )
    fig.update_layout(
        width=800, height=450,
        font=dict(family="Arial", size=13), title_font_size=15,
    )
    # alt: Scatter plot showing a positive correlation between operating hours and PM duration; heavy Type-A vehicles (orange squares) sit above light Type-B vehicles (blue circles) at the same operating hours
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Each dot is one vehicle entering PM. Orange squares = Type-A (heavy). "
        "Blue circles = Type-B (light). "
        "Vehicle age and last PM duration are also tracked, but the models below "
        "focus on the two strongest drivers: operating hours and vehicle type."
    )

# ──────────────────────────────────────────────────────────────────────────────

def render_human_way():
    df      = generate_data()
    vids    = df["vehicle_id"].values
    ohrs    = df["operating_hours"].values
    vtypes  = df["vehicle_type"].values
    actual  = df["current_pm_duration_hrs"].values

    # ── Pre-compute all group averages ──────────────────────────────────────
    overall_avg = float(df["current_pm_duration_hrs"].mean())
    light_avg   = float(df.loc[df["vehicle_type"] == 0, "current_pm_duration_hrs"].mean())
    heavy_avg   = float(df.loc[df["vehicle_type"] == 1, "current_pm_duration_hrs"].mean())
    high_wear   = df["operating_hours"] > 2500
    heavy_mask  = df["vehicle_type"] == 1
    high_heavy  = float(df.loc[ high_wear &  heavy_mask, "current_pm_duration_hrs"].mean())
    high_light  = float(df.loc[ high_wear & ~heavy_mask, "current_pm_duration_hrs"].mean())
    low_heavy   = float(df.loc[~high_wear &  heavy_mask, "current_pm_duration_hrs"].mean())
    low_light   = float(df.loc[~high_wear & ~heavy_mask, "current_pm_duration_hrs"].mean())

    # ── Predictions for each tree ───────────────────────────────────────────
    preds1 = np.full(len(df), overall_avg)
    preds2 = np.where(vtypes == 1, heavy_avg, light_avg)
    preds3 = officer_rules_vec(ohrs, vtypes, high_heavy, high_light, low_heavy, low_light)
    mae1   = mean_absolute_error(actual, preds1)
    mae2   = mean_absolute_error(actual, preds2)
    mae3   = mean_absolute_error(actual, preds3)

    st.title("🧑 The Human Way — Officer Samer's Rules of Thumb")
    st.markdown(
        "Officer Samer has run the motor pool for years. He developed a simple "
        "**decision tree** (a mental checklist) he runs through before every PM.\n\n"
        "Watch how much better his predictions get as he adds more splits to his tree."
    )

    # ── Line-chart helper ───────────────────────────────────────────────────
    def _line_chart(preds, mae, title, pred_label, pred_colour):
        worst_i   = int(np.argmax(np.abs(actual - preds)))
        worst_vid = int(vids[worst_i])
        worst_err = actual[worst_i] - preds[worst_i]
        fig = go.Figure()
        fig.add_scatter(
            x=vids, y=actual,
            mode="lines+markers", name="Actual PM Duration",
            line=dict(color=BLUE, width=2),
            marker=dict(
                size=8,
                color=[ORANGE if t else BLUE for t in vtypes],
                symbol=["square" if t else "circle" for t in vtypes],
            ),
        )
        fig.add_scatter(
            x=vids, y=np.round(preds, 1),
            mode="lines", name=f"{pred_label}  (MAE = {mae:.1f})",
            line=dict(color=pred_colour, width=2.5, dash="dot"),
        )
        fig.add_shape(type="line",
            x0=worst_vid, y0=actual[worst_i],
            x1=worst_vid, y1=preds[worst_i],
            line=dict(color=RED, width=1.5, dash="dash"))
        fig.add_annotation(
            x=worst_vid,
            y=(actual[worst_i] + preds[worst_i]) / 2,
            text=f"Biggest miss:<br>{abs(worst_err):.1f} hrs",
            showarrow=True, arrowhead=2, ax=65, ay=0,
            font=dict(color=RED, size=11),
            bgcolor="white", bordercolor=RED, borderwidth=1,
        )
        fig.update_layout(
            title=title,
            xaxis_title="Vehicle ID", yaxis_title="PM Duration (hours)",
            template="plotly_white", height=380,
            font=dict(family="Arial", size=12), title_font_size=14,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, x=0),
            margin=dict(b=70),
        )
        return fig

    # ── Tabs ────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        f"🌱 Tree 1 — Fleet Average  (MAE = {mae1:.1f})",
        f"🌿 Tree 2 — Heavy / Light  (MAE = {mae2:.1f})",
        f"🌳 Tree 3 — Wear × Type  (MAE = {mae3:.1f})",
    ])

    with tab1:
        st.markdown(
            "**Simplest possible rule:** Ignore all context and predict the "
            "*fleet-wide average* PM duration for every vehicle — the classic "
            "\"PM normally takes 6 hours\" estimate."
        )
        st.plotly_chart(_build_dt1_fig(overall_avg), use_container_width=True)
        st.plotly_chart(
            _line_chart(preds1, mae1,
                        f"Tree 1: Always predict the fleet average ({overall_avg:.1f} hrs)",
                        f"Prediction = {overall_avg:.1f} hrs (flat line)", "#9E9E9E"),
            use_container_width=True,
        )
        st.info(
            f"💡 With only one group, every vehicle gets the same prediction: "
            f"**{overall_avg:.1f} hrs**. A worn-out Type-A tank and a nearly-new Type-B truck "
            f"are treated identically. MAE = **{mae1:.1f}**. This is our starting baseline."
        )

    with tab2:
        st.markdown(
            "**One split:** Separate heavy (Type-A) vehicles from light (Type-B) vehicles. "
            "Each group gets its own average, so heavy vehicles get a longer reserved slot."
        )
        st.plotly_chart(_build_dt2_fig(light_avg, heavy_avg), use_container_width=True)
        st.plotly_chart(
            _line_chart(preds2, mae2,
                        f"Tree 2: Light avg ({light_avg:.1f} hrs) vs Heavy avg ({heavy_avg:.1f} hrs)",
                        "Prediction (vehicle-type split)", ORANGE),
            use_container_width=True,
        )
        st.info(
            f"💡 Splitting by vehicle type drops MAE from {mae1:.1f} → **{mae2:.1f}** "
            f"({mae1 - mae2:.1f} hours better per job). "
            f"Light average: **{light_avg:.1f} hrs** — Heavy average: **{heavy_avg:.1f} hrs**. "
            f"Operating hours are still completely ignored."
        )

    with tab3:
        st.markdown(
            "**Two splits:** First check operating hours (high-wear vs low-wear), "
            "then check vehicle type. This gives four distinct groups, each with its own average."
        )
        st.plotly_chart(
            _build_decision_tree_fig(high_heavy, high_light, low_heavy, low_light),
            use_container_width=True,
        )
        st.plotly_chart(
            _line_chart(preds3, mae3,
                        "Tree 3: Operating Hours × Vehicle Type (4 groups)",
                        "Prediction (4-group tree)", GREEN),
            use_container_width=True,
        )
        st.info(
            f"💡 Adding operating hours drops MAE from {mae2:.1f} → **{mae3:.1f}** "
            f"({mae2 - mae3:.1f} hours better per job). "
            f"Leaf values: high-wear heavy **{high_heavy:.1f} hrs**, high-wear light **{high_light:.1f} hrs**, "
            f"low-wear heavy **{low_heavy:.1f} hrs**, low-wear light **{low_light:.1f} hrs**. "
            f"Real durations still vary around each average — but predictions are now much closer."
        )

    st.markdown("---")
    st.markdown("### How do the three trees compare?")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tree 1 MAE", f"{mae1:.1f}", help="One leaf — fleet average")
    c2.metric("Tree 2 MAE", f"{mae2:.1f}",
              delta=f"{mae2 - mae1:+.1f} vs Tree 1", delta_color="inverse",
              help="Two leaves — heavy vs light")
    c3.metric("Tree 3 MAE", f"{mae3:.1f}",
              delta=f"{mae3 - mae2:+.1f} vs Tree 2", delta_color="inverse",
              help="Four leaves — operating hours × vehicle type")
    st.caption(
        "MAE = Mean Absolute Error (avg prediction mistake in hours per job). "
        "Lower is better."
    )
    st.success(
        f"✅ **Key takeaway:** The more relevant groups we define, the lower the error. "
        f"MAE {mae1:.1f} → {mae2:.1f} → **{mae3:.1f}** hours/job. "
        f"But writing rules by hand doesn't scale — the Formula Way and ML Way "
        f"will find these splits automatically."
    )


# ──────────────────────────────────────────────────────────────────────────────

def render_formula_way():
    df = generate_data()

    st.title("📐 The Formula Way — Lieutenant Samira's Approach")
    st.markdown(
        "Lieutenant Samira says: *\"If I plot operating hours against PM duration, "
        "I should see a straight line.\"* Let's check."
    )

    # ──────────────────────────────────────────────────────────────────────
    # 2.1  Scatter first
    # ──────────────────────────────────────────────────────────────────────
    st.markdown("#### Step 1 — Look at the data first")
    fig_s = px.scatter(
        df,
        x="operating_hours", y="current_pm_duration_hrs",
        color=df["vehicle_type"].map({0: "Type-B (Light)", 1: "Type-A (Heavy)"}),
        symbol=df["vehicle_type"].map({0: "Type-B (Light)", 1: "Type-A (Heavy)"}),
        color_discrete_map={"Type-B (Light)": BLUE, "Type-A (Heavy)": ORANGE},
        template="plotly_white",
        labels={
            "operating_hours": "Operating Hours",
            "current_pm_duration_hrs": "PM Duration (hours)",
            "color": "Vehicle Type", "symbol": "Vehicle Type",
        },
        title="Operating Hours vs. PM Duration",
    )
    fig_s.update_layout(
        width=800, height=450,
        font=dict(family="Arial", size=13), title_font_size=15,
    )
    # alt: Scatter plot with heavy Type-A vehicles as orange squares and light Type-B vehicles as blue circles; the data shows a clear upward trend from left to right
    st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("---")
    # ──────────────────────────────────────────────────────────────────────
    # 2.2  Manual slider fitting  —  sliders left | chart right
    # ──────────────────────────────────────────────────────────────────────
    st.markdown("#### Step 2 — Tune the formula yourself")
    st.markdown(
        "Lieutenant Samira's formula has three numbers (**parameters**) to find: "
        "`predicted = m1 × operating_hours_k + m2 × vehicle_type + c`  \n"
        "(*operating_hours_k = operating hours ÷ 1,000, so the sliders stay easy to handle*)  \n"
        "Drag the sliders to minimise the MAE — you're doing what the machine does, by hand!"
    )

    col_ctrl, col_chart = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.markdown("**Adjust the parameters:**")
        st.markdown(
            "<small>Each slider controls one part of the formula:<br>"
            "<b>m1</b> = extra hours per 1,000 operating hours<br>"
            "<b>m2</b> = heavy-vehicle (Type-A) bonus<br>"
            "<b>c</b> = baseline PM duration</small>",
            unsafe_allow_html=True,
        )
        m1 = st.slider(
            "m1 — Operating-hours slope (per 1,000 hrs)",
            min_value=0.0, max_value=5.0, value=0.0, step=0.25,
        )
        m2 = st.slider(
            "m2 — Heavy-vehicle (Type-A) bonus",
            min_value=0.0, max_value=8.0, value=0.0, step=0.5,
        )
        c = st.slider(
            "c — Baseline PM duration (hours)",
            min_value=-5.0, max_value=15.0, value=5.0, step=0.5,
        )
        samira_preds = m1 * df["operating_hours_k"].values + m2 * df["vehicle_type"].values + c
        mae_samira   = mean_absolute_error(df["current_pm_duration_hrs"].values, samira_preds)
        # Persist for ML Way comparison
        st.session_state["samira_m1"]  = m1
        st.session_state["samira_m2"]  = m2
        st.session_state["samira_c"]   = c
        st.session_state["samira_mae"] = float(mae_samira)
        st.metric("Current MAE", f"{mae_samira:.1f} hours")
        st.markdown(
            "<small>💡 Try to get MAE as low as possible.</small>",
            unsafe_allow_html=True,
        )

    with col_chart:
        fig2 = go.Figure()
        fig2.add_scatter(
            x=df[df["vehicle_type"] == 0]["vehicle_id"],
            y=df[df["vehicle_type"] == 0]["current_pm_duration_hrs"],
            mode="markers", name="Actual (Type-B, light)",
            marker=dict(color=BLUE, symbol="circle", size=8),
        )
        fig2.add_scatter(
            x=df[df["vehicle_type"] == 1]["vehicle_id"],
            y=df[df["vehicle_type"] == 1]["current_pm_duration_hrs"],
            mode="markers", name="Actual (Type-A, heavy)",
            marker=dict(color=ORANGE, symbol="square", size=9),
        )
        fig2.add_scatter(
            x=df["vehicle_id"], y=samira_preds,
            mode="lines", name="Your prediction",
            line=dict(color=RED, width=2.5),
        )
        fig2.update_layout(
            title="Your Formula Predictions vs. Actual PM Duration",
            xaxis_title="Vehicle ID", yaxis_title="PM Duration (hours)",
            template="plotly_white", height=400,
            font=dict(family="Arial", size=12), title_font_size=14,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="left", x=0),
            margin=dict(b=70),
        )
        # alt: Vehicle-by-vehicle chart — blue circles = light-vehicle actual duration, orange squares = heavy-vehicle actual duration, red line = the officer's formula prediction that responds to all three sliders
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "Watch the **red line** move as you drag any slider. "
            "m1 steepens or flattens it overall. "
            "m2 lifts the line on heavy-vehicle jobs (orange squares). "
            "c raises or lowers everything."
        )

    st.info(
        "💡 **You just tuned three numbers manually** — one for each feature's effect and one baseline. "
        "This is exactly what the machine will do automatically in the next section. "
        "Head to **🤖 The ML Way** to see how it compares!"
    )


# ──────────────────────────────────────────────────────────────────────────────

def render_ml_way():
    df = generate_data()
    m1_opt, m2_opt, c_opt, lr_preds, mae_lr = fit_linear_regression()

    # ── Lieutenant Samira's slider values from Formula Way ────────────────
    samira_m1  = float(st.session_state.get("samira_m1",  1.0))
    samira_m2  = float(st.session_state.get("samira_m2",  2.0))
    samira_c   = float(st.session_state.get("samira_c",   4.0))
    samira_preds = (
        samira_m1 * df["operating_hours_k"].values
        + samira_m2 * df["vehicle_type"].values
        + samira_c
    )
    samira_mae = float(
        st.session_state.get("samira_mae")
        or mean_absolute_error(df["current_pm_duration_hrs"].values, samira_preds)
    )

    # ── Training state ─────────────────────────────────────────────────────
    y_vals = df["current_pm_duration_hrs"].values
    X_oh   = df["operating_hours_k"].values
    X_type = df["vehicle_type"].values
    y_mean = float(np.mean(y_vals))
    N_EPOCHS = 60

    if "ml_trained" not in st.session_state:
        st.session_state.ml_trained = False
        st.session_state.ml_m1 = 0.0
        st.session_state.ml_m2 = 0.0
        st.session_state.ml_c  = y_mean

    trained = bool(st.session_state.ml_trained)

    # ── Vehicle-axis chart builder ──────────────────────────────────────────
    def _vehicle_chart(m1, m2, c, ep=None):
        preds = m1 * X_oh + m2 * X_type + c
        mae_v = mean_absolute_error(y_vals, preds)
        is_trn = ep is not None
        line_colour = GREEN if (is_trn or trained) else "#AAAAAA"
        if is_trn:
            title = f"Vehicle-by-Vehicle Predictions  |  Epoch {ep}/{N_EPOCHS}  |  MAE = {mae_v:.1f}"
        elif trained:
            title = f"Vehicle-by-Vehicle Predictions — Trained  |  Machine MAE = {mae_v:.1f}"
        else:
            title = "Vehicle-by-Vehicle Predictions  (click Train to start)"
        fig = go.Figure()
        light = df[df["vehicle_type"] == 0]
        heavy = df[df["vehicle_type"] == 1]
        fig.add_scatter(
            x=light["vehicle_id"], y=light["current_pm_duration_hrs"],
            mode="markers", name="Actual (Type-B, light)",
            marker=dict(color=BLUE, symbol="circle", size=7),
        )
        fig.add_scatter(
            x=heavy["vehicle_id"], y=heavy["current_pm_duration_hrs"],
            mode="markers", name="Actual (Type-A, heavy)",
            marker=dict(color=ORANGE, symbol="square", size=9),
        )
        fig.add_scatter(
            x=df["vehicle_id"], y=np.round(preds, 1),
            mode="lines+markers",
            name=f"Machine prediction  (MAE = {mae_v:.1f})",
            line=dict(color=line_colour, width=2),
            marker=dict(size=4),
        )
        fig.update_layout(
            title=title,
            xaxis_title="Vehicle ID", yaxis_title="PM Duration (hours)",
            template="plotly_white", height=420,
            font=dict(family="Arial", size=12), title_font_size=14,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="left", x=0),
            margin=dict(b=70),
        )
        return fig, mae_v

    st.title("🤖 The ML Way — Captain Fares's Approach")
    st.markdown(
        "> *\"Captain Fares takes the exact same formula structure Lieutenant Samira used — "
        "but instead of dragging sliders by hand, he gives the maintenance history to the machine "
        "and lets it find the best m1, m2, and c automatically.\"*"
    )

    # ── Live chart + Train button ───────────────────────────────────────────
    st.markdown(
        "#### Watch the machine learn — epoch by epoch\n\n"
        "The machine uses **gradient descent**: it starts with a flat guess "
        "(m1=0, m2=0, c=average PM duration) and nudges m1, m2, c step-by-step in the direction "
        "that shrinks the error the most, until it converges on an optimal answer."
    )

    chart_ph = st.empty()
    col_mae, col_status = st.columns([1, 2])
    mae_ph    = col_mae.empty()
    status_ph = col_status.empty()

    m1_cur = st.session_state.ml_m1
    m2_cur = st.session_state.ml_m2
    c_cur  = st.session_state.ml_c
    fig_cur, mae_cur = _vehicle_chart(m1_cur, m2_cur, c_cur)
    chart_ph.plotly_chart(fig_cur, use_container_width=True)
    mae_ph.metric("Machine's MAE", f"{mae_cur:.1f} hours")
    if trained:
        status_ph.success("✅ Training complete! Click Train to re-run.")

    if st.button("🚀 Train", type="primary", key="train_btn"):
        oh_mean = float(np.mean(X_oh));    oh_std = float(np.std(X_oh))
        ty_mean = float(np.mean(X_type));  ty_std = max(float(np.std(X_type)), 1e-8)
        oh_norm = (X_oh - oh_mean) / oh_std
        ty_norm = (X_type - ty_mean) / ty_std

        m1_n, m2_n, c_n = 0.0, 0.0, y_mean
        n  = len(y_vals)
        lr = 0.15

        for ep in range(1, N_EPOCHS + 1):
            preds_n = m1_n * oh_norm + m2_n * ty_norm + c_n
            err     = preds_n - y_vals
            m1_n   -= lr * (2 / n) * float(np.dot(err, oh_norm))
            m2_n   -= lr * (2 / n) * float(np.dot(err, ty_norm))
            c_n    -= lr * (2 / n) * float(np.sum(err))

            m1_d =  m1_n / oh_std
            m2_d =  m2_n / ty_std
            c_d  =  c_n - m1_d * oh_mean - m2_d * ty_mean

            fig_ep, mae_ep = _vehicle_chart(m1_d, m2_d, c_d, ep=ep)
            chart_ph.plotly_chart(fig_ep, use_container_width=True)
            mae_ph.metric(
                "Machine's MAE", f"{mae_ep:.1f} hours",
                delta=f"Epoch {ep}/{N_EPOCHS}, m1={m1_d:.2f}, m2={m2_d:.2f}, c={c_d:.2f}", delta_color="off",
            )
            status_ph.info(f"🔄 Training... epoch {ep} of {N_EPOCHS}")
            time.sleep(0.12)

        st.session_state.ml_m1     = m1_d
        st.session_state.ml_m2     = m2_d
        st.session_state.ml_c      = c_d
        st.session_state.ml_trained = True

        status_ph.success(f"✅ Done! Machine MAE = {mae_ep:.1f} hours after {N_EPOCHS} epochs.")

    st.markdown("---")
    # ── Parameter comparison ────────────────────────────────────────────────
    st.markdown("#### What values did the machine find?")
    st.markdown(
        "Lieutenant Samira's formula: `predicted = m1 × operating_hours_k + m2 × vehicle_type + c`\n\n"
        "scikit-learn's `LinearRegression` computes the analytically optimal values "
        "in a single pass — here's how they compare:"
    )
    comp_df = pd.DataFrame({
        "Parameter":           ["Operating-hours slope", "Heavy-vehicle bonus", "Intercept"],
        "Symbol":              ["m1", "m2", "c"],
        "Samira's best guess":   [round(samira_m1, 2), round(samira_m2, 1), round(samira_c, 1)],
        "Machine's optimal":   [round(m1_opt, 2), round(m2_opt, 2), round(c_opt, 2)],
        "Plain English": [
            f"Each 1,000 extra hrs adds ~{m1_opt:.1f} hrs of PM",
            f"Heavy vehicles add ~{m2_opt:.1f} hrs of PM",
            "Baseline PM duration when hours = 0, light vehicle",
        ],
    })
    st.dataframe(comp_df, use_container_width=True)
    st.caption(
        "Samira's best guess = the slider values you left in the Formula Way section. "
        "Machine's optimal = found automatically from the same 30 maintenance jobs."
    )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📐 Samira's MAE", f"{samira_mae:.1f} hours")
        st.caption("Your slider values from the Formula Way section.")
    with col2:
        improvement = samira_mae - mae_lr
        st.metric(
            "🤖 Machine's MAE", f"{mae_lr:.1f} hours",
            delta=f"{improvement:+.1f} hours vs Samira",
            delta_color="inverse",
        )
        st.caption("scikit-learn LinearRegression on the same 30 maintenance jobs.")

    st.info(
        f"💡 **The machine uses the exact same formula Lieutenant Samira wrote.** "
        f"The only difference: Samira tuned three numbers by hand, "
        f"while the machine searched the entire parameter space and found "
        f"m1 = {m1_opt:.2f}, m2 = {m2_opt:.2f}, c = {c_opt:.2f} — "
        f"the mathematically optimal combination for this dataset."
    )

# ──────────────────────────────────────────────────────────────────────────────

def render_comparison():
    return  # section removed
    df = generate_data()
    m1_opt, m2_opt, c_opt, lr_preds, mae_lr = fit_linear_regression()
    _, rf_preds_all, _mae_tr, _, _, _, heat_preds = fit_random_forest(30)

    ohrs   = df["operating_hours"].values
    vtypes = df["vehicle_type"].values
    actual = df["current_pm_duration_hrs"].values

    officer_preds = officer_rules_vec(ohrs, vtypes)
    mae_officer   = mean_absolute_error(actual, officer_preds)
    mae_rf        = mean_absolute_error(actual, rf_preds_all)

    st.title("⚖️ The Big Comparison — Three Approaches, One Problem")
    # (kept for parity with lesson_1_app.py; not reachable from navigation)


# ──────────────────────────────────────────────────────────────────────────────

def render_reflect_quiz():
    return  # section removed
    st.title("💡 Reflect & Quiz")
    # (kept for parity with lesson_1_app.py; not reachable from navigation)


# ══════════════════════════════════════════════════════════════════════════════
# ML Defined
# ══════════════════════════════════════════════════════════════════════════════

def render_ml_defined():
    rng = np.random.RandomState(42)

    st.title("🧠 What Is Machine Learning — A Map")
    st.markdown(
        "> *You just watched a machine learn from maintenance history. "
        "Now let's name what happened — and map it to the broader landscape of AI.*"
    )

    # ── Anchor to the experiment ─────────────────────────────────────────────
    st.markdown("### From the fleet-maintenance experiment to a formal definition")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**📥 Input (Features)**\nOperating Hours, Vehicle Type")
    with c2:
        st.info("**📤 Output (Label)**\nPM Duration (hours)")
    with c3:
        st.info("**🔁 Training**\nFind m1, m2, c that minimise prediction error across all 30 jobs")

    st.success(
        "### Machine Learning — working definition\n\n"
        "**Machine learning** is the process of giving a computer **data** "
        "(examples of inputs and correct outputs) and letting it automatically find "
        "the **mathematical pattern** that best maps one to the other — "
        "without a human writing explicit rules.\n\n"
        "The \"learning\" happens when the machine searches for the set of internal numbers "
        "(**parameters**) that minimise prediction errors across all training examples."
    )

    st.markdown("---")
    # ── Three Types ───────────────────────────────────────────────────────────
    st.markdown("## The Three Types of Machine Learning")

    tab_sup, tab_unsup, tab_rl = st.tabs([
        "🏫 Supervised Learning",
        "🔍 Unsupervised Learning",
        "🎮 Reinforcement Learning",
    ])

    # ── Supervised ────────────────────────────────────────────────────────────
    with tab_sup:
        st.markdown(
            "#### Supervised Learning\n\n"
            "The machine is given **labelled** training examples — pairs of (input, correct output). "
            "It learns to predict the output for new, unseen inputs.\n\n"
            "👉 **The PM-duration experiment was supervised**: each maintenance job had "
            "operating hours + vehicle type (inputs) AND actual PM duration "
            "(the label the machine optimised towards)."
        )
        x_sup  = rng.uniform(20, 400, 30)     # convoy distance, miles
        y_sup  = 0.9 * x_sup + 15 + rng.normal(0, 18, 30)   # fuel used, gallons
        m_s, b_s = np.polyfit(x_sup, y_sup, 1)
        x_line = np.linspace(20, 400, 100)
        fig_sup = go.Figure()
        fig_sup.add_scatter(
            x=x_sup, y=y_sup, mode="markers", name="Labelled training examples",
            marker=dict(color=BLUE, size=9),
        )
        fig_sup.add_scatter(
            x=x_line, y=m_s * x_line + b_s, mode="lines",
            name="Learned model (predicts fuel from distance)",
            line=dict(color=GREEN, width=2.5),
        )
        fig_sup.add_annotation(
            x=320, y=m_s * 320 + b_s + 35,
            text="Machine learned this mapping\nfrom labelled data",
            showarrow=True, arrowhead=2, ax=-70, ay=-40,
            font=dict(size=11), bgcolor="white", bordercolor=GREEN,
        )
        fig_sup.update_layout(
            title="Supervised Learning — Regression  (convoy distance → fuel used)",
            xaxis_title="Convoy Distance (miles)", yaxis_title="Fuel Used (gallons)",
            template="plotly_white", height=390,
            font=dict(family="Arial", size=12),
            legend=dict(orientation="h", yanchor="bottom", y=-0.28, x=0),
            margin=dict(b=65),
        )
        st.plotly_chart(fig_sup, use_container_width=True)
        st.caption(
            "**Examples:** PM-duration forecasting (this lesson), fuel-consumption forecasting, "
            "sensor-failure prediction, ammunition-resupply forecasting, casualty-evacuation ETA."
        )

    # ── Unsupervised ──────────────────────────────────────────────────────────
    with tab_unsup:
        st.markdown(
            "#### Unsupervised Learning\n\n"
            "The machine is given **unlabelled** data — only inputs, no correct answers. "
            "It discovers hidden structure, groups, or patterns on its own.\n\n"
            "There is no 'right answer' to check against; "
            "the machine finds what naturally belongs together."
        )
        cx = np.concatenate([rng.normal(2, 0.5, 22), rng.normal(8, 0.6, 22), rng.normal(5, 0.5, 20)])
        cy = np.concatenate([rng.normal(8, 0.6, 22), rng.normal(4, 0.5, 22), rng.normal(1.5, 0.4, 20)])
        cluster_names = ["Logistics tempo"] * 22 + ["Combat tempo"] * 22 + ["Recon tempo"] * 20
        cluster_arr   = np.array(cluster_names)
        fig_unsup = go.Figure()
        for name, col in [("Logistics tempo", BLUE), ("Combat tempo", ORANGE), ("Recon tempo", GREEN)]:
            mask = cluster_arr == name
            fig_unsup.add_scatter(
                x=cx[mask], y=cy[mask], mode="markers", name=f"Cluster: {name}",
                marker=dict(color=col, size=10),
            )
        fig_unsup.update_layout(
            title="Unsupervised Learning — Clustering  (unit ops patterns, no labels needed)",
            xaxis_title="Missions per Week", yaxis_title="Avg Mission Duration (hrs)",
            template="plotly_white", height=390,
            font=dict(family="Arial", size=12),
            legend=dict(orientation="h", yanchor="bottom", y=-0.28, x=0),
            margin=dict(b=65),
        )
        st.plotly_chart(fig_unsup, use_container_width=True)
        st.caption(
            "The machine was given ONLY mission-tempo data — no unit labels. "
            "It discovered three natural operating patterns by itself. "
            "**Examples:** unit grouping by operational tempo, terrain clustering, "
            "anomaly detection in sensor telemetry."
        )

    # ── Reinforcement Learning ────────────────────────────────────────────────
    with tab_rl:
        st.markdown(
            "#### Reinforcement Learning\n\n"
            "An **agent** learns by interacting with an **environment**. "
            "It takes actions, receives **rewards** (or penalties), "
            "and gradually learns which actions lead to the best long-term outcome.\n\n"
            "There is no fixed dataset — the agent generates its own experience through trial and error."
        )
        episodes = np.arange(1, 201)
        rewards  = -100 + 120 * (1 - np.exp(-episodes / 60)) + rng.normal(0, 12, 200)
        window   = 15
        smooth   = np.convolve(rewards, np.ones(window) / window, mode="valid")
        offset   = window // 2
        fig_rl   = go.Figure()
        fig_rl.add_scatter(
            x=episodes, y=rewards, mode="lines", name="Episode reward (noisy)",
            line=dict(color=BLUE, width=1.5),
        )
        fig_rl.add_scatter(
            x=episodes[offset: offset + len(smooth)], y=smooth,
            mode="lines", name="Trend (smoothed)",
            line=dict(color=GREEN, width=3),
        )
        fig_rl.add_hline(y=0, line_dash="dash", line_color="grey",
                         annotation_text="Break-even", annotation_position="right")
        fig_rl.add_annotation(
            x=180, y=float(smooth[-1]) + 12,
            text="Agent has learned\neffective policy",
            showarrow=True, arrowhead=2, ax=-60, ay=-35,
            font=dict(size=11), bgcolor="white", bordercolor=GREEN,
        )
        fig_rl.add_annotation(
            x=10, y=float(rewards[:10].mean()) - 18,
            text="Random actions\n(no policy yet)",
            showarrow=True, arrowhead=2, ax=30, ay=30,
            font=dict(size=11), bgcolor="white", bordercolor=RED,
        )
        fig_rl.update_layout(
            title="Reinforcement Learning — Resupply Drone Improves Over Episodes",
            xaxis_title="Training Episode", yaxis_title="Total Reward",
            template="plotly_white", height=390,
            font=dict(family="Arial", size=12),
            legend=dict(orientation="h", yanchor="bottom", y=-0.28, x=0),
            margin=dict(b=65),
        )
        st.plotly_chart(fig_rl, use_container_width=True)
        st.caption(
            "Early episodes: random routing → poor reward (fuel wasted, late deliveries). "
            "As the agent explores and learns from mistakes, cumulative reward climbs. "
            "**Examples:** autonomous resupply-drone routing, EOD-robot arm control, "
            "wargame simulation AI."
        )

    st.markdown("---")
    # ── Predictive vs Generative ────────────────────────────────────────────
    st.markdown("## Predictive AI vs. Generative AI")
    st.markdown(
        "Not all AI produces a number or a category — some AI **generates entirely new content**. "
        "These two paradigms serve fundamentally different purposes."
    )

    col_pred, col_gen = st.columns(2)

    with col_pred:
        st.markdown("### 📊 Predictive AI")
        st.markdown(
            "Takes inputs and outputs a **prediction** — a number, a category, or a probability.\n\n"
            "**Output type:** Single value or label\n\n"
            "**Examples:**\n"
            "- Will this vehicle fail before its next PM? (Yes / No)\n"
            "- PM duration for tomorrow's job? (a number)\n"
            "- Is this sensor reading an anomaly? (probability)\n"
            "- What component needs replacement? (category)\n\n"
            "**You've been using Predictive AI in this lesson.** "
            "Captain Fares's model predicts a specific number: expected PM duration."
        )
        xx  = np.linspace(0, 10, 60)
        yy  = np.linspace(0, 10, 60)
        XXg, YYg = np.meshgrid(xx, yy)
        boundary = (XXg + YYg > 10).astype(float)
        x0 = rng.uniform(0, 4, 25);  y0 = rng.uniform(0, 9, 25)
        x1 = rng.uniform(6, 10, 25); y1 = rng.uniform(2, 10, 25)
        fig_pred = go.Figure()
        fig_pred.add_contour(
            x=xx, y=yy, z=boundary,
            colorscale=[[0, "#BBDEFB"], [1, "#FFCCBC"]],
            showscale=False, opacity=0.45,
            contours=dict(showlabels=False),
        )
        fig_pred.add_scatter(x=x0, y=y0, mode="markers", name="Class A (no failure risk)",
            marker=dict(color=BLUE, size=8))
        fig_pred.add_scatter(x=x1, y=y1, mode="markers", name="Class B (failure risk)",
            marker=dict(color=ORANGE, size=8, symbol="square"))
        xb = np.array([0, 10])
        fig_pred.add_scatter(x=xb, y=10 - xb, mode="lines", name="Decision boundary",
            line=dict(color=RED, width=2, dash="dash"))
        fig_pred.add_annotation(x=2, y=8.5, text="Predict: No failure risk",
            showarrow=False, font=dict(size=11, color=BLUE), bgcolor="white")
        fig_pred.add_annotation(x=8, y=2, text="Predict: Failure risk",
            showarrow=False, font=dict(size=11, color=ORANGE), bgcolor="white")
        fig_pred.update_layout(
            title="Predictive AI — Classification boundary",
            xaxis_title="Engine Hours Since Overhaul", yaxis_title="Vibration Sensor Reading",
            template="plotly_white", height=350,
            font=dict(family="Arial", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0),
            margin=dict(b=70),
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    with col_gen:
        st.markdown("### ✨ Generative AI")
        st.markdown(
            "Takes a prompt or noise and **generates new content** — "
            "text, images, audio, code, or video.\n\n"
            "**Output type:** Rich media (not just a number)\n\n"
            "**Examples:**\n"
            "- ChatGPT / GPT-4: drafts mission briefs and reports\n"
            "- DALL·E / Stable Diffusion: generates synthetic training imagery\n"
            "- GitHub Copilot: generates code for simulation tools\n"
            "- Sora: generates video for wargame scenarios\n"
            "- Suno: generates audio/music\n\n"
            "**Trained on massive self-supervised datasets** (most of the internet) "
            "to learn the underlying distribution of human-generated content."
        )
        modalities = ["Text", "Code", "Images", "Audio", "Video"]
        maturity   = [96, 88, 85, 72, 58]
        fig_gen = go.Figure()
        fig_gen.add_bar(
            x=modalities, y=maturity,
            marker_color=[GREEN, BLUE, ORANGE, "#9C27B0", RED],
            text=[f"{v}%" for v in maturity],
            textposition="outside",
        )
        fig_gen.update_layout(
            title="Generative AI — 2024 capability maturity by output type (illustrative)",
            xaxis_title="Output Modality", yaxis_title="Capability Maturity (%)",
            template="plotly_white", height=350,
            font=dict(family="Arial", size=11),
            yaxis=dict(range=[0, 115]),
            showlegend=False,
            margin=dict(b=40),
        )
        st.plotly_chart(fig_gen, use_container_width=True)

    st.markdown("---")
    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown("### Summary — how they all connect")
    summary_df = pd.DataFrame({
        "Approach": [
            "🏫 Supervised Learning",
            "🔍 Unsupervised Learning",
            "🎮 Reinforcement Learning",
            "📊 Predictive AI",
            "✨ Generative AI",
        ],
        "Labelled data?": ["✅ Yes", "❌ No", "❌ No (uses rewards)", "✅ Yes", "Mostly no (self-supervised)"],
        "Output type": [
            "Prediction (number / class)",
            "Structure / clusters",
            "Best action policy",
            "Single value or label",
            "New content (text, image…)",
        ],
        "Example": [
            "PM-duration forecasting, fuel prediction",
            "Unit clustering by ops tempo",
            "Autonomous resupply routing",
            "All supervised ML models (this lesson)",
            "ChatGPT, DALL·E, Copilot",
        ],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.info(
        "💡 **The PM-duration model you just built is Supervised Predictive AI** — "
        "one of the most widely deployed forms of machine learning. "
        "Everything else builds on the same core idea: give the machine data, "
        "define what 'error' means, and minimise it."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.sidebar.title("🪖 What is Machine Learning")
    #st.sidebar.markdown("**What Is Machine Learning?**")
    #st.sidebar.markdown("*Modern AI/ML Course — Week 1, Module 0.1*")
    st.sidebar.markdown("---")

    section = st.sidebar.radio("Navigate to section:", SECTIONS, index=0)

    # Scroll to top on every section change.
    # Retries over 800 ms to catch Streamlit's async DOM patching.
    st.components.v1.html(
        f"""<script>
        (function() {{
            var selectors = [
                '[data-testid="stApp"]',
                '[data-testid="stAppViewContainer"]',
                '[data-testid="stMain"]',
                '[data-testid="stMainBlockContainer"]',
                'section.main',
                '.main',
            ];
            function scrollUp() {{
                selectors.forEach(function(s) {{
                    var el = window.parent.document.querySelector(s);
                    if (el) el.scrollTop = 0;
                }});
                window.parent.scrollTo(0, 0);
                window.parent.document.documentElement.scrollTop = 0;
                window.parent.document.body.scrollTop = 0;
            }}
            [0, 100, 250, 500, 800].forEach(function(t) {{
                setTimeout(scrollUp, t);
            }});
        }})(); /* {section} */
        </script>""",
        height=0,
    )

    st.sidebar.markdown("---")
    with st.sidebar.expander("ℹ️ About this lesson"):
        st.markdown(
            "This is a fully interactive lesson — no code is visible. "
            "Use the sliders, charts, and quizzes to build intuition.\n\n"
            "Every concept follows the pattern:\n"
            "**Story → Data → Human → Formula → ML → ML Defined**"
        )

    if section == SECTIONS[0]:
        render_story()
    elif section == SECTIONS[1]:
        render_data()
    elif section == SECTIONS[2]:
        render_human_way()
    elif section == SECTIONS[3]:
        render_formula_way()
    elif section == SECTIONS[4]:
        render_ml_way()
    elif section == SECTIONS[5]:
        render_ml_defined()


if __name__ == "__main__":
    main()
