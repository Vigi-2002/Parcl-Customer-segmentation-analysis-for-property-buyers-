"""
╔══════════════════════════════════════════════════════════════╗
║      PARCL  –  Property Buyer Segmentation Dashboard        ║
║      Streamlit Web Application                               ║
╚══════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import plotly.express       as px
import plotly.graph_objects as go
import streamlit as st
from   sklearn.preprocessing import StandardScaler
from   sklearn.cluster       import KMeans
from   sklearn.decomposition import PCA

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title            = "Parcl · Buyer Segmentation",
    page_icon             = "🏙️",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ──────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif; background:#0D1117; color:#C9D1D9; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#161B22 0%,#0D1117 100%);
    border-right: 1px solid #21262D;
}
[data-testid="stMetric"] {
    background:#161B22; border:1px solid #21262D;
    border-radius:12px; padding:1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color:#8B949E !important; font-size:0.78rem; }
[data-testid="stMetricValue"] { color:#E6EDF3 !important; font-size:1.6rem; font-weight:700; }
.section-title {
    font-size:1.15rem; font-weight:700; color:#E6EDF3;
    border-left:4px solid #4361EE; padding-left:0.6rem; margin:1.5rem 0 0.8rem;
}
[data-baseweb="tab-list"] { background:#161B22; border-radius:10px; padding:4px; gap:4px; }
[data-baseweb="tab"]      { border-radius:8px !important; color:#8B949E !important; font-weight:600; }
[aria-selected="true"]    { background:#4361EE !important; color:#fff !important; }
hr { border-color:#21262D !important; }
.stPlotlyChart { border:1px solid #21262D; border-radius:12px; overflow:hidden; }
.header-banner {
    background:linear-gradient(135deg,#1C2128 0%,#161B22 50%,#1C2128 100%);
    border:1px solid #21262D; border-radius:16px; padding:1.6rem 2rem; margin-bottom:1.5rem;
}
.header-banner h1 { color:#E6EDF3; margin:0; font-size:2rem; font-weight:700; }
.header-banner p  { color:#8B949E; margin:0.3rem 0 0; font-size:0.95rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# COLOUR PALETTE & THEME HELPER
# ──────────────────────────────────────────────────────────────
SEG_COLORS = {
    "Luxury Investors":  "#F3722C",
    "Global Investors":  "#4361EE",
    "Corporate Buyers":  "#7209B7",
    "First-Time Buyers": "#F72585",
}
PIE_COLORS = ["#4361EE","#F72585","#7209B7","#F3722C","#4CC9F0","#90BE6D"]

# ── IMPORTANT: _BASE has NO xaxis/yaxis keys.
#    Those are applied separately via update_xaxes / update_yaxes
#    to avoid "multiple values for keyword argument" errors.
_BASE = dict(
    paper_bgcolor = "#0D1117",
    plot_bgcolor  = "#161B22",
    font_color    = "#C9D1D9",
    colorway      = list(SEG_COLORS.values()),
)
_AXIS = dict(gridcolor="#21262D", zerolinecolor="#21262D")

def tl(fig, **extra):
    """Apply dark theme + axis grid.  All caller kwargs go into update_layout directly."""
    fig.update_layout(**_BASE, margin=dict(l=20,r=20,t=45,b=20), **extra)
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(**_AXIS)
    return fig


# ──────────────────────────────────────────────────────────────
# DATA PIPELINE  (cached)
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⚙️  Running segmentation pipeline …")
def load_and_segment():
    clients = pd.read_csv("Dataset/clients.csv")
    props   = pd.read_csv("Dataset/properties.csv")

    clients = clients.drop_duplicates(subset="client_id")
    for col in ["client_type","gender","country","region",
                "acquisition_purpose","referral_channel","loan_applied"]:
        clients[col] = clients[col].str.strip().str.title()

    clients["dob"] = pd.to_datetime(clients["date_of_birth"], dayfirst=False, errors="coerce")
    today = pd.Timestamp("2024-01-01")
    clients["age"] = ((today - clients["dob"]).dt.days / 365.25).round(1)
    clients = clients[(clients["age"] >= 10) & (clients["age"] <= 110)]

    props["sale_price"] = (props["sale_price"].astype(str)
                           .str.replace(r"[\$,]", "", regex=True).astype(float))

    prop_agg = (props[props["listing_status"] == "Sold"]
                .rename(columns={"client_ref": "client_id"})
                .groupby("client_id")
                .agg(num_properties   =("listing_id",     "count"),
                     total_investment =("sale_price",      "sum"),
                     avg_unit_area    =("floor_area_sqft", "mean"))
                .reset_index())

    df = clients.merge(prop_agg, on="client_id", how="left")
    df["num_properties"]   = df["num_properties"].fillna(0)
    df["total_investment"] = df["total_investment"].fillna(0)
    df["avg_unit_area"]    = df["avg_unit_area"].fillna(df["avg_unit_area"].median())

    df["loan_enc"]    = (df["loan_applied"]       == "Yes").astype(int)
    df["type_enc"]    = (df["client_type"]         == "Company").astype(int)
    df["purpose_enc"] = (df["acquisition_purpose"] == "Investment").astype(int)

    ohe   = pd.get_dummies(df[["referral_channel","country"]], drop_first=False).astype(int)
    X_raw = pd.concat([
        df[["age","satisfaction_score","num_properties",
            "total_investment","avg_unit_area",
            "loan_enc","type_enc","purpose_enc"]].reset_index(drop=True),
        ohe.reset_index(drop=True),
    ], axis=1).fillna(0)

    X  = StandardScaler().fit_transform(X_raw)
    km = KMeans(n_clusters=4, init="k-means++", n_init=15, random_state=42)
    df["cluster"] = km.fit_predict(X)

    order = (df.groupby("cluster")["total_investment"].mean()
               .sort_values(ascending=False).index.tolist())
    roles = ["Luxury Investors","Global Investors","Corporate Buyers","First-Time Buyers"]
    df["segment"] = df["cluster"].map({o: r for o, r in zip(order, roles)})

    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    df["pc1"] = coords[:, 0]
    df["pc2"] = coords[:, 1]

    return df, pca.explained_variance_ratio_

df_full, pca_ev = load_and_segment()


# ──────────────────────────────────────────────────────────────
# SIDEBAR – FILTERS
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏙️ Parcl Analytics")
    st.markdown("**Buyer Segmentation Suite**")
    st.divider()
    st.markdown("### 🔍 Filter Controls")

    all_countries = sorted(df_full["country"].unique())
    sel_countries = st.multiselect("🌍 Country", all_countries, default=all_countries)

    reg_avail  = sorted(df_full[df_full["country"].isin(sel_countries)]["region"].unique())
    sel_regions = st.multiselect("📍 Region", reg_avail, default=reg_avail)

    purposes    = sorted(df_full["acquisition_purpose"].unique())
    sel_purposes = st.multiselect("🎯 Acquisition Purpose", purposes, default=purposes)

    ctypes     = sorted(df_full["client_type"].unique())
    sel_ctypes = st.multiselect("👤 Client Type", ctypes, default=ctypes)

    st.divider()
    st.caption("k=4 used in pipeline (optimal by Elbow + Silhouette)")
    st.divider()
    st.markdown("<small style='color:#8B949E'>Parcl · Data Science Internship<br>"
                "Pipeline: Cleaning → Encoding → Scaling → K-Means</small>",
                unsafe_allow_html=True)

mask = (
    df_full["country"].isin(sel_countries) &
    df_full["region"].isin(sel_regions)    &
    df_full["acquisition_purpose"].isin(sel_purposes) &
    df_full["client_type"].isin(sel_ctypes)
)
df = df_full[mask].copy()


# ──────────────────────────────────────────────────────────────
# HEADER + KPIs
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <h1>🏙️ Parcl Buyer Segmentation</h1>
  <p>Data-driven intelligence on property buyer types, investment patterns &amp; geographic behavior</p>
</div>
""", unsafe_allow_html=True)

buyers_all = df[df["total_investment"] > 0]
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Clients",     f"{len(df):,}")
k2.metric("Active Buyers",     f"{len(buyers_all):,}")
k3.metric("Total Investment",  f"${buyers_all['total_investment'].sum()/1e6:.1f}M")
k4.metric("Avg Satisfaction",  f"{df['satisfaction_score'].mean():.2f} / 5")
k5.metric("Loan Take-up Rate", f"{(df['loan_applied']=='Yes').mean()*100:.1f}%")
st.divider()


# ──────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Segmentation Overview",
    "💰  Investor Behavior",
    "🌍  Geographic Analysis",
    "🔬  Segment Insights",
])


# ════════════════════════════════════════════════════════════════
# TAB 1 – SEGMENTATION OVERVIEW
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Buyer Cluster Distribution</div>', unsafe_allow_html=True)

    seg_counts        = df["segment"].value_counts().reset_index()
    seg_counts.columns = ["segment","count"]
    seg_counts["color"] = seg_counts["segment"].map(SEG_COLORS)

    c1, c2 = st.columns(2)

    # Donut
    fig_donut = go.Figure(go.Pie(
        labels=seg_counts["segment"], values=seg_counts["count"], hole=0.62,
        marker=dict(colors=seg_counts["color"].tolist(),
                    line=dict(color="#0D1117", width=2)),
        textinfo="percent+label", textfont=dict(size=12, color="#E6EDF3"),
        hovertemplate="<b>%{label}</b><br>Clients: %{value:,}<br>Share: %{percent}<extra></extra>",
    ))
    tl(fig_donut, title="Segment Distribution", height=360,
       annotations=[dict(text=f"<b>{len(df):,}</b><br>clients",
                         x=0.5, y=0.5, font_size=16, showarrow=False,
                         font_color="#E6EDF3")],
       legend=dict(orientation="v", x=1, y=0.5))
    c1.plotly_chart(fig_donut, use_container_width=True)

    # Horizontal bar – note: yaxis update done via update_yaxes, NOT inside tl()
    fig_bar = go.Figure(go.Bar(
        x=seg_counts["count"], y=seg_counts["segment"], orientation="h",
        marker_color=seg_counts["color"].tolist(),
        text=seg_counts["count"].apply(lambda x: f"{x:,}"),
        textposition="outside", textfont=dict(color="#E6EDF3"),
        hovertemplate="<b>%{y}</b><br>Clients: %{x:,}<extra></extra>",
    ))
    tl(fig_bar, title="Client Count by Segment", xaxis_title="Number of Clients", height=360)
    fig_bar.update_yaxes(categoryorder="total ascending")   # safe – no conflict
    c2.plotly_chart(fig_bar, use_container_width=True)

    # PCA scatter
    st.markdown('<div class="section-title">PCA 2D Cluster Projection</div>', unsafe_allow_html=True)
    fig_pca = px.scatter(
        df, x="pc1", y="pc2", color="segment",
        color_discrete_map=SEG_COLORS, opacity=0.65,
        hover_data={"pc1": False, "pc2": False,
                    "age": True, "satisfaction_score": True,
                    "total_investment": ":.0f", "country": True},
        labels={"pc1": f"PC1 ({pca_ev[0]*100:.1f}% var)",
                "pc2": f"PC2 ({pca_ev[1]*100:.1f}% var)"},
        title="K-Means Clusters in PCA Space",
    )
    fig_pca.update_traces(marker=dict(size=6, line=dict(width=0)))
    tl(fig_pca, height=480, legend=dict(orientation="h", y=-0.12))
    st.plotly_chart(fig_pca, use_container_width=True)

    # Segment cards
    st.markdown('<div class="section-title">Segment Profiles at a Glance</div>', unsafe_allow_html=True)
    icons = {"Luxury Investors":"💎","Global Investors":"🌍",
             "Corporate Buyers":"🏢","First-Time Buyers":"🏠"}
    card_cols = st.columns(4)
    for col, seg in zip(card_cols,
                        ["Luxury Investors","Global Investors","Corporate Buyers","First-Time Buyers"]):
        sub = df[df["segment"] == seg]
        if len(sub) == 0:
            col.info(f"{seg} — no data after filter"); continue
        c_hex   = SEG_COLORS[seg]
        b_sub   = sub[sub["total_investment"] > 0]
        avg_inv = b_sub["total_investment"].mean() if len(b_sub) else float("nan")
        inv_str = f"${avg_inv:,.0f}" if not np.isnan(avg_inv) else "N/A"
        col.markdown(f"""
        <div style="background:#161B22;border:1.5px solid {c_hex};border-radius:14px;
                    padding:1.1rem;text-align:center">
          <div style="font-size:2rem">{icons[seg]}</div>
          <div style="color:{c_hex};font-weight:700;font-size:.95rem;margin:.3rem 0">{seg}</div>
          <div style="color:#E6EDF3;font-size:1.3rem;font-weight:700">{len(sub):,}</div>
          <div style="color:#8B949E;font-size:.75rem">clients</div>
          <hr style="border-color:#21262D;margin:.6rem 0">
          <div style="font-size:.78rem;color:#C9D1D9">
            Avg Age: <b>{sub['age'].mean():.1f}</b><br>
            Satisfaction: <b>{sub['satisfaction_score'].mean():.2f}/5</b><br>
            Avg Investment: <b>{inv_str}</b>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 2 – INVESTOR BEHAVIOR
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Investment Patterns by Segment</div>', unsafe_allow_html=True)

    buyers = df[df["total_investment"] > 0]
    c1, c2 = st.columns(2)

    fig_inv = px.box(
        buyers, x="segment", y="total_investment",
        color="segment", color_discrete_map=SEG_COLORS,
        title="Investment Distribution per Segment",
        labels={"total_investment":"Total Investment (USD)","segment":""},
        points="outliers",
    )
    fig_inv.update_traces(quartilemethod="exclusive")
    fig_inv.update_yaxes(tickprefix="$", tickformat=".2s")
    tl(fig_inv, height=400, showlegend=False)
    c1.plotly_chart(fig_inv, use_container_width=True)

    fig_area = px.violin(
        buyers, x="segment", y="avg_unit_area",
        color="segment", color_discrete_map=SEG_COLORS,
        box=True, points=False,
        title="Avg Unit Area per Segment (sqft)",
        labels={"avg_unit_area":"Avg Unit Area (sqft)","segment":""},
    )
    tl(fig_area, height=400, showlegend=False)
    c2.plotly_chart(fig_area, use_container_width=True)

    st.markdown('<div class="section-title">Financing & Purchase Purpose</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    loan_df = df.groupby(["segment","loan_applied"]).size().reset_index(name="count")
    fig_loan = px.bar(
        loan_df, x="segment", y="count", color="loan_applied",
        barmode="group", color_discrete_sequence=["#4361EE","#F72585"],
        title="Loan Applied by Segment",
        labels={"count":"Clients","loan_applied":"Loan Applied","segment":""},
    )
    tl(fig_loan, height=380, legend=dict(orientation="h", y=-0.18))
    c1.plotly_chart(fig_loan, use_container_width=True)

    purp_df = df.groupby(["segment","acquisition_purpose"]).size().reset_index(name="count")
    fig_purp = px.bar(
        purp_df, x="segment", y="count", color="acquisition_purpose",
        barmode="stack", color_discrete_sequence=["#7209B7","#F3722C"],
        title="Acquisition Purpose by Segment",
        labels={"count":"Clients","acquisition_purpose":"Purpose","segment":""},
    )
    tl(fig_purp, height=380, legend=dict(orientation="h", y=-0.18))
    c2.plotly_chart(fig_purp, use_container_width=True)

    st.markdown('<div class="section-title">Age vs Investment Value</div>', unsafe_allow_html=True)
    fig_scat = px.scatter(
        buyers, x="age", y="total_investment",
        color="segment", color_discrete_map=SEG_COLORS,
        size="avg_unit_area", size_max=18, opacity=0.7,
        hover_data={"country":True,"satisfaction_score":True,
                    "num_properties":True,"avg_unit_area":":.0f"},
        title="Age vs Total Investment  (bubble = unit area)",
        labels={"age":"Age (years)","total_investment":"Total Investment (USD)"},
    )
    fig_scat.update_yaxes(tickprefix="$", tickformat=".2s")
    tl(fig_scat, height=480, legend=dict(orientation="h", y=-0.12))
    st.plotly_chart(fig_scat, use_container_width=True)

    st.markdown('<div class="section-title">Referral Channel Mix</div>', unsafe_allow_html=True)
    ref_df = df.groupby(["segment","referral_channel"]).size().reset_index(name="count")
    ref_df["pct"] = ref_df["count"] / ref_df.groupby("segment")["count"].transform("sum") * 100
    fig_ref = px.bar(
        ref_df, x="segment", y="pct", color="referral_channel",
        barmode="stack", color_discrete_sequence=["#4CC9F0","#90BE6D","#F3722C"],
        title="Referral Channel Share by Segment (%)",
        labels={"pct":"Share (%)","referral_channel":"Channel","segment":""},
    )
    tl(fig_ref, height=380, legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fig_ref, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 3 – GEOGRAPHIC ANALYSIS
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Buyer Segments by Country</div>', unsafe_allow_html=True)

    country_seg = df.groupby(["country","segment"]).size().reset_index(name="count")
    dominant    = (country_seg.loc[country_seg.groupby("country")["count"].idxmax()]
                   [["country","segment"]].rename(columns={"segment":"dominant_segment"}))
    country_tot = df.groupby("country").agg(
        total_clients    =("client_id","count"),
        avg_investment   =("total_investment","mean"),
        avg_satisfaction =("satisfaction_score","mean"),
    ).reset_index().merge(dominant, on="country")

    iso3 = {"Usa":"USA","Canada":"CAN","Uk":"GBR","Germany":"DEU",
            "France":"FRA","Australia":"AUS","Belgium":"BEL",
            "Mexico":"MEX","Russia":"RUS","Denmark":"DNK"}
    country_tot["iso3"] = country_tot["country"].map(iso3)

    fig_map = px.choropleth(
        country_tot, locations="iso3", color="total_clients",
        hover_name="country",
        hover_data={"iso3":False,"total_clients":True,
                    "avg_investment":":,.0f","avg_satisfaction":":.2f",
                    "dominant_segment":True},
        color_continuous_scale=[[0,"#1C2128"],[0.25,"#3A0CA3"],[0.6,"#4361EE"],[1,"#4CC9F0"]],
        title="Client Count by Country",
        labels={"total_clients":"Clients"},
    )
    fig_map.update_geos(
        bgcolor="#0D1117", showland=True, landcolor="#1C2128",
        showocean=True, oceancolor="#0D1117",
        showframe=False, coastlinecolor="#30363D", countrycolor="#30363D",
    )
    # Choropleth: use update_layout directly (no xaxis/yaxis conflict here)
    fig_map.update_layout(
        **_BASE, height=480,
        geo=dict(bgcolor="#0D1117"),
        coloraxis_colorbar=dict(title="Clients"),
        margin=dict(l=20, r=20, t=45, b=20),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown('<div class="section-title">Segment Distribution by Country</div>', unsafe_allow_html=True)
    fig_cbar = px.bar(
        country_seg, x="country", y="count", color="segment",
        color_discrete_map=SEG_COLORS, barmode="stack",
        title="Buyer Segments per Country",
        labels={"count":"Clients","segment":"Segment","country":"Country"},
    )
    tl(fig_cbar, height=420, legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fig_cbar, use_container_width=True)

    st.markdown('<div class="section-title">Regional Breakdown (Top 30 Regions)</div>', unsafe_allow_html=True)
    reg_df      = df.groupby(["country","region","segment"]).size().reset_index(name="count")
    top_regions = reg_df.groupby("region")["count"].sum().nlargest(30).index
    reg_top     = reg_df[reg_df["region"].isin(top_regions)]

    fig_tree = px.treemap(
        reg_top,
        path=[px.Constant("World"),"country","region","segment"],
        values="count", color="segment",
        color_discrete_map={"(?)":"#21262D", **SEG_COLORS},
        title="Buyer Segments – Country → Region → Segment",
    )
    fig_tree.update_traces(textfont=dict(color="#E6EDF3"),
                           marker=dict(line=dict(color="#0D1117", width=1.5)))
    fig_tree.update_layout(**_BASE, height=540, margin=dict(l=20,r=20,t=45,b=20))
    st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown('<div class="section-title">Investment Purpose Heatmap</div>', unsafe_allow_html=True)
    pur_heat = pd.crosstab(df["country"], df["acquisition_purpose"])
    fig_heat = go.Figure(go.Heatmap(
        z=pur_heat.values,
        x=pur_heat.columns.tolist(),
        y=pur_heat.index.tolist(),
        colorscale=[[0,"#161B22"],[0.5,"#3A0CA3"],[1,"#4CC9F0"]],
        text=pur_heat.values, texttemplate="%{text}",
        textfont={"size":12,"color":"#E6EDF3"},
        hovertemplate="Country: %{y}<br>Purpose: %{x}<br>Clients: %{z}<extra></extra>",
    ))
    tl(fig_heat, title="Clients: Country × Acquisition Purpose", height=400)
    st.plotly_chart(fig_heat, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 4 – SEGMENT INSIGHTS
# ════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Descriptive Statistics per Segment</div>', unsafe_allow_html=True)

    sel_seg    = st.selectbox("Select Segment", list(SEG_COLORS.keys()))
    sub        = df[df["segment"] == sel_seg]

    if len(sub) == 0:
        st.warning("No clients match the current filter for this segment.")
        st.stop()

    c_hex      = SEG_COLORS[sel_seg]
    buyers_sub = sub[sub["total_investment"] > 0]
    avg_inv    = buyers_sub["total_investment"].mean() if len(buyers_sub) else float("nan")

    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Clients in Segment", f"{len(sub):,}")
    kc2.metric("Avg Age",            f"{sub['age'].mean():.1f} yrs")
    kc3.metric("Avg Satisfaction",   f"{sub['satisfaction_score'].mean():.2f}")
    kc4.metric("Loan Rate",          f"{(sub['loan_applied']=='Yes').mean()*100:.1f}%")

    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric("Active Buyers",    f"{len(buyers_sub):,}")
    bc2.metric("Avg Investment",   f"${avg_inv:,.0f}" if not np.isnan(avg_inv) else "N/A")
    bc3.metric("Avg # Properties", f"{sub['num_properties'].mean():.2f}")
    bc4.metric("Avg Unit Area",    f"{sub['avg_unit_area'].mean():.0f} sqft")

    st.divider()
    st.markdown("**Full Descriptive Statistics**")
    stats_cols = ["age","satisfaction_score","num_properties","total_investment","avg_unit_area"]
    st.dataframe(
        sub[stats_cols].describe().round(2).T
            .style.background_gradient(cmap="Blues", subset=["mean","50%"])
                  .format("{:.2f}"),
        use_container_width=True,
    )

    st.divider()
    c1, c2 = st.columns(2)

    fig_age = px.histogram(
        sub, x="age", nbins=20, color_discrete_sequence=[c_hex],
        title=f"Age Distribution – {sel_seg}",
        labels={"age":"Age (years)"}, opacity=0.85,
    )
    fig_age.add_vline(x=sub["age"].median(), line_dash="dash", line_color="#F72585",
                      annotation_text=f"Median: {sub['age'].median():.1f}",
                      annotation_font_color="#F72585")
    tl(fig_age, height=360)
    c1.plotly_chart(fig_age, use_container_width=True)

    sat_vc = sub["satisfaction_score"].value_counts().sort_index().reset_index()
    sat_vc.columns = ["score","count"]
    fig_sat = px.bar(sat_vc, x="score", y="count",
                     color_discrete_sequence=[c_hex],
                     title=f"Satisfaction Score – {sel_seg}",
                     labels={"score":"Score","count":"Clients"}, opacity=0.85)
    tl(fig_sat, height=360)
    c2.plotly_chart(fig_sat, use_container_width=True)

    if len(buyers_sub):
        st.markdown('<div class="section-title">Investment & Property Detail</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        fig_tinv = px.histogram(
            buyers_sub, x="total_investment", nbins=25,
            color_discrete_sequence=[c_hex],
            title=f"Total Investment Distribution – {sel_seg}",
            labels={"total_investment":"Total Investment (USD)"}, opacity=0.85,
        )
        fig_tinv.update_xaxes(tickprefix="$", tickformat=".2s")
        tl(fig_tinv, height=360)
        c1.plotly_chart(fig_tinv, use_container_width=True)

        fig_nprp = px.histogram(
            buyers_sub, x="num_properties", nbins=10,
            color_discrete_sequence=[c_hex],
            title=f"# Properties Purchased – {sel_seg}",
            labels={"num_properties":"Number of Properties"}, opacity=0.85,
        )
        tl(fig_nprp, height=360)
        c2.plotly_chart(fig_nprp, use_container_width=True)

    st.markdown('<div class="section-title">Demographics & Referral</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    for col_w, field, title in zip([p1, p2, p3],
                                   ["gender","referral_channel","country"],
                                   ["Gender Split","Referral Channel","Country Mix"]):
        vc = sub[field].value_counts().reset_index()
        vc.columns = [field,"count"]
        fig_p = px.pie(vc, names=field, values="count",
                       color_discrete_sequence=PIE_COLORS, title=title, hole=0.45)
        fig_p.update_traces(textinfo="percent+label",
                            textfont=dict(size=10, color="#E6EDF3"),
                            marker=dict(line=dict(color="#0D1117", width=2)))
        fig_p.update_layout(**_BASE, height=320, showlegend=False,
                            margin=dict(l=10,r=10,t=40,b=10))
        col_w.plotly_chart(fig_p, use_container_width=True)

    st.markdown('<div class="section-title">Client Data Explorer</div>', unsafe_allow_html=True)
    show_cols = ["client_id","first_name","last_name","age","country","region",
                 "acquisition_purpose","loan_applied","satisfaction_score",
                 "num_properties","total_investment","avg_unit_area","referral_channel"]
    st.dataframe(
        sub[show_cols].sort_values("total_investment", ascending=False).reset_index(drop=True),
        use_container_width=True, height=380,
    )
    st.download_button(
        label=f"⬇️  Export {sel_seg} data as CSV",
        data=sub[show_cols].to_csv(index=False),
        file_name=f"parcl_{sel_seg.lower().replace(' ','_')}.csv",
        mime="text/csv",
    )