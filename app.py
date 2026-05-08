import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="AI Model Intelligence Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #10a37f, #1a7fe8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #888; font-size: 1rem; margin-bottom: 1.5rem; }
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid #2a2a3e;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 6px 16px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    models = pd.read_csv("data/models.csv", parse_dates=["release_date"])
    benchmarks = pd.read_csv("data/benchmarks.csv")
    pricing = pd.read_csv("data/pricing.csv", parse_dates=["release_date"])
    merged = models.merge(benchmarks, on="name").merge(
        pricing[["name", "input_per_1m", "output_per_1m"]], on="name", how="left"
    )
    return models, benchmarks, pricing, merged


models, benchmarks, pricing, merged = load_data()

OPENAI_COLOR = "#10a37f"
ANTHROPIC_COLOR = "#c084fc"
GOOGLE_COLOR = "#4285F4"
META_COLOR = "#f97316"
COMPANY_COLORS = {
    "OpenAI": OPENAI_COLOR,
    "Anthropic": ANTHROPIC_COLOR,
    "Google": GOOGLE_COLOR,
    "Meta": META_COLOR,
}
BENCHMARK_LABELS = {
    "MMLU": "MMLU (General Knowledge)",
    "HumanEval": "HumanEval (Coding)",
    "MATH": "MATH (Mathematics)",
    "GPQA": "GPQA (PhD-level Science)",
    "MGSM": "MGSM (Multilingual Math)",
}

st.markdown('<p class="main-header">AI Model Intelligence Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Benchmarks · Pricing · Competitive Landscape · 2020 – 2025</p>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🏆 Benchmarks", "💰 Pricing Trends", "⚔️ Competitor Analysis", "🔬 Model Clustering"]
)


# ── TAB 1: OVERVIEW ────────────────────────────────────────────────────────────
with tab1:
    openai_models = models[models["company"] == "OpenAI"]
    best_mmlu = benchmarks.merge(models[["name", "company"]], on="name")
    best_mmlu_openai = best_mmlu[best_mmlu["company"] == "OpenAI"]["MMLU"].max()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("OpenAI Models Tracked", len(openai_models))
    col2.metric("Total Models in Dataset", len(models))
    col3.metric("Best OpenAI MMLU Score", f"{best_mmlu_openai:.1f}%")
    col4.metric(
        "Cheapest OpenAI Input Price",
        f"${pricing[pricing['company']=='OpenAI']['input_per_1m'].min():.2f}/1M",
    )

    st.markdown("---")
    st.subheader("Model Release Timeline")

    timeline_df = models.copy()
    timeline_df["year"] = timeline_df["release_date"].dt.year
    timeline_df["color"] = timeline_df["company"].map(COMPANY_COLORS)
    timeline_df["size"] = np.log1p(timeline_df["context_window_k"]) * 8

    fig_timeline = px.scatter(
        timeline_df,
        x="release_date",
        y="context_window_k",
        color="company",
        size="size",
        hover_name="name",
        hover_data={"release_date": True, "context_window_k": True, "type": True, "size": False},
        color_discrete_map=COMPANY_COLORS,
        labels={"release_date": "Release Date", "context_window_k": "Context Window (K tokens)"},
        log_y=True,
    )
    fig_timeline.update_layout(
        template="plotly_dark",
        legend_title="Company",
        height=420,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    st.subheader("OpenAI Model Specs at a Glance")
    display_cols = ["name", "release_date", "context_window_k", "params_B", "type"]
    st.dataframe(
        openai_models[display_cols]
        .sort_values("release_date")
        .rename(columns={
            "name": "Model", "release_date": "Released",
            "context_window_k": "Context (K)", "params_B": "Params (B)", "type": "Type",
        })
        .reset_index(drop=True),
        use_container_width=True,
    )


# ── TAB 2: BENCHMARKS ──────────────────────────────────────────────────────────
with tab2:
    st.subheader("Benchmark Score Comparison")
    benchmark_choice = st.selectbox(
        "Select benchmark",
        list(BENCHMARK_LABELS.keys()),
        format_func=lambda x: BENCHMARK_LABELS[x],
    )
    filter_company = st.multiselect(
        "Filter by company", options=models["company"].unique().tolist(),
        default=models["company"].unique().tolist(),
    )

    bench_df = benchmarks.merge(models[["name", "company", "release_date"]], on="name")
    bench_df = bench_df[bench_df["company"].isin(filter_company)].sort_values(benchmark_choice, ascending=True)

    fig_bar = px.bar(
        bench_df,
        x=benchmark_choice,
        y="name",
        color="company",
        orientation="h",
        color_discrete_map=COMPANY_COLORS,
        text=bench_df[benchmark_choice].apply(lambda v: f"{v:.1f}%"),
        labels={benchmark_choice: "Score (%)", "name": ""},
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        template="plotly_dark",
        height=520,
        showlegend=True,
        margin=dict(l=20, r=60, t=20, b=40),
        xaxis=dict(range=[0, 105]),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("OpenAI Benchmark Progress Over Time")
    openai_bench = benchmarks.merge(
        models[["name", "company", "release_date"]], on="name"
    ).query("company == 'OpenAI'").sort_values("release_date")

    fig_line = go.Figure()
    for bench, label in BENCHMARK_LABELS.items():
        fig_line.add_trace(go.Scatter(
            x=openai_bench["name"],
            y=openai_bench[bench],
            mode="lines+markers",
            name=label,
            line=dict(width=2),
            marker=dict(size=8),
        ))
    fig_line.update_layout(
        template="plotly_dark",
        yaxis_title="Score (%)",
        xaxis_title="Model (chronological)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Radar Chart — Top Models Head-to-Head")
    top_names = ["GPT-4o", "o1", "o3", "Claude 3.7 Sonnet", "Gemini 2.0 Flash", "Llama 3.1 405B"]
    radar_df = benchmarks[benchmarks["name"].isin(top_names)]
    categories = list(BENCHMARK_LABELS.keys())

    fig_radar = go.Figure()
    for _, row in radar_df.iterrows():
        company = models.loc[models["name"] == row["name"], "company"].values[0]
        fig_radar.add_trace(go.Scatterpolar(
            r=[row[c] for c in categories] + [row[categories[0]]],
            theta=categories + [categories[0]],
            fill="toself",
            name=row["name"],
            line_color=COMPANY_COLORS.get(company, "#888"),
            opacity=0.7,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        template="plotly_dark",
        height=480,
        margin=dict(l=60, r=60, t=40, b=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ── TAB 3: PRICING ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader("API Pricing Over Time (Input — per 1M tokens)")

    openai_pricing = pricing[pricing["company"] == "OpenAI"].sort_values("release_date")

    fig_price = px.line(
        openai_pricing,
        x="release_date",
        y="input_per_1m",
        markers=True,
        text="name",
        labels={"release_date": "Release Date", "input_per_1m": "Input Price ($/1M tokens)"},
        color_discrete_sequence=[OPENAI_COLOR],
    )
    fig_price.update_traces(textposition="top center", marker=dict(size=10))
    fig_price.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("Input vs Output Pricing — All Providers")
    pricing_display = pricing[pricing["input_per_1m"] > 0].copy()

    fig_scatter = px.scatter(
        pricing_display,
        x="input_per_1m",
        y="output_per_1m",
        color="company",
        size=np.ones(len(pricing_display)) * 15,
        hover_name="name",
        text="name",
        color_discrete_map=COMPANY_COLORS,
        labels={
            "input_per_1m": "Input Price ($/1M tokens)",
            "output_per_1m": "Output Price ($/1M tokens)",
        },
        log_x=True,
        log_y=True,
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(
        template="plotly_dark",
        height=440,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Cost-Efficiency: MMLU Score per Dollar (Input)")
    efficiency_df = merged.copy()
    efficiency_df = efficiency_df[efficiency_df["input_per_1m"] > 0]
    efficiency_df["mmlu_per_dollar"] = efficiency_df["MMLU"] / efficiency_df["input_per_1m"]

    fig_eff = px.bar(
        efficiency_df.sort_values("mmlu_per_dollar", ascending=False),
        x="name",
        y="mmlu_per_dollar",
        color="company",
        color_discrete_map=COMPANY_COLORS,
        labels={"mmlu_per_dollar": "MMLU Points / $1 Input", "name": ""},
        text=efficiency_df.sort_values("mmlu_per_dollar", ascending=False)["mmlu_per_dollar"].apply(
            lambda v: f"{v:.0f}"
        ),
    )
    fig_eff.update_traces(textposition="outside")
    fig_eff.update_layout(
        template="plotly_dark",
        height=420,
        xaxis_tickangle=-35,
        margin=dict(l=40, r=20, t=20, b=80),
    )
    st.plotly_chart(fig_eff, use_container_width=True)


# ── TAB 4: COMPETITOR ANALYSIS ─────────────────────────────────────────────────
with tab4:
    st.subheader("Benchmark Comparison: OpenAI vs Competitors")

    col_left, col_right = st.columns(2)
    openai_model = col_left.selectbox(
        "OpenAI model",
        models[models["company"] == "OpenAI"]["name"].tolist(),
        index=models[models["company"] == "OpenAI"]["name"].tolist().index("GPT-4o"),
    )
    competitor_model = col_right.selectbox(
        "Competitor model",
        models[models["company"] != "OpenAI"]["name"].tolist(),
        index=0,
    )

    compare_names = [openai_model, competitor_model]
    compare_df = benchmarks[benchmarks["name"].isin(compare_names)].set_index("name")

    fig_comp = go.Figure()
    for model_name in compare_names:
        company = models.loc[models["name"] == model_name, "company"].values[0]
        scores = [compare_df.loc[model_name, b] for b in BENCHMARK_LABELS]
        fig_comp.add_trace(go.Bar(
            name=model_name,
            x=list(BENCHMARK_LABELS.values()),
            y=scores,
            marker_color=COMPANY_COLORS.get(company, "#888"),
        ))
    fig_comp.update_layout(
        barmode="group",
        template="plotly_dark",
        yaxis_title="Score (%)",
        yaxis_range=[0, 105],
        height=400,
        margin=dict(l=40, r=20, t=20, b=60),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    col_a, col_b = st.columns(2)
    for col, model_name in zip([col_a, col_b], compare_names):
        row = merged[merged["name"] == model_name].iloc[0]
        company = row["company"]
        col.markdown(f"**{model_name}** ({company})")
        col.markdown(f"- Released: `{row['release_date'].strftime('%b %Y')}`")
        col.markdown(f"- Context: `{row['context_window_k']}K` tokens")
        col.markdown(f"- Input price: `${row['input_per_1m']:.2f}` / 1M tokens")
        avg_bench = np.mean([row[b] for b in BENCHMARK_LABELS])
        col.markdown(f"- Avg benchmark: `{avg_bench:.1f}%`")

    st.markdown("---")
    st.subheader("All Models — Average Benchmark Score")
    avg_df = benchmarks.copy()
    avg_df["avg_score"] = avg_df[list(BENCHMARK_LABELS.keys())].mean(axis=1)
    avg_df = avg_df.merge(models[["name", "company"]], on="name").sort_values("avg_score", ascending=False)

    fig_avg = px.bar(
        avg_df,
        x="name",
        y="avg_score",
        color="company",
        color_discrete_map=COMPANY_COLORS,
        text=avg_df["avg_score"].apply(lambda v: f"{v:.1f}%"),
        labels={"avg_score": "Average Score (%)", "name": ""},
    )
    fig_avg.update_traces(textposition="outside")
    fig_avg.update_layout(
        template="plotly_dark",
        height=420,
        xaxis_tickangle=-35,
        yaxis_range=[0, 110],
        margin=dict(l=40, r=20, t=20, b=80),
    )
    st.plotly_chart(fig_avg, use_container_width=True)


# ── TAB 5: CLUSTERING ──────────────────────────────────────────────────────────
with tab5:
    st.subheader("Model Clustering via K-Means + PCA")
    st.markdown(
        "Models are clustered by their benchmark profiles using K-Means. "
        "PCA reduces the 5 benchmark dimensions to 2 for visualization."
    )

    n_clusters = st.slider("Number of clusters (K)", min_value=2, max_value=5, value=3)

    features = list(BENCHMARK_LABELS.keys())
    X = benchmarks[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    cluster_df = benchmarks[["name"]].copy()
    cluster_df["Cluster"] = [f"Cluster {i+1}" for i in labels]
    cluster_df["PC1"] = X_pca[:, 0]
    cluster_df["PC2"] = X_pca[:, 1]
    cluster_df = cluster_df.merge(models[["name", "company"]], on="name")

    fig_cluster = px.scatter(
        cluster_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        symbol="company",
        text="name",
        hover_data={"PC1": False, "PC2": False, "Cluster": True, "company": True},
        labels={"PC1": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)",
                "PC2": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)"},
    )
    fig_cluster.update_traces(textposition="top center", marker=dict(size=10))
    fig_cluster.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.subheader("Cluster Composition")
    for i in range(n_clusters):
        members = cluster_df[cluster_df["Cluster"] == f"Cluster {i+1}"]["name"].tolist()
        avg_scores = benchmarks[benchmarks["name"].isin(members)][features].mean()
        with st.expander(f"Cluster {i+1} — {len(members)} models"):
            col_m, col_s = st.columns([1, 2])
            col_m.markdown("**Models:**\n" + "\n".join(f"- {m}" for m in members))
            fig_avg_radar = go.Figure(go.Scatterpolar(
                r=avg_scores.tolist() + [avg_scores.iloc[0]],
                theta=features + [features[0]],
                fill="toself",
                line_color="#10a37f",
            ))
            fig_avg_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                template="plotly_dark",
                height=280,
                margin=dict(l=40, r=40, t=20, b=20),
                showlegend=False,
            )
            col_s.plotly_chart(fig_avg_radar, use_container_width=True)

    st.subheader("Feature Importance (PCA Loadings)")
    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=["PC1", "PC2"],
    ).reset_index().rename(columns={"index": "Benchmark"})

    fig_load = px.bar(
        loadings.melt(id_vars="Benchmark", var_name="Component", value_name="Loading"),
        x="Benchmark",
        y="Loading",
        color="Component",
        barmode="group",
        labels={"Loading": "PCA Loading", "Benchmark": ""},
        color_discrete_sequence=["#10a37f", "#c084fc"],
    )
    fig_load.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_load, use_container_width=True)
