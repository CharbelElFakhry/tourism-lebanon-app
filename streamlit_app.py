# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
from urllib.parse import urlsplit, unquote
import plotly.express as px

st.set_page_config(page_title="Tourism in Lebanon — Story Dashboard", layout="wide")

st.title("How Restaurants Shape Tourism in Lebanon?")

st.markdown("""
## Introduction

What do we remember every single Lebanese summer?  
**Mountains, beaches, food, and festivals!**  
And of course, the restaurants : each season they invest a lot in marketing, 
launch new locations, reinvent their menus, and craft irresistible offers.  

But have we ever stopped to ask: **what is the real impact of restaurants on a town’s tourism?**  

In this application, using Lebanon’s tourism dataset, we explore **how restaurants shape the tourism performance of towns and regions**.  
Through interactive visualizations, we identify the **Top restaurant hubs**, analyze their **tourism scores**, and compare patterns across **districts and governorates**.  
The goal is to uncover interesting insights and start a discussion on the role of restaurants within Lebanon’s broader tourism story.
""")
# Part 0
st.markdown("""
## 0) About the Dataset

Before diving into the story, let’s briefly look at the dataset we are using.  
It was created by **Christian Gharzouzi** on *September 4, 2024* and is linked to the dataset 
**Tourism-Lebanon-2023**.

**Dataset details:**
- **Issued on:** 2024-09-04  
- **Scope:** Tourism data for Lebanon at the **Governorate, District, and Town** levels  
- **Shape:** (1137 rows × 22 columns)  
- **Measures included:**  
  - Total number of restaurants, hotels, cafes, and guest houses  
  - Existence of tourism attractions that could be exploited or developed  
  - Existence of initiatives and projects in the past five years to improve the tourism sector  
  - Indicators showing whether restaurants, hotels, cafes, or guest houses exist in a given town  

This dataset gives us a structured way to explore **how infrastructure and services 
are distributed across Lebanon’s towns and regions**, and how they might relate 
to tourism performance.
""")

# Part 1: Top Towns by Restaurants
# Loading the dataset
df = pd.read_csv("Tourism in Lebanon.csv")

with st.expander("📂 Click here to preview the dataset"):
    st.dataframe(df, use_container_width=True)

st.subheader("1) Top Towns by Number of Restaurants")
st.markdown("""
Lebanon has many towns, so we’ll keep things simple and start with the **Top 10 towns by number of restaurants**.  
This gives a quick sense of **where dining capacity is concentrated** and sets up our story about how restaurants relate to tourism performance.
""")

def clean_ref_area(v):
    if pd.isna(v):
        return pd.NA
    s = str(v)
    try:
        path = urlsplit(s).path or s
    except Exception:
        path = s
    seg = path.rsplit('/', 1)[-1]
    seg = seg.split('?', 1)[0].split('#', 1)[0]
    seg = unquote(seg).replace('_', ' ').replace('-', ' ').strip()
    return seg or pd.NA

df["refArea_clean"] = df["refArea"].apply(clean_ref_area)

# Top-N control
top_n = st.slider("Select number of top towns to display", min_value=5, max_value=20, value=10, step=1)

# compute Top-N dynamically (based on total restaurants)
top_rest = (
    df.groupby(["Town", "refArea_clean"], as_index=False)["Total number of restaurants"]
      .sum()
      .sort_values("Total number of restaurants", ascending=False)
      .head(top_n)
)

# Guard: if nothing to plot
if top_rest.empty:
    st.info("No data to display for the current selection.")
else:
    x = top_rest["Town"].tolist()
    y_true = top_rest["Total number of restaurants"].astype(float).values
    y_max = float(y_true.max())
    y0 = np.full_like(y_true, y_max, dtype=float)
    y_pad = y_max * 1.1 if y_max > 0 else 1.0

    # animated bar chart 
    N = 10  
    frames = []
    for t in np.linspace(0, 1, N):
        y_t = y0 - t * (y0 - y_true)
        frames.append(go.Frame(name=f"f{t:.2f}",
                               data=[go.Bar(x=x, y=y_t, marker_color="#1f60b4")]))

    fig = go.Figure(
        data=[go.Bar(x=x, y=y0, marker_color="#1f60b4")],
        frames=frames,
    )

    fig.update_layout(
        title=f"Top {top_n} Towns by Number of Restaurants",
        xaxis_title="Town",
        yaxis_title="Total number of restaurants",
        yaxis=dict(range=[0, y_pad]),
        showlegend=False,
        margin=dict(l=60, r=20, t=70, b=60),
        height=520,
        updatemenus=[]  
    )
    fig.update_xaxes(tickangle=-20)

    html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        auto_play=True,
        animation_opts={
            "frame": {"duration": 35, "redraw": True},  
            "transition": {"duration": 0},
            "fromcurrent": True,
            "mode": "immediate"
        },
    )
    components.html(html, height=560, scrolling=False)

st.markdown("""
### Analysis

Looking at the **Top 10 towns by number of restaurants**, we immediately notice a **large variability**:  
- The leading town hosts **100 restaurants**,  
- While the 10th-ranked town has only about **40**.  

This gap shows how **unevenly distributed** restaurants are across Lebanese towns.  
And if we zoom out to consider **all towns in Lebanon**, the variability in restaurant numbers 
is even more hilighted, ranging from towns with dozens of restaurants to others with very few or none at all.
""")

st.markdown("### Which areas do the Top 10 towns belong to?")

# Count how many of the top 10 towns fall into each area
area_counts = top_rest["refArea_clean"].value_counts().reset_index()
area_counts.columns = ["Area", "Number of Top Towns"]

# Plot as a horizontal bar chart
fig_area = px.bar(
    area_counts,
    y="Area",
    x="Number of Top Towns",
    text="Number of Top Towns",
    orientation="h",
    color="Area",
    title="Areas Represented in the Top 10 Towns by Restaurants"
)
fig_area.update_layout(
    height=420,
    margin=dict(l=100, r=20, t=60, b=40),
    xaxis_title="Number of Top Towns",
    yaxis_title="Area",
    showlegend=False
)
st.plotly_chart(fig_area, use_container_width=True)

st.markdown("""
### Analysis

From the distribution of areas among the **Top 10 towns by restaurants**, 
we see that **3 of these towns belong to Baabda District**.  

This shows that **Baabda stands out geographically**, as multiple top-performing towns 
are located within the same district.  

It suggests that restaurants (and potentially tourism activity) are not only concentrated 
in specific towns, but also **cluster within certain districts**, which could make these areas even more attractive to tourists.""")



# Part 2: Restaurants vs Tourism Index 

st.subheader("2) What is the relationship between Restaurants and Tourism Index?")
st.markdown("""
After reviewing the **top towns by restaurants** and the **areas they belong to**, 
it’s time to look at the **relationship between the total number of restaurants and the Tourism Index**. 
Seeing this link can help us understand whether places with more dining options also tend to perform better in tourism.
""")

# prepare clean arrays 
mask = df[["Total number of restaurants", "Tourism Index", "Town"]].dropna().index
x = df.loc[mask, "Total number of restaurants"].to_numpy(dtype=float)
y = df.loc[mask, "Tourism Index"].to_numpy(dtype=float)
town = df.loc[mask, "Town"].to_numpy()

# jitter to reduce overlap + sort by x for better line animation
rng = np.random.default_rng(42)
xj = x + rng.uniform(-0.2, 0.2, size=len(x))
yj = y + rng.uniform(-0.2, 0.2, size=len(y))
order = np.argsort(xj)
xj, yj, town = xj[order], yj[order], town[order]

# data-range (concise) for line & axes 
if len(xj):
    x_min, x_max = float(xj.min()), float(xj.max())
    y_min, y_max = float(yj.min()), float(yj.max())
    # small padding for nicer view
    x_pad = 0.05 * max(1.0, x_max - x_min)
    y_pad = 0.10 * max(1.0, y_max - y_min)
    x_range = [x_min - x_pad, x_max + x_pad]
    y_range = [y_min - y_pad, y_max + y_pad]
else:
    x_range, y_range = [0, 1], [0, 1]

# common style (no xaxis/yaxis here) 
COMMON_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(color="#0b2e4f"),
    margin=dict(l=60, r=20, t=70, b=60),
    height=520,
    showlegend=False
)

base_scatter = go.Scatter(
    x=xj, y=yj, mode="markers",
    marker=dict(size=8, opacity=0.6, color="#1f60b4", line=dict(width=1, color="white")),
    hoverinfo="text", text=town, showlegend=False
)

show_line = st.checkbox("Show regression line", value=False)

if not show_line or len(xj) < 2:
    # STATIC
    fig = go.Figure(data=[base_scatter])
    fig.update_layout(
        title="Restaurants vs Tourism Index",
        xaxis_title="Total number of restaurants",
        yaxis_title="Tourism Index",
        **COMMON_LAYOUT
    )
    fig.update_xaxes(range=x_range, zeroline=False, gridcolor="#eaeaea")
    fig.update_yaxes(range=y_range, zeroline=False, gridcolor="#eaeaea")
    st.plotly_chart(fig, use_container_width=True, theme=None)

else:
    # REGRESSION + ANIMATION
    m, b = np.polyfit(xj, yj, 1)
    x_line = np.linspace(float(xj.min()), float(xj.max()), 200)
    y_line = m * x_line + b

    frames = []
    step = 20
    for j in range(step, len(x_line) + 1, step):
        frames.append(go.Frame(
            name=f"line_{j}",
            data=[
                base_scatter,
                go.Scatter(x=x_line[:j], y=y_line[:j], mode="lines",
                           line=dict(width=3, color="#0b2e4f"))
            ]
        ))

    final_line = go.Scatter(x=x_line, y=y_line, mode="lines",
                            line=dict(width=3, color="#0b2e4f"))
    for _ in range(8):
        frames.append(go.Frame(name="hold", data=[base_scatter, final_line]))

    fig = go.Figure(
        data=[base_scatter, go.Scatter(x=[], y=[], mode="lines",
                                       line=dict(width=3, color="#0b2e4f"))],
        frames=frames
    )
    fig.update_layout(
        title="Restaurants vs Tourism Index (regression reveals on click)",
        xaxis_title="Total number of restaurants",
        yaxis_title="Tourism Index",
        updatemenus=[],
        **COMMON_LAYOUT
    )
    fig.update_xaxes(range=x_range, zeroline=False, gridcolor="#eaeaea")
    fig.update_yaxes(range=y_range, zeroline=False, gridcolor="#eaeaea")

    html = fig.to_html(
        include_plotlyjs="cdn", full_html=False, auto_play=True,
        animation_opts={
            "frame": {"duration": 45, "redraw": True},
            "transition": {"duration": 0},
            "fromcurrent": True, "mode": "immediate"
        },
    )
    components.html(html, height=560, scrolling=False)


st.markdown("""
### Analysis

From this visualization, we can see that **there is some positive relationship** between the 
**total number of restaurants in a town** and its **Tourism Index**.  
Although the correlation is not perfect, the regression line shows 
that towns with **more restaurants tend to score higher on tourism performance**.  

This supports the idea that a **nice food scene contributes to tourism appeal**. 
However, since variability remains high, it tells us that restaurants are **only one of several factors** 
that affect tourism in towns.
""")

# Part 3: Tourism Index Across Top Towns and Their Regions
st.subheader("3) Tourism Index Across Top Towns and Their Regions")
st.markdown("""
After seeing a link between **restaurant counts** and the **Tourism Index**, 
let’s check towns directly: compare all towns’ Tourism Index values, and 
**highlight our Top-10 towns by restaurants** to see if they generally sit higher.
""")

# Slider: minimum restaurants per town 
# compute total restaurants per town 
town_totals = (
    df.groupby(["Town", "refArea_clean"], as_index=False)["Total number of restaurants"]
      .sum()
      .rename(columns={"Total number of restaurants": "RestaurantsTotal"})
)

max_rest = int(town_totals["RestaurantsTotal"].max())
min_rest = st.slider(
    "Minimum restaurants per town to include",
    min_value=0, max_value=max_rest, value=0, step=5
)

# towns that pass the threshold
kept_towns = town_totals.loc[town_totals["RestaurantsTotal"] >= min_rest, "Town"].tolist()
base_df = df[df["Town"].isin(kept_towns)].copy()

# Top-10 by restaurants (for highlighting/filter toggle)
top10 = (
    town_totals.sort_values("RestaurantsTotal", ascending=False)
               .head(10)["Town"].tolist()
)

# Toggle: show only top-restaurant towns 
only_top = st.checkbox("Show only Top-10 restaurant towns", value=False)

if only_top:
    plot_df = base_df[base_df["Town"].isin(top10)].copy()
else:
    plot_df = base_df.copy()

# order towns by median Tourism Index in the current view
if not plot_df.empty:
    town_order = (plot_df.groupby("Town")["Tourism Index"]
                         .median()
                         .sort_values()
                         .index.tolist())
else:
    town_order = []

# building the figure 
if plot_df.empty:
    st.info("No towns match the current filters.")
else:
    if not only_top:
        fig_towns = px.strip(
            plot_df, x="Town", y="Tourism Index",
            color_discrete_sequence=["#AEB7BF"],  
        )
        fig_towns.update_traces(jitter=0.3, marker_opacity=0.35, marker_size=6)
    else:
        # if only top towns: start with an empty fig, we’ll add colored layer next
        fig_towns = px.strip(plot_df.iloc[0:0], x="Town", y="Tourism Index")

    # overlay: highlight top-restaurant towns (within current view)
    hl_df = plot_df[plot_df["Town"].isin(top10)].copy()
    if not hl_df.empty:
        fig_hl = px.strip(
            hl_df, x="Town", y="Tourism Index", color="Town",
            hover_data=["refArea_clean"],
        )
        fig_hl.update_traces(
            jitter=0.3, marker_size=10,
            marker_line_width=1, marker_line_color="white", opacity=0.95
        )
        for tr in fig_hl.data:
            fig_towns.add_trace(tr)

    # layout & ordering
    fig_towns.update_layout(
        title=(
            "Tourism Index across Towns"
            + (" — Top-10 by Restaurants highlighted" if not only_top else " — Top-10 by Restaurants")
        ),
        xaxis_title="Town", yaxis_title="Tourism Index",
        margin=dict(l=60, r=20, t=70, b=120),
        showlegend=False,
        template="plotly_white",
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color="#0b2e4f"),
    )
    if town_order:
        fig_towns.update_xaxes(categoryorder="array", categoryarray=town_order, tickangle=45)
    fig_towns.update_yaxes(gridcolor="#eaeaea")
    fig_towns.update_xaxes(gridcolor="#f3f3f3")

    st.plotly_chart(fig_towns, use_container_width=True, theme=None)

    # caption
    st.caption(
        f"Showing towns with ≥ **{min_rest}** restaurants. "
        + ("Only Top-10 restaurant towns are displayed." if only_top else "Top-10 towns are highlighted in color.")
    )

st.markdown("""
### Analysis

As we look over our **Top 10 highlighted towns**, we notice that:  
- **2 towns** cluster around a Tourism Index of **6**,  
- **5 towns** score close to **9**,  
- and the **remaining 3 towns** reach the maximum of **10**.  

This affirms that while **most of the Top 10 towns align with higher tourism scores**, 
there is still some divergence — showing that **having many restaurants does not automatically 
guarantee stronger tourism outcomes**.
""")


# base data 
area_counts = df["refArea_clean"].value_counts(dropna=True)
keep_areas = area_counts[area_counts >= 10].index
dff_area = df[df["refArea_clean"].isin(keep_areas)].copy()

# regions that host any of the current Top-N restaurant towns
highlight_areas = sorted(top_rest["refArea_clean"].dropna().unique().tolist())

# toggle: show only highlighted regions
only_highlight = st.checkbox("Show only regions that contain Top-restaurant towns", value=False)

if only_highlight:
    plot_df = dff_area[dff_area["refArea_clean"].isin(highlight_areas)].copy()
else:
    plot_df = dff_area.copy()

if plot_df.empty:
    st.info("No regions match the current selection.")
else:
    # order areas by median Tourism Index in the current view
    order = (plot_df.groupby("refArea_clean")["Tourism Index"]
                       .median()
                       .sort_values()
                       .index.tolist())

    # tag for coloring (highlight vs other)
    plot_df["Region group"] = np.where(
        plot_df["refArea_clean"].isin(highlight_areas),
        "Contains Top-restaurant town(s)", "Other regions"
    )

    fig_area_dist = px.box(
        plot_df, x="refArea_clean", y="Tourism Index",
        color="Region group",
        category_orders={"refArea_clean": order},
        points="outliers",
        color_discrete_map={
            "Contains Top-restaurant town(s)": "#e67e22",  
            "Other regions": "#1f60b4"                     
        },
        title="Tourism Index Distribution by Region"
              + (" — highlighted regions only" if only_highlight else " (highlight shows regions with Top-restaurant towns)")
    )
    fig_area_dist.update_layout(
        xaxis_title="Area (Governorate / District)",
        yaxis_title="Tourism Index",
        template="plotly_white",
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color="#0b2e4f"),
        margin=dict(l=60, r=20, t=70, b=110),
        showlegend=not only_highlight 
    )
    fig_area_dist.update_xaxes(tickangle=45, gridcolor="#f3f3f3")
    fig_area_dist.update_yaxes(gridcolor="#eaeaea")

    st.plotly_chart(fig_area_dist, use_container_width=True, theme=None)

    st.caption(
        "Orange = regions that contain at least one Top-restaurant town."
        + (" Showing only highlighted regions." if only_highlight else " Toggle the checkbox to view only these regions.")
    )

st.markdown("""
### Analysis

On the **regional side**, when comparing districts and governorates, we check whether 
the **Top 10 restaurant towns** show the same relationship at a broader scale.  

What we quickly notice is that regions containing towns with many restaurants 
**do not always display uniform tourism performance**. Often there is just one 
outstanding town driving the numbers.  

For example:  
- **Byblos** and **Matn** districts appear early in the plot, **below the mid-range**, 
  showing relatively low Tourism Index performance.  
- **Baabda**, which contains three of the Top 10 towns, comes directly after Byblos and Matn, 
  also not among the top performers.  
- Only **Nabatieh Governorate** ranks close to the highest-performing region.  

This tells us that the **town-level relationship** (more restaurants → higher Tourism Index) 
does not necessarily translate to the **regional level**.  
A high number of restaurants in towns may be tied instead to **town-specific factors** 
such as marketing campaigns, festivals, or unique local attractions.
""")

st.markdown("""
## Conclusion

As this Lebanese summer comes to an end, so does our journey of exploring how **restaurants 
and tourism performance** connect across towns and regions.  

Here’s what we found:  
- Even within the Top 10 towns, the restaurant counts range widely, from nearly 100 in the top town down to around 40 in the last one.  
- Multiple of these towns are concentrated in just a few **districts**, indicating at geographic clustering.  
- At the **town level**, more restaurants often associated with a higher **Tourism Index**, 
  though not always. This shows that having restaurants alone doesn't guarantee success.  
- At the **regional level**, the pattern weakens: some districts with top towns (like Byblos, Matn, and Baabda) 
  low tourism indices, while others (like Nabatieh) perform well.  

In short, while Lebanon’s vibrant food scene is a key ingredient for tourism, the full recipe 
also depends on **many other factors** — such as marketing campaigns, seasonal festivals, 
infrastructure, or unique attractions. Some of these factors might not be captured in the dataset 
we used, but they still play a crucial role in shaping the country’s tourism story each summer.
""")

