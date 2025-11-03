import os
import argparse
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

DATASET_INFO = {
    "CHAOS (2018)": {"accessible": 20, "countries": ["TR"]},
    "BTCV (2015)": {"accessible": 47, "countries": ["US"]},
    "Pancreas-CT (2015)": {"accessible": 42, "countries": ["US"]},
    "CT-ORG (2020)": {"accessible": 140, "countries": ["DE", "NL", "CA", "FR", "IL", "US"]},
    "WORD (2021)": {"accessible": 120, "countries": ["CN"]},
    "LiTS (2019)": {"accessible": 130, "countries": ["DE", "NL", "CA", "FR", "IL"]},
    "AMOS22 (2022)": {"accessible": 200, "countries": ["CN"]},
    "KiTS (2023)": {"accessible": 489, "countries": ["US"]},
    "AbdomenCT-1K (2021, 2023)": {"accessible": 1000, "countries": ["DE", "NL", "CA", "FR", "IL", "US", "CN"]},
    "MSD-CT (2021)": {"accessible": 945, "countries": ["US"]},
    "FLARE’23 (2022)": {"accessible": 4100, "countries": []},
    "Abdominal Trauma Det (2023)": {"accessible": 4711, "countries": ["CL", "DE", "ES", "TR", "AUS", "TH", "CN", "MA", "MT", "CA", "IE", "BR", "BA"]},
}

DATASET_INFO_TEST = {
    "Site A (UCSF)": {"accessible": 22000, "countries": ["US"]},
    "Site B (PH)": {"accessible": 4000, "countries": ["PL"]},
    "Site C (PUTH)": {"accessible": 8000, "countries": ["CN"]},
}

COUNTRY_MAP = {
    "US": "United States", "DE": "Germany", "NL": "Netherlands", "CA": "Canada", "FR": "France",
    "IL": "Israel", "IE": "Ireland", "BR": "Brazil", "BA": "Bosnia and Herzegovina", "CN": "China",
    "TR": "Turkey", "CH": "Switzerland", "AUS": "Australia", "TH": "Thailand",
    "CL": "Chile", "ES": "Spain", "MA": "Morocco", "MT": "Malta", "PL": "Poland",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize dataset contributions with world map and Sankey diagram")
    parser.add_argument('--map_output', type=str, default='world_map.png')
    parser.add_argument('--sankey_output', type=str, default='sankey_diagram.png')
    parser.add_argument('--sankey_output_test', type=str, default='sankey_diagram_test.png')
    parser.add_argument('--map_width', type=int, default=1000)
    parser.add_argument('--map_height', type=int, default=500)
    parser.add_argument('--sankey_width', type=int, default=450)
    parser.add_argument('--sankey_height', type=int, default=600)
    parser.add_argument('--fontsize', type=int, default=18)
    parser.add_argument('--map_color', type=float, nargs=3, default=[112/255, 48/255, 160/255])
    parser.add_argument('--map_color_test', type=float, nargs=3, default=[1.0, 0.0, 0.0])
    return parser.parse_args()

def build_dataframe(dataset_info, country_map):
    records = []
    for dataset, info in dataset_info.items():
        countries = info['countries']
        if countries:
            per_country = info['accessible'] // len(countries)
            for country in countries:
                records.append({
                    "dataset": dataset,
                    "country_code": country,
                    "country_name": country_map.get(country, country),
                    "accessible": per_country
                })
    return pd.DataFrame(records)

def plot_world_map(df1, df2, output_path, width, height, color1, color2, fontsize):
    import plotly.graph_objects as go

    def to_rgb_str(c): return f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})'
    font_scaled = fontsize * (width / 800)

    # Combine accessible values to get shared log scaling
    all_values = pd.concat([df1["accessible"], df2["accessible"]])
    log_all = np.log10(all_values + 1)
    max_log = log_all.max()

    size_max_scaled = 100 * (width / 800)

    # Add log-scaled size columns to df1 and df2 with slight power scaling
    df1 = df1.copy()
    df2 = df2.copy()
    exponent = 3.0
    df1["log_size"] = (np.log10(df1["accessible"] + 1) / max_log) ** exponent * size_max_scaled
    df2["log_size"] = (np.log10(df2["accessible"] + 1) / max_log) ** exponent * size_max_scaled

    fig = go.Figure()

    # Plot main data
    fig.add_trace(go.Scattergeo(
        locationmode='country names',
        locations=df1["country_name"],
        text=df1["country_name"] + " (" + df1["accessible"].astype(str) + ")",
        marker=dict(
            size=df1["log_size"],
            color=to_rgb_str(color1),
            opacity=0.8,
            line=dict(width=2, color="white")
        ),
        name='Training set',
        showlegend=True
    ))

    # Plot test data
    fig.add_trace(go.Scattergeo(
        locationmode='country names',
        locations=df2["country_name"],
        text=df2["country_name"] + " (" + df2["accessible"].astype(str) + ")",
        marker=dict(
            size=df2["log_size"],
            color="rgba(0,0,0,0)",
            line=dict(width=3, color=to_rgb_str(color2)),
            opacity=1.0
        ),
        name='Test set',
        showlegend=True
    ))

    fig.update_layout(
        geo=dict(projection_type="natural earth"),
        font=dict(size=font_scaled),
        paper_bgcolor='white',
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(
            x=0.05,
            y=0.55,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
    )

    fig.write_image(output_path, width=width, height=height)
    print(f"Saved combined world map to {output_path}")


def plot_sankey(df, output_path, width, height, color, fontsize):
    countries = df["country_name"].unique().tolist()
    datasets = df["dataset"].unique().tolist()
    labels = countries + datasets

    source = [labels.index(row["country_name"]) for _, row in df.iterrows()]
    target = [labels.index(row["dataset"]) for _, row in df.iterrows()]
    value = df["accessible"].tolist()

    color_hex = f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})'

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=30,
            line=dict(color="black", width=0.5),
            label=labels,
            color=color_hex,
            hoverlabel=dict(font=dict(size=fontsize))
        ),
        link=dict(source=source, target=target, value=value)
    )])

    fig.update_layout(
        font=dict(size=fontsize),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    fig.write_image(output_path, width=width, height=height)
    print(f"Saved Sankey diagram to {output_path}")

def main():
    args = parse_args()
    df_main = build_dataframe(DATASET_INFO, COUNTRY_MAP)
    df_test = build_dataframe(DATASET_INFO_TEST, COUNTRY_MAP)
    df_country_main = df_main.groupby("country_name")["accessible"].sum().reset_index()
    df_country_test = df_test.groupby("country_name")["accessible"].sum().reset_index()

    # World map with both
    plot_world_map(
        df_country_main,
        df_country_test,
        args.map_output,
        args.map_width,
        args.map_height,
        args.map_color,
        args.map_color_test,
        args.fontsize
    )

    # Sankey diagrams
    plot_sankey(df_main, args.sankey_output, args.sankey_width, args.sankey_height, args.map_color, args.fontsize)
    plot_sankey(df_test, args.sankey_output_test, args.sankey_width * 0.75, args.sankey_height, args.map_color_test, args.fontsize)

if __name__ == "__main__":
    main()
