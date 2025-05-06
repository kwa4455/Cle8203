
import plotly.graph_objects as go

def apply_glass_style(fig: go.Figure, theme: dict, font_size: str = "16px") -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(255, 255, 255, 0.1)",
        plot_bgcolor="rgba(255, 255, 255, 0.1)",
        font=dict(color=theme.get("text", "#000000"), size=int(font_size.replace("px", ""))),
        margin=dict(l=40, r=40, t=60, b=40),
        title_font=dict(size=20, color=theme.get("text", "#000000"), family="Poppins"),
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.3)",
            font_size=14,
            font_family="Poppins"
        ),
    )

    # Add a rectangle shape behind the plot area to simulate shadow and blur
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
        fillcolor="rgba(255, 255, 255, 0.15)",
        line=dict(width=0),
        layer="below"
    )

    return fig






def plotly_table(df, theme, font_size="16px"):
    headers = list(df.columns)
    cells = [df[col].astype(str).tolist() for col in headers]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color=theme["button"],
            font=dict(color='white', size=int(font_size.replace("px", ""))),
            align='center',
            line_color='darkslategray',
            height=32
        ),
        cells=dict(
            values=cells,
            fill_color='rgba(255,255,255,0.4)',
            font=dict(color=theme["text"], size=int(font_size.replace("px", ""))),
            align='left',
            line_color='lightgray',
            height=28
        )
    )])

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=20, b=10)
    )
    return fig
