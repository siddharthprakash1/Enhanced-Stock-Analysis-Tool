import html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..tokens import COLORS


def render_dashboard(symbol, bars, kpis, verdict_rows, out_path) -> None:
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True, vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(x=bars.index, open=bars.Open, high=bars.High, low=bars.Low, close=bars.Close,
                  increasing_line_color=COLORS["up"], decreasing_line_color=COLORS["down"], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=bars.index, y=bars.Close.rolling(50).mean(), line=dict(color=COLORS["accent"]), name="SMA50"), row=1, col=1)
    ret = bars.Close.pct_change()
    fig.add_trace(go.Bar(x=bars.index, y=ret, marker_color=COLORS["accent"], name="Daily return"), row=2, col=1)
    fig.update_layout(template="plotly_white", height=560,
                      font=dict(family="JetBrains Mono"), margin=dict(l=20, r=20, t=20, b=20))
    fig.update_xaxes(rangeslider_visible=False)
    chart_html = fig.to_html(full_html=False, include_plotlyjs="inline")

    strip = "".join(
        f'<div class="kpi"><div class="kl">{html.escape(str(k["label"]))}</div>'
        f'<div class="kv">{html.escape(str(k["value"]))}{html.escape(str(k["unit"]))}</div></div>'
        for k in kpis
    )
    rows = "".join(
        f'<tr><td>{html.escape(str(r["text"]))}</td>'
        f'<td class="st-{r["status"]}">{r["status"]}</td></tr>'
        for r in verdict_rows
    )
    page_html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{symbol}</title><style>
    body{{margin:0;font-family:'Space Grotesk',system-ui,sans-serif;color:{COLORS['ink']}}}
    .layout{{display:flex}} .sidebar{{width:150px;background:#FAFBFC;border-right:1px solid {COLORS['hairline']};padding:14px;min-height:100vh}}
    .sidebar a{{display:block;padding:8px 0;color:#6b7280;text-decoration:none;font-size:13px}}
    .main{{flex:1;padding:18px}} .strip{{background:{COLORS['panel']};color:{COLORS['data_text']};border-radius:10px;padding:12px;display:flex;gap:18px;font-family:'JetBrains Mono',monospace}}
    .kl{{color:#8B949E;font-size:9px}} .kv{{font-size:15px}}
    table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:14px}} td{{padding:6px;border-bottom:1px solid {COLORS['hairline']}}}
    .st-supported{{color:{COLORS['verified']}}} .st-contradicted{{color:{COLORS['down']}}} .st-corrected{{color:{COLORS['caution']}}}
    </style></head><body><div class="layout">
    <nav class="sidebar"><b>{symbol}</b><a>Overview</a><a>Technical</a><a>Fundamental</a><a>Risk</a><a>Valuation</a><a>Verification</a></nav>
    <div class="main"><div class="strip">{strip}</div>{chart_html}
    <h3>Verification</h3><table>{rows}</table></div></div></body></html>"""
    from pathlib import Path; Path(out_path).write_text(page_html)
