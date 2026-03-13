# ui/theme.py
import streamlit as st

def inject_theme():
    st.markdown(
        """
        <style>
        :root{
          --bg:#0b1220;
          --panel:#101a2e;
          --card:#0f1a30;
          --muted:#8ea0bf;
          --text:#e8eefc;
          --accent:#6ee7ff;
          --accent2:#a78bfa;
          --danger:#fb7185;
          --ok:#34d399;
          --border:rgba(255,255,255,0.08);
          --shadow: 0 12px 30px rgba(0,0,0,0.35);
          --radius: 18px;
        }
        html, body, [class*="css"] { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto; }
        .stApp { background: radial-gradient(1200px 600px at 10% 0%, rgba(110,231,255,0.12), transparent 60%),
                           radial-gradient(1200px 600px at 90% 0%, rgba(167,139,250,0.10), transparent 60%),
                           var(--bg); color: var(--text); }

        /* Hide Streamlit default header */
        header { visibility: hidden; height: 0px; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }

        .wrap { max-width: 1200px; margin: 0 auto; padding: 12px 4px 20px; }
        .topbar {
          display:flex; align-items:center; justify-content:space-between;
          padding: 14px 18px; border:1px solid var(--border); background:rgba(16,26,46,0.70);
          backdrop-filter: blur(10px); border-radius: var(--radius); box-shadow: var(--shadow);
        }
        .brand { display:flex; gap:12px; align-items:center; }
        .logo {
          width:40px; height:40px; border-radius:14px;
          background: linear-gradient(135deg, var(--accent), var(--accent2));
          box-shadow: 0 10px 20px rgba(110,231,255,0.18);
        }
        .title h1 { font-size: 18px; margin: 0; letter-spacing: 0.2px; }
        .title p { font-size: 12px; margin: 0; color: var(--muted); }

        .grid { display:grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px; }
        @media (max-width: 900px){ .grid { grid-template-columns: 1fr; } }

        .card {
          border: 1px solid var(--border);
          background: rgba(15,26,48,0.75);
          border-radius: var(--radius);
          padding: 16px;
          box-shadow: var(--shadow);
        }
        .card h2 { margin:0 0 6px; font-size: 14px; }
        .card p { margin:0 0 10px; color: var(--muted); font-size: 12px; }

        .pill {
          display:inline-flex; align-items:center; gap:8px;
          border: 1px solid var(--border); border-radius: 999px;
          padding: 6px 10px; color: var(--muted); font-size: 12px;
          background: rgba(255,255,255,0.03);
        }
        .pill b { color: var(--text); font-weight: 600; }

        .kpis { display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
        @media (max-width: 900px){ .kpis { grid-template-columns: repeat(2, 1fr);} }

        .kpi {
          padding: 14px; border-radius: 16px;
          border: 1px solid var(--border);
          background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
        }
        .kpi .label { color: var(--muted); font-size: 11px; }
        .kpi .value { font-size: 20px; font-weight: 700; margin-top: 6px; }
        .kpi .delta { color: var(--muted); font-size: 11px; margin-top: 4px; }

        .cta {
          border-radius: 16px !important;
          border: 1px solid rgba(110,231,255,0.35) !important;
          background: linear-gradient(135deg, rgba(110,231,255,0.25), rgba(167,139,250,0.20)) !important;
          color: var(--text) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
