import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import datetime
import joblib
import base64
import yfinance as yf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# ─────────────────────────────────────────────────────────────────────
# 1) GLOBAL PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Investimate Dashboard", page_icon="🔐", layout="wide")

# Google Fonts + default font families
st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?
        family=Playfair+Display:wght@700&
        family=Josefin+Sans:wght@300;400;600&display=swap');

      html, body, [class*="css"] {
        font-family: 'Josefin Sans', sans-serif !important;
      }
      h1, h2, h3, h4, h5, h6,
      .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
      .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 700 !important;
      }
    </style>
""", unsafe_allow_html=True)

# Background helper
def set_bg_from_local(image_path):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
          background-image: url("data:image/png;base64,{b64}");
          background-size: cover;
          background-position: center;
          background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)

set_bg_from_local("images/background.png")


# ─────────────────────────────────────────────────────────────────────
# 2) LOAD MODELS & DATA
# ─────────────────────────────────────────────────────────────────────
# RandomForest models
aapl_model = joblib.load("models/aapl_model.joblib")
msft_model = joblib.load("models/msft_model.joblib")
klac_model = joblib.load("models/klac_model.joblib")
qcom_model = joblib.load("models/qcom_model.joblib")
crwd_model = joblib.load("models/crwd_model.joblib")
pltr_model = joblib.load("models/pltr_model.joblib")
# XGBoost risk‐classifier
xgb_model  = joblib.load("models/best_xgb_model.joblib")

# 2.1) Full enriched dataset for classification & clustering
@st.cache_data
def load_full_data():
    df = pd.read_csv("final_data.csv", parse_dates=["date_value"])
    return df.rename(columns={"date_value":"date"}).set_index("date")

full_df = load_full_data()

# 2.2) Fact/rule dataset (for risk‐prediction & for rule‐based strategies)
@st.cache_data
def load_fact_data():
    df = pd.read_csv("fact_dataset.csv", parse_dates=["date_value"])
    return df.rename(columns={"date_value":"date"}).set_index("date")

fact_df = load_fact_data()
risk_feats  = xgb_model.feature_names_in_.tolist()
risk_levels = ["Low Risk","Moderate Risk","High Risk"]

# 2.3) Rule‐based data for buy/hold, MA‐crossover, RSI
@st.cache_data
def load_rule_data():
    df = pd.read_csv("fact_dataset.csv", parse_dates=["date_value"])
    return df.rename(columns={"date_value":"date"}).set_index("date")

data_rule = load_rule_data()


# ────────────────────────────────────────────────────────────────
# Hard-coded risk buckets (model retained *only* for display)
# ────────────────────────────────────────────────────────────────
risk_by_company = {
    "AAPL": "Low Risk",
    "MSFT": "Low Risk",
    "KLAC": "Moderate Risk",
    "QCOM": "Moderate Risk",
    "CRWD": "High Risk",
    "PLTR": "High Risk"
}

# ─────────────────────────────────────────────────────────────────────
# 3) SHARED HELPERS & SECTIONS
# ─────────────────────────────────────────────────────────────────────
def display_link_cards(links, banner_src=None):
    st.markdown("---")
    st.subheader("📰 Related News & Nasdaq 100 Context")
    st.write("Below are curated news articles and macroeconomic resources relevant to this company’s place in the Nasdaq 100.")
    if banner_src:
        st.image(banner_src, width=500, use_column_width=False)
    cols = st.columns(len(links), gap="small")
    for col, link in zip(cols, links):
        title, desc, url, *rest = link
        thumb = rest[0] if rest else None
        with col:
            if thumb:
                st.image(thumb, width=100)
            st.markdown(f"""
                <div style="
                  background: rgba(255,255,255,0.85);
                  padding:12px;
                  border-radius:8px;
                  box-shadow:0 1px 3px rgba(0,0,0,0.12);
                  text-align:center;
                ">
                  <h4 style="margin:0;font-size:1rem;">
                    <a href="{url}" target="_blank"
                       style="text-decoration:none;color:#1f77b4;">
                      {title}
                    </a>
                  </h4>
                  <hr style="border:none;border-top:1px dotted #888;margin:8px 0;" />
                  <p style="margin:4px 0 0;color:#333;font-size:0.9rem;font-style:italic;">
                    {desc}
                  </p>
                </div>
            """, unsafe_allow_html=True)

def risk_prediction_section(ticker, model):
                    # (Tiny reminder banner)
    st.warning(
        "⚠️ The risk categories shown here are *hard-coded* for this demo. "
        "In a real app you’d display the model’s own prediction instead."
    )
    
    df_t = full_df[full_df["company_prefix"] == ticker].sort_index()

    if df_t.empty:
        st.warning("⚠️ No enriched data found for " + ticker)
        return
    latest = df_t.iloc[-1]
    feats   = model.feature_names_in_.tolist()
    X       = latest[feats].values.reshape(1,-1)
    cls     = model.predict(X)[0]
    label   = risk_levels[cls]
    probs   = model.predict_proba(X)[0]
    cmap    = {"Low Risk":"#2a9d8f","Moderate Risk":"#f4a261","High Risk":"#e63946"}

    # Banner
    st.markdown(f"""
      <div style="
        background:#eef7fa;
        padding:14px;
        border-radius:8px;
        margin-top:16px;
        margin-bottom:12px;
        display:flex; align-items:center;
      ">
        <div style="flex:1; font-family:'Playfair Display';font-size:1.1rem;color:#264653;">
          Predicted Risk Category for <strong>{ticker}</strong>:
        </div>
        <div style="
          font-family:'Playfair Display';
          font-size:1.2rem;
          font-weight:700;
          color:{cmap[label]};
        ">
          {label}
        </div>
      </div>
    """, unsafe_allow_html=True)

    # Probability cards
    cols = st.columns(len(risk_levels), gap="small")
    for col, lvl, p in zip(cols, risk_levels, probs):
        col.markdown(f"""
        <div style="
          background:white;
          padding:10px;
          border-radius:6px;
          box-shadow:0 1px 2px rgba(0,0,0,0.1);
          text-align:center;
        ">
          <div style="font-size:0.9rem;color:#555;font-weight:500;">
            {lvl}
          </div>
          <div style="
            font-size:1.4rem;
            color:{cmap[lvl]};
            font-weight:700;
            margin-top:4px;
          ">
            {p*100:.1f}%
          </div>
        </div>
        """, unsafe_allow_html=True)

def cluster_insight_section(ticker, model):
    st.markdown("---")
    st.subheader("📊 Model Feature Importance")
    feats = model.feature_names_in_.tolist()
    imps = model.feature_importances_
    df_imp = (pd.DataFrame({"Importance":imps}, index=feats)
                .sort_values("Importance",ascending=False)
                .reset_index().rename(columns={"index":"Feature"}))

    # Top-3 banner
    top3 = df_imp.head(3)
    summ = ", ".join(f"<strong>{r.Feature}</strong> ({r.Importance:.2f})"
                     for _,r in top3.iterrows())
    st.markdown(f"""
      <div style="
        background-color:#f0f8ff;
        padding:12px;
        border-left:4px solid #264653;
        border-radius:6px;
        margin-bottom:16px;
      ">
        <p style="margin:0;font-size:1rem;color:#333;">
          <strong style="color:#0072B2;">
            Top 3 drivers for {ticker}:
          </strong> {summ}
        </p>
      </div>
    """, unsafe_allow_html=True)

    # Interactive bar chart
    fig = px.bar(
      df_imp[::-1], x="Importance", y="Feature", orientation="h",
      title=f"{ticker} Feature Importances",
      labels={"Importance":"Rel. Importance","Feature":""},
      hover_data={"Importance":":.2f"},
      color_discrete_sequence=["#264653"]
    )
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Importance: %{x:.2f}<extra></extra>")
    fig.update_layout(
      plot_bgcolor="#e8e8e8", paper_bgcolor="#e8e8e8",
      font_family="Josefin Sans", font_color="#222",
      title_font_family="Playfair Display", title_font_size=14, title_font_color="#111",
      xaxis=dict(showgrid=True,gridcolor="white",tickfont=dict(color="#222"),title_font=dict(color="black",size=12)),
      yaxis=dict(tickfont=dict(color="#222",size=11)),
      hoverlabel=dict(bgcolor="white",font_color="black"),
      margin=dict(l=100,r=20,t=60,b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Risk prediction underneath
    risk_prediction_section(ticker, xgb_model)

def classification_section(ticker, model):
    st.markdown("---")
    st.subheader(f"Company Trend Predicted by {ticker} Model")
    st.info("""
      **What this model predicts:**  
      • It forecasts **only** the direction of the **next trading day**.  
      • 📈 **Bullish (Up)** means tomorrow’s closing price is expected to be _higher_.  
      • 📉 **Bearish (Down)** means tomorrow’s closing price is expected to be _lower_.
    """)

    # Live signal (simulated)
    order = model.feature_names_in_.tolist()
    sim_in = {
      "RSI":np.random.uniform(10,90),
      "atr":np.random.uniform(1,5),
      "macd":np.random.uniform(-5,5),
      "Denoised_Close":np.random.uniform(100,300),
      "stock_market_volatility_vix_index":np.random.uniform(10,30),
      "volume":np.random.uniform(1e6,1e7)
    }
    X_live = pd.DataFrame([[sim_in[f] for f in order]],columns=order)
    pred   = model.predict(X_live)[0]
    # build text + color
    signal_text  = "📈 Trending Up (Bullish)" if pred == 1 else "📉 Trending Down (Bearish)"
    signal_color = "#2a9d8f" if pred == 1 else "#e63946"

    # render via markdown with HTML
    st.markdown(f"""
    <div style="
        background-color: {signal_color}22;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 16px;
        text-align: center;
    ">
      <span style="
          font-size: 1.5rem;
          font-weight: 700;
          color: {signal_color};
      ">
        Latest Model Signal for {ticker}: {signal_text}
      </span>
    </div>
    """, unsafe_allow_html=True)


    with st.expander("📊 View input features"):
        st.dataframe(X_live.T.rename(columns={0:"Value"}))

    # Historical performance
    hist  = full_df[full_df["company_prefix"]==ticker].copy()
    y_true=hist["Target"].astype(int)
    X_hist=hist[order]
    y_pred=model.predict(X_hist)
    acc  = accuracy_score(y_true,y_pred)
    prec = precision_score(y_true,y_pred)
    rec  = recall_score(y_true,y_pred)

    # build each metric card’s HTML
    cards = []
    for lbl, val in [("Accuracy", acc), ("Precision", prec), ("Recall", rec)]:
        cards.append(
            f'<div style="'
            'flex:1; background:white; padding:8px 12px; '
            'border-radius:6px; text-align:center; '
            'box-shadow:0 1px 3px rgba(0,0,0,0.1);'
            '">'
            f'<div style="font-size:14px;color:#555;">{lbl}</div>'
            f'<div style="font-size:18px;color:#2a9d8f;font-weight:600;">{val:.2f}</div>'
            '</div>'
        )
    cards_html = "".join(cards)

    # then render the whole container
    st.markdown(f"""
    <div style="
    display:flex;
    gap:16px;
    background-color:#f0f8ff;
    padding:12px;
    border-radius:8px;
    margin-bottom:16px;
    ">
    {cards_html}
    </div>
    """, unsafe_allow_html=True)


    # Interactive confusion matrix
    cm = confusion_matrix(y_true,y_pred,labels=[0,1])
    cm_df = pd.DataFrame(
      cm,
      index=["Actual ↓ Down","Actual ↓ Up"],
      columns=["Predicted → Down","Predicted → Up"]
    )
    fig = px.imshow(
      cm_df, text_auto=True, color_continuous_scale="Blues", aspect="auto"
    )
    fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x}: %{z}<extra></extra>")
    fig.update_layout(
      title=f"{ticker} Confusion Matrix",
      title_font_family="Playfair Display", title_font_size=16, title_font_color="#111",
      font=dict(family="Josefin Sans", color="#111", size=14),
      xaxis=dict(tickfont=dict(color="#111")), yaxis=dict(tickfont=dict(color="#111")),
      plot_bgcolor="#e8e8e8", paper_bgcolor="#e8e8e8", margin=dict(l=60,r=20,t=60,b=60)
    )
    st.plotly_chart(fig, use_container_width=True)

def stock_chart_section(ticker):
    st.markdown("---")
    st.subheader("Stock Price Visualization")
    tf = st.selectbox("Select timeframe:", ["1M","3M","6M","1Y","5Y"], key=f"{ticker}_tf")
    periods={"1M":"1mo","3M":"3mo","6M":"6mo","1Y":"1y","5Y":"5y"}
    data=yf.Ticker(ticker).history(period=periods[tf])
    if data.empty:
        st.warning("No data for that timeframe.")
        return
    dfp = data.reset_index().rename(columns={"Date":"Date","Close":"Close"})
    fig = px.line(dfp, x="Date", y="Close",
                  title=f"{ticker} – Last {tf}",
                  labels={"Close":"Price (USD)"})
    fig.update_traces(
      mode="lines+markers", marker=dict(size=6),
      line=dict(width=2.5,color="#0072B2"),
      hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Price: $%{y:.2f}<extra></extra>"
    )
    fig.update_layout(
      font_family="Josefin Sans", font_color="#222",
      title_font_family="Playfair Display", title_font_size=18, title_font_color="#111",
      plot_bgcolor="#fafafa", paper_bgcolor="#fafafa",
      hoverlabel=dict(bgcolor="white",font_color="black"),
      xaxis=dict(showgrid=True,gridcolor="white",tickangle=30,
                 title_font=dict(size=12,color="#222"),tickfont=dict(color="#222")),
      yaxis=dict(showgrid=True,gridcolor="white",
                 tickformat="$,.0f",title_font=dict(size=12,color="#222"),tickfont=dict(color="#222")),
      margin=dict(l=40,r=20,t=60,b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

def page_footer():
    st.markdown("---")
    st.caption("Prototype dashboard – classification model applied to simulated input data.")


# ─────────────────────────────────────────────────────────────────────
# 4) PAGE FUNCTIONS (ML MODELS)
# ─────────────────────────────────────────────────────────────────────
def apple_page():
    st.image("images/apple_logo.png", width=80)
    st.title("AAPL Market Trend Classification")
    st.markdown("This dashboard uses a Random Forest classifier to assess whether Apple Inc.'s market trend is up or down.")
    with st.expander("ℹ OVERVIEW", expanded=True):
        st.markdown(
            """
            <div style="font-family: Georgia, serif; line-height:1.5; font-size:0.95rem;">
            
            **Apple Inc. (AAPL):**  Apple is one of the **largest** components of the Nasdaq Composite…
            
            - **Company-specific news** (Seeking Alpha) delivers deep dives…
            - **Investor Relations** updates give you the official story…
            - **FRED economic data** shows the U.S. macro backdrop…
            - **IMF World Economic Outlook** places Apple’s global sales in context…

            > **How to use these links:**  
            > • **Seeking Alpha** & **Investor Relations** = company deep-dives.  
            > • **FRED** & **IMF** = macroeconomic & global context.
            </div>
            """,
            unsafe_allow_html=True
        )
    links = [
        ("Seeking Alpha", "Analyst insights & community discussion on AAPL", "https://seekingalpha.com/symbol/AAPL"),
        ("FRED Data",      "Federal Reserve economic data for broader context", "https://fred.stlouisfed.org/"),
        ("IMF WEO",        "Latest IMF World Economic Outlook reports",      "https://www.imf.org/en/Publications/WEO")
    ]
    display_link_cards(links, banner_src="images/apple_news.jpeg")
    cluster_insight_section("AAPL", xgb_model)
    classification_section("AAPL", aapl_model)
    stock_chart_section("AAPL")
    page_footer()


def microsoft_page():
    st.image("images/microsoft_logo.png", width=250)
    st.title("MSFT Market Trend Classification")
    st.markdown("This dashboard uses a Random Forest classifier to assess whether Microsoft’s market trend is up or down.")
    with st.expander("ℹ️ Microsoft Corp. (MSFT) Overview", expanded=True):
        st.markdown(
            """
            <div style="font-family: Georgia, serif; line-height:1.5; font-size:0.95rem;">
                            
            **Microsoft Corp. (MSFT):**  
            Microsoft is the **second-largest** component of the Nasdaq-100 (around **10%** weight), renowned for its dominant position in enterprise software, cloud computing (Azure), and productivity platforms. Its stable, subscription-based revenue model makes it a cornerstone for many tech-focused portfolios.

            - **Company-specific news** (Seeking Alpha) covers Azure growth, Office 365 adoption, and AI/ML initiatives.  
            - **Investor Relations** updates provide direct insights on quarterly results, guidance, and strategic direction.  
            - **FRED economic data** (business investment, Fed policy) contextualizes corporate IT spending trends.  
            - **IMF World Economic Outlook** offers a global perspective on GDP forecasts impacting enterprise tech budgets.

            > **How to use these links:**  
            > • **Seeking Alpha** & **Investor Relations** = deep dives into Microsoft’s core business drivers.  
            > • **FRED** & **IMF** = macroeconomic & global context for the broader tech environment.
            """,
            unsafe_allow_html=True)
    links = [
        ("Microsoft on Seeking Alpha", "Analyst insights & community discussion on MSFT", "https://seekingalpha.com/symbol/MSFT"),
        ("Microsoft Investor Relations", "Official financial reports and updates", "https://www.microsoft.com/en-us/Investor"),
        ("FRED Data", "Federal Reserve economic data for broader context", "https://fred.stlouisfed.org/"),
        ("IMF WEO", "Latest IMF World Economic Outlook reports", "https://www.imf.org/en/Publications/WEO")
    ]
    display_link_cards(links, banner_src="images/microsoft_news.jpeg")
    cluster_insight_section("MSFT", xgb_model)
    classification_section("MSFT", msft_model)
    stock_chart_section("MSFT")
    page_footer()

def kla_page():
    st.image("images/kla_logo.png", width=250)
    st.title("KLAC Market Trend Classification")
    st.markdown("This dashboard uses a Random Forest classifier to assess whether KLA’s market trend is up or down.")
    with st.expander("ℹ️ KLA Corp. (KLAC) Overview", expanded=True):
        st.markdown("""
            
            <div style="font-family: Georgia, serif; line-height:1.5; font-size:0.95rem;">
                    
            **KLA Corp. (KLAC):**  KLA is a **leading** semiconductor process-control equipment manufacturer and a mid-cap member of the Nasdaq-100. Its revenue directly tracks capital spending in wafer fabs, making it a bellwether for chip-cycle health.

            - **Company-specific news** (Seeking Alpha) dives into fab equipment orders, technology upgrades, and customer rollouts.  
            - **Investor Relations** updates share quarterly results, capital expenditure guidance, and customer pipeline insights.  
            - **FRED economic data** (industrial production, business investment) reveals trends in semiconductor manufacturing demand.  
            - **IMF World Economic Outlook** provides a global macro backdrop on technology investments across key regions.

            > **How to use these links:**  
            > • **Seeking Alpha** & **Investor Relations** = in-depth analysis of KLA’s equipment demand and customer health.  
            > • **FRED** & **IMF** = broad economic context for semiconductor capital spending cycles.
                """, unsafe_allow_html=True)

    links = [
        ("KLA on Seeking Alpha", "Analyst insights & community discussion on KLAC", "https://seekingalpha.com/symbol/KLAC"),
        ("KLA Investor Relations", "Official financial reports and updates", "https://ir.kla.com/"),
        ("FRED Data", "Federal Reserve economic data for broader context", "https://fred.stlouisfed.org/"),
        ("IMF WEO", "Latest IMF World Economic Outlook reports", "https://www.imf.org/en/Publications/WEO")
    ]    
    display_link_cards(links, banner_src="images/kla_news.jpeg")
    cluster_insight_section("KLAC", xgb_model)
    classification_section("KLAC", klac_model)
    stock_chart_section("KLAC")
    page_footer()

def qualcomm_page():
    st.image("images/qualcomm_logo.png", width=350)
    st.title("QCOM Market Trend Classification")
    st.markdown("This dashboard uses a Random Forest classifier to assess whether Qualcomm’s market trend is up or down.")
    with st.expander("ℹ️ Qualcomm Inc. (QCOM) Overview", expanded=True):
        st.markdown("""
            
            <div style="font-family: Georgia, serif; line-height:1.5; font-size:0.95rem;">        
            **Qualcomm Inc. (QCOM):**  Qualcomm is a **key** designer of mobile processors and wireless-chip standards (CDMA/5G) and a mid-cap Nasdaq-100 member. Its fortunes rise and fall with the wireless cycle, making it a barometer for 5G adoption and IoT growth.

            - **Company-specific news** (Seeking Alpha) delves into 5G chipset rollouts, patent and licensing developments, and handset wins.  
            - **Investor Relations** updates cover quarterly results, royalty revenue breakdowns, and strategic partnerships.  
            - **FRED economic data** (telecom investment, durable goods orders) shows trends in carrier capex and device replacement cycles.  
            - **IMF World Economic Outlook** offers a global view on smartphone penetration and infrastructure spending.

            > **How to use these links:**  
            > • **Seeking Alpha** & **Investor Relations** = deep dives on Qualcomm’s technology leadership and licensing business.  
            > • **FRED** & **IMF** = macro & global context for the wireless and semiconductor markets.
                """, unsafe_allow_html=True)

    links = [
        ("Qualcomm on Seeking Alpha", "Analyst insights & community discussion on QCOM", "https://seekingalpha.com/symbol/QCOM"),
        ("Qualcomm Investor Relations", "Official financial reports and updates", "https://investor.qualcomm.com/"),
        ("FRED Data", "Federal Reserve economic data for broader context", "https://fred.stlouisfed.org/"),
        ("IMF WEO", "Latest IMF World Economic Outlook reports", "https://www.imf.org/en/Publications/WEO")]
    
    display_link_cards(links, banner_src="images/qualcomm_news.png")
    cluster_insight_section("QCOM", xgb_model)
    classification_section("QCOM", qcom_model)
    stock_chart_section("QCOM")
    page_footer()

def crowdstrike_page():
    st.image("images/crowdstrike_logo.png", width=350)
    st.title("CRWD Market Trend Classification")
    st.markdown("This dashboard uses a Random Forest classifier to assess whether CrowdStrike’s market trend is up or down.")
    with st.expander("ℹ️ CrowdStrike Holdings Inc. (CRWD) Overview", expanded=True):
        st.markdown("""
            
            <div style="font-family: Georgia, serif; line-height:1.5; font-size:0.95rem;">
                            
            **CrowdStrike Holdings Inc. (CRWD):** CrowdStrike is a **fast-growing**, cloud-native cybersecurity platform added to the Nasdaq-100 in recent years. Its subscription-based model and strong net-retention rates make it a bellwether for enterprise security spending.

            - **Company-specific news** (Seeking Alpha) covers ARR growth, new security modules, and threat-intelligence wins.  
            - **Investor Relations** updates share quarterly results, customer-add rates, and platform roadmap details.  
            - **FRED business-survey indices** (non-defense capital goods, business confidence) hint at corporate IT and security budgets.  
            - **IMF World Economic Outlook** provides a global backdrop for IT and cybersecurity spending trends.

            > **How to use these links:**  
            > • **Seeking Alpha** & **Investor Relations** = deep dives on CrowdStrike’s growth drivers and product roadmap.  
            > • **FRED** & **IMF** = macroeconomic & global context for security-software adoption and enterprise IT investment.
                """, unsafe_allow_html=True)

    links = [
        ("CrowdStrike on Seeking Alpha", "Analyst insights & community discussion on CRWD", "https://seekingalpha.com/symbol/CRWD"),
        ("CrowdStrike Investor Relations", "Official financial reports and updates", "https://ir.crowdstrike.com/"),
        ("FRED Data", "Federal Reserve economic data for broader context", "https://fred.stlouisfed.org/"),
        ("IMF WEO", "Latest IMF World Economic Outlook reports", "https://www.imf.org/en/Publications/WEO")]
    display_link_cards(links, banner_src="images/crowdstrike_news.jpeg")
    cluster_insight_section("CRWD", xgb_model)
    classification_section("CRWD", crwd_model)
    stock_chart_section("CRWD")
    page_footer()

def palantir_page():
    st.image("images/palantir_logo.png", width=250)
    st.title("PLTR Market Trend Classification")
    st.markdown("This dashboard uses a Random Forest classifier to assess whether Palantir’s market trend is up or down.")
    with st.expander("ℹ️ Palantir Technologies Inc. (PLTR) Overview", expanded=True):
        st.markdown("""
                    
            <div style="font-family: Georgia, serif; line-height:1.5; font-size:0.95rem;">
                            
            **Palantir Technologies Inc. (PLTR):**  
            Palantir provides advanced **data-analytics** platforms for both government and enterprise clients. A more **volatile** Nasdaq-100 name, its stock often reacts sharply to new contract awards and platform deployments.

            - **Company-specific news** (Seeking Alpha) covers contract wins, platform enhancements, and partnership announcements.  
            - **Investor Relations** updates share earnings details, commercial versus government revenue splits, and strategic roadmap insights.  
            - **FRED indicators** (government spending, procurement budgets) show funding trends in defense and public-sector IT.  
            - **IMF World Economic Outlook** offers a macro view of global public-sector and enterprise IT investment levels.

            > **How to use these links:**  
            > • **Seeking Alpha** & **Investor Relations** = focused deep dives on Palantir’s contracts and product evolution.  
            > • **FRED** & **IMF** = high-level context on government and enterprise spending that drives Palantir’s pipeline.
                """, unsafe_allow_html=True)

    links = [
        ("Palantir on Seeking Alpha", "Analyst insights & community discussion on PLTR", "https://seekingalpha.com/symbol/PLTR"),
        ("Palantir Investor Relations", "Official financial reports and updates", "https://investors.palantir.com/"),
        ("FRED Data", "Federal Reserve economic data for broader context", "https://fred.stlouisfed.org/"),
        ("IMF WEO", "Latest IMF World Economic Outlook reports", "https://www.imf.org/en/Publications/WEO")]
    display_link_cards(links, banner_src="images/palantir_news.jpg")
    cluster_insight_section("PLTR", xgb_model)
    classification_section("PLTR", pltr_model)
    stock_chart_section("PLTR")
    page_footer()

# ─────────────────────────────────────────────────────────────────────
# 5) RULE‐BASED STRATEGIES
# ─────────────────────────────────────────────────────────────────────
def buy_and_hold_from_cleaned_data(data, company, inv):
    df = data[data["company_prefix"]==company].copy()
    if df.empty:
        st.warning("No data for " + company); return
    buy = df["close_value"].iloc[0]
    shares = inv//buy
    rem = inv - shares*buy
    df["Portfolio Value"] = df["close_value"]*shares + rem
    return df

def moving_average_crossover(data, company, inv):
    df = data[data["company_prefix"]==company].copy()
    if df.empty:
        st.warning("No data for " + company); return
    df["short_ma"]=df["sma_10"]; df["long_ma"]=df["sma_50"]
    pos=0; cash=inv; pv=[]
    for i in range(len(df)):
        price = df["close_value"].iat[i]
        if df["short_ma"].iat[i]>df["long_ma"].iat[i] and pos==0:
            pos=cash/price; cash=0
        elif df["short_ma"].iat[i]<df["long_ma"].iat[i] and pos>0:
            cash=pos*price; pos=0
        pv.append(pos*price+cash)
    df["Portfolio Value"]=pv
    return df

def rsi_based_strategy(data, company, inv, rsi_column="rsi_14", buy_thr=30, sell_thr=70):
    df = data[data["company_prefix"]==company].copy()
    if df.empty:
        st.warning("No data for "+company); return
    df["signal"]=0
    for i in range(1,len(df)):
        r0, r1 = df[rsi_column].iat[i-1], df[rsi_column].iat[i]
        if r0<buy_thr<=r1: df.at[df.index[i],"signal"]=1
        elif r0>sell_thr>=r1: df.at[df.index[i],"signal"]=-1
    pos=0; cash=inv; pv=[]
    for i in range(len(df)):
        price=df["close_value"].iat[i]; sig=df["signal"].iat[i]
        if sig==1 and pos==0: pos=cash/price; cash=0
        elif sig==-1 and pos>0: cash=pos*price; pos=0
        pv.append(pos*price+cash)
    df["Portfolio Value"]=pv
    return df

def plot_portfolio_value(df, prefix):
    dfp = df.reset_index().rename(columns={"date":"Date"})
    fig = px.line(dfp, x="Date", y="Portfolio Value", title=f"Buy & Hold – {prefix}")
    fig.update_traces(mode="lines+markers", marker=dict(size=6), line=dict(color="#2a9d8f",width=2.5),
                      hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Value: $%{y:.2f}<extra></extra>")
    fig.update_layout(plot_bgcolor="#2b2b2b",paper_bgcolor="#2b2b2b",font_color="white",
                      xaxis=dict(showgrid=True,gridcolor="#444",tickfont=dict(color="white")),
                      yaxis=dict(showgrid=True,gridcolor="#444",tickfont=dict(color="white")),
                      title_font_color="white")
    st.plotly_chart(fig, use_container_width=True)

def plot_ma_crossover(df, prefix):
    dfp = df.reset_index().rename(columns={"date":"Date"})
    fig = px.line(dfp, x="Date", y=["close_value","short_ma","long_ma","Portfolio Value"],
                  title=f"MA Crossover – {prefix}")
    for trace,color in zip(fig.data,["#0072B2","#E69F00","#56B4E9","#2a9d8f"]):
        trace.update(line=dict(color=color,width=2),hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{legendgroup}: %{y:.2f}<extra></extra>")
    fig.update_layout(plot_bgcolor="#2b2b2b",paper_bgcolor="#2b2b2b",font_color="white",
                      xaxis=dict(showgrid=True,gridcolor="#444",tickfont=dict(color="white")),
                      yaxis=dict(showgrid=True,gridcolor="#444",tickfont=dict(color="white")),
                      title_font_color="white",legend=dict(font=dict(color="white")))
    st.plotly_chart(fig, use_container_width=True)

def plot_rsi_strategy(df, prefix):
    dfp = df.reset_index().rename(columns={"date":"Date"})
    buy_thr = st.session_state.get("buy_thr",30)
    sell_thr= st.session_state.get("sell_thr",70)
    fig = px.line(dfp, x="Date", y=["close_value","rsi_14","Portfolio Value"],
                  title=f"RSI Strategy – {prefix}")
    fig.data[0].update(line=dict(color="#0072B2",width=2.5))
    fig.data[1].update(line=dict(color="#8E44AD",width=2))
    fig.data[2].update(line=dict(color="#2a9d8f",width=2,dash="dash"))
    fig.add_hline(y=buy_thr,line_dash="dash",line_color="#2a9d8f",
                  annotation_text="Buy Thr",annotation_font_color="white")
    fig.add_hline(y=sell_thr,line_dash="dash",line_color="#e63946",
                  annotation_text="Sell Thr",annotation_font_color="white")
    fig.update_layout(plot_bgcolor="#2b2b2b",paper_bgcolor="#2b2b2b",font_color="white",
                      xaxis=dict(showgrid=True,gridcolor="#444",tickfont=dict(color="white")),
                      yaxis=dict(showgrid=True,gridcolor="#444",tickfont=dict(color="white")),
                      title_font_color="white",legend=dict(font=dict(color="white")))
    st.plotly_chart(fig, use_container_width=True)

def buy_hold_page():
    st.header("Buy & Hold Strategy")
    company = st.selectbox("Company", ["AAPL","MSFT","KLAC","QCOM","CRWD","PLTR"])
    inv     = st.number_input("Investment ($)",1000.0,step=100.0)
    if st.button("Run"):
        df = buy_and_hold_from_cleaned_data(data_rule,company,inv)
        if df is not None:
            plot_portfolio_value(df,company)
            st.dataframe(df[["close_value","Portfolio Value"]].tail())

def ma_crossover_page():
    st.header("Moving Average Crossover")
    company = st.selectbox("Company", ["AAPL","MSFT","KLAC","QCOM","CRWD","PLTR"])
    inv     = st.number_input("Investment ($)",1000.0,step=100.0)
    if st.button("Run"):
        df = moving_average_crossover(data_rule,company,inv)
        if df is not None:
            plot_ma_crossover(df,company)
            st.dataframe(df[["close_value","short_ma","long_ma","Portfolio Value"]].tail())

def rsi_page():
    st.header("RSI-Based Strategy")
    company = st.selectbox("Company", ["AAPL","MSFT","KLAC","QCOM","CRWD","PLTR"])
    inv     = st.number_input("Investment ($)",1000.0,step=100.0)
    buy_thr = st.slider("Buy Threshold",0,100,30)
    sell_thr= st.slider("Sell Threshold",0,100,70)
    st.session_state.buy_thr, st.session_state.sell_thr = buy_thr, sell_thr
    if st.button("Run"):
        df = rsi_based_strategy(data_rule,company,inv,"rsi_14",buy_thr,sell_thr)
        if df is not None:
            plot_rsi_strategy(df,company)
            st.dataframe(df[["close_value","rsi_14","Portfolio Value"]].tail())


# ─────────────────────────────────────────────────────────────────────
# 6) LOGIN / SIGNUP / DISCLAIMER / PROFILE FLOW
# ─────────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "login"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "disclaimer_passed" not in st.session_state:
    st.session_state.disclaimer_passed = False

# -- LOGIN PAGE
if not st.session_state.authenticated and st.session_state.page == "login":
    st.markdown("<h1 style='color:#e63946;text-align:center;'>🔐 Welcome to Investimate</h1>", unsafe_allow_html=True)
    with st.container():
        st.markdown("""
          <div style='background:#1e1e2f;padding:40px;border-radius:10px;margin-top:20px;'>
            <h2 style='text-align:center;color:white;'>Login to your account</h2>
        """, unsafe_allow_html=True)
        st.text_input("Email",key="email")
        st.text_input("Password",type="password",key="password")
        c1,c2 = st.columns([1,2])
        with c1: st.checkbox("Remember me")
        with c2: st.markdown("<div style='text-align:right;'><a href='#'>Forgot password?</a></div>",unsafe_allow_html=True)
        if st.button("Login"):
            if st.session_state.password=="1234":
                st.session_state.authenticated=True
                st.session_state.page="disclaimer"
                st.rerun()
            else:
                st.error("Invalid credentials.")
        st.markdown("</div>",unsafe_allow_html=True)
    st.markdown("<hr>",unsafe_allow_html=True)
    st.markdown("""
      <div style='text-align:center;'>
        <p style='font-weight:bold;'>Or login with:</p>
        <button>🔍 Google</button>
        <button>🍎 Apple</button>
        <button>🪪 Email</button>
      </div>
    """,unsafe_allow_html=True)
    if st.button("Don't have an account? Sign Up"):
        st.session_state.page="signup"
        st.rerun()

# -- SIGN UP PAGE
elif st.session_state.page=="signup":
    st.title("📝 Create Your Investimate Account")
    st.checkbox("Stay anonymous",key="signup_anon")
    if not st.session_state.signup_anon:
        st.text_input("First Name",key="signup_first")
        st.text_input("Last Name",key="signup_last")
    st.text_input("Email",key="signup_email")
    st.text_input("Password",type="password",key="signup_pass")
    st.text_input("Confirm Password",type="password",key="signup_confirm")
    c1,c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Back to Login"):
            st.session_state.page="login"; st.rerun()
    with c2:
        if st.session_state.signup_pass and st.session_state.signup_confirm and \
           st.session_state.signup_pass==st.session_state.signup_confirm=="1234":
            if st.button("✅ Continue"):
                st.session_state.authenticated=True
                st.session_state.page="disclaimer"
                st.session_state.signup_name = ("Anonymous" if st.session_state.signup_anon
                                               else f"{st.session_state.signup_first} {st.session_state.signup_last}")
                st.rerun()
        else:
            st.button("✅ Continue",disabled=True)
    st.markdown("""
      <div style='text-align:center;'>
        <p style='font-weight:bold;'>Or sign up with:</p>
        <button>🔍 Google</button>
        <button>🍎 Apple</button>
        <button>🪪 Email</button>
      </div>
    """,unsafe_allow_html=True)

# -- DISCLAIMER PAGE
elif not st.session_state.disclaimer_passed:
    st.title("⚠️ AI System Disclaimer")
    st.markdown("""
    **This application uses AI models for educational purposes only.**  
    No financial decisions should be made solely on its outputs.
    """)
    agree = st.checkbox("I agree and am aware of these limitations.")
    if agree and st.button("Next"):
        st.session_state.disclaimer_passed=True
        st.rerun()

# -- PROFILE & WELCOME FLOW
else:
    if "step_experience_done" not in st.session_state:
        st.session_state.step_experience_done=False
    if "step_welcome_done" not in st.session_state:
        st.session_state.step_welcome_done=False
    if "experience_level" not in st.session_state:
        st.session_state.experience_level=None
    if "age" not in st.session_state:
        st.session_state.age=None

    # Step 1: Age & Experience
    if not st.session_state.step_experience_done:
        st.markdown("**Step 1 of 3: Investor Profile Setup**")
        st.progress(33)
        st.title("Investor Profile Setup")
        st.session_state.age = st.slider("Select your age:",18,100,30)
        st.markdown("Select your investment experience level:")
        b1,b2,b3 = st.columns(3)
        with b1:
            if st.button("🟢 Beginner"): st.session_state.experience_level="Beginner"
        with b2:
            if st.button("🟡 Intermediate"): st.session_state.experience_level="Intermediate"
        with b3:
            if st.button("🔴 Experienced"): st.session_state.experience_level="Experienced"
        if st.session_state.experience_level:
            st.success(f"You chose: {st.session_state.experience_level}")
            if st.button("Continue"):
                st.session_state.step_experience_done=True; st.rerun()

    # Step 2: Welcome
    elif st.session_state.step_experience_done and not st.session_state.step_welcome_done:
        st.markdown("**Step 2 of 3: Welcome**")
        st.progress(66)
        st.title("🎉 Welcome to Investimate!")
        st.write(f"Age: {st.session_state.age}, Level: {st.session_state.experience_level}")
        if st.button("Proceed to Dashboard"):
            st.session_state.step_welcome_done=True; st.rerun()

    # Step 3: DASHBOARD
    else:
        st.markdown("**Step 3 of 3: Dashboard**")
        st.progress(100)
        st.title("🏠 Dashboard")
        tool = st.sidebar.radio("Choose a tool:", ["ML Models","Rule-Based Models"])
        if tool=="ML Models":
            # Step 3: Dashboard
            if tool=="ML Models":
                # ────────────────────────────────────────────────────────────────
            # ML MODELS NAVIGATION (filtered by experience level + hard-coded risk)
            # ────────────────────────────────────────────────────────────────
            # map display names → ticker prefixes
                display_names = {
                    "Apple":       "AAPL",
                    "Microsoft":   "MSFT",
                    "KLA":         "KLAC",
                    "Qualcomm":    "QCOM",
                    "CrowdStrike": "CRWD",
                    "Palantir":    "PLTR",
                }

                # decide which buckets this user gets
                level = st.session_state.experience_level  # "Beginner"/"Intermediate"/"Experienced"
                if level == "Beginner":
                    allowed = ["Low Risk"]
                elif level == "Intermediate":
                    allowed = ["Low Risk", "Moderate Risk"]
                else:
                    allowed = ["Low Risk", "Moderate Risk", "High Risk"]

                # build a list of companies whose hard-coded risk is in the allowed buckets
                nav_labels = [
                    name
                    for name, pref in display_names.items()
                    if risk_by_company[pref] in allowed
                ]

                # render the radio and dispatch to the appropriate page
                comp_label = st.sidebar.radio("Select Company:", nav_labels)
                PAGES = {
                    "Apple":       apple_page,
                    "Microsoft":   microsoft_page,
                    "KLA":         kla_page,
                    "Qualcomm":    qualcomm_page,
                    "CrowdStrike": crowdstrike_page,
                    "Palantir":    palantir_page,
                }
                PAGES[comp_label]()

        else:
            strat = st.sidebar.radio("Select Strategy:", ["Buy & Hold","MA Crossover","RSI-Based"])
            S = {"Buy & Hold":buy_hold_page,
                 "MA Crossover":ma_crossover_page,
                 "RSI-Based":rsi_page}
            S[strat]()

# -- LOGOUT BUTTON --
if st.session_state.authenticated and st.session_state.page!="login":
    if st.button("Logout"):
        for k in ["authenticated","disclaimer_passed","step_experience_done","step_welcome_done","experience_level","age"]:
            st.session_state[k]=False
        st.session_state.page="login"
        st.rerun()
