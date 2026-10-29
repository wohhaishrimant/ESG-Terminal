import streamlit as st
import pandas as pd
import plotly.express as px
import os

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="ESG Analytics & Gemini AI Advisory Terminal",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .phase-card { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        border-top: 5px solid #2e7d32; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.04); 
        margin-bottom: 20px; 
        min-height: 240px;
    }
    .phase-title { font-weight: 700; color: #1b5e20; margin-bottom: 10px; font-size: 1.1rem; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SESSION STATE INITIALIZATION
# ==============================================================================
if 'working_df' not in st.session_state:
    st.session_state['working_df'] = pd.DataFrame({
        'Company': [
            'Reliance Industries', 'TCS', 'HDFC Bank', 'ICICI Bank', 'Infosys', 
            'ITC', 'Larsen & Toubro', 'Bharti Airtel', 'State Bank of India', 'Wipro',
            'HCLTech', 'Maruti Suzuki', 'Axis Bank', 'Titan Company', 'Tech Mahindra'
        ],
        'Sector': [
            'Energy', 'Information Technology', 'Financials', 'Financials', 'Information Technology',
            'FMCG', 'Infrastructure', 'Telecommunications', 'Financials', 'Information Technology',
            'Information Technology', 'Automobile', 'Financials', 'Consumer Discretionary', 'Information Technology'
        ],
        'Scope_1_2_Emissions_kt': [4500.2, 76.6, 25.4, 28.1, 65.2, 110.5, 340.0, 190.2, 45.0, 54.0, 88.4, 210.0, 18.2, 12.0, 45.1],
        'Renewable_Energy_Pct': [35.0, 79.0, 45.0, 42.0, 62.5, 50.0, 30.0, 28.0, 20.0, 58.0, 45.0, 38.0, 40.0, 35.0, 52.0],
        'Female_Workforce_Pct': [18.2, 35.3, 26.5, 28.0, 38.6, 30.0, 14.2, 22.1, 25.4, 36.2, 29.5, 9.5, 31.2, 45.0, 33.4],
        'Avg_Training_Hours': [30, 55, 40, 42, 68, 35, 28, 32, 25, 60, 45, 38, 40, 50, 50],
        'Board_Independence_Pct': [50.0, 62.5, 75.0, 70.0, 70.0, 58.0, 64.0, 60.0, 55.0, 66.7, 63.6, 53.0, 72.0, 61.0, 72.7],
        'Women_On_Board_Pct': [16.7, 25.0, 27.3, 30.0, 30.0, 20.0, 18.2, 22.0, 20.0, 27.3, 22.2, 15.0, 25.0, 33.0, 27.3]
    })

# ==============================================================================
# 3. SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.header("⚙️ Data & Model Controls")
st.sidebar.subheader("🤖 Gemini AI Advisory Engine")
gemini_api_key = st.sidebar.text_input("Gemini API Key (Optional)", type="password", key="sidebar_gemini_key")

st.sidebar.markdown("---")
st.sidebar.subheader("➕ Add Custom Peer")
with st.sidebar.expander("Open Custom Entry Form"):
    with st.form("custom_company_form"):
        c_name = st.text_input("Company Name")
        c_sector = st.selectbox("Sector", ["Information Technology", "Financials", "Energy", "FMCG", "Automobile", "Infrastructure", "Other"])
        c_emissions = st.number_input("Scope 1&2 Emissions (kt CO2e)", min_value=0.1, value=40.0)
        c_renewable = st.number_input("Renewable Energy (%)", min_value=0.0, max_value=100.0, value=50.0)
        c_female = st.number_input("Female Workforce (%)", min_value=0.0, max_value=100.0, value=30.0)
        c_training = st.number_input("Avg Training Hours", min_value=0, value=45)
        c_board_ind = st.number_input("Board Independence (%)", min_value=0.0, max_value=100.0, value=65.0)
        c_women_board = st.number_input("Women on Board (%)", min_value=0.0, max_value=100.0, value=25.0)
        
        submit_btn = st.form_submit_button("Add Peer")
        if submit_btn and c_name:
            df_state = st.session_state['working_df']
            if c_name in df_state['Company'].values:
                st.sidebar.warning(f"{c_name} already exists.")
            else:
                custom_row = {
                    'Company': c_name, 'Sector': c_sector, 'Scope_1_2_Emissions_kt': c_emissions,
                    'Renewable_Energy_Pct': c_renewable, 'Female_Workforce_Pct': c_female,
                    'Avg_Training_Hours': c_training, 'Board_Independence_Pct': c_board_ind,
                    'Women_On_Board_Pct': c_women_board
                }
                st.session_state['working_df'] = pd.concat([df_state, pd.DataFrame([custom_row])], ignore_index=True)
                st.sidebar.success(f"Added {c_name} to active dataset!")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Active Dataset Filter")
sector_list = ["All Sectors"] + list(st.session_state['working_df']['Sector'].unique())
selected_sector = st.sidebar.selectbox("Filter by Industry Sector", sector_list, key="sidebar_sector_filter")

if selected_sector != "All Sectors":
    active_df = st.session_state['working_df'][st.session_state['working_df']['Sector'] == selected_sector].copy()
else:
    active_df = st.session_state['working_df'].copy()

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Pillar Weighting Matrix")
w_e = st.sidebar.slider("Environment Weight (E)", 0.0, 1.0, 0.4, 0.05, key="slider_we")
w_s = st.sidebar.slider("Social Weight (S)", 0.0, 1.0, 0.3, 0.05, key="slider_ws")
w_g = st.sidebar.slider("Governance Weight (G)", 0.0, 1.0, 0.3, 0.05, key="slider_wg")

# ==============================================================================
# 4. NORMALIZATION & COMPOSITE SCORING ENGINE
# ==============================================================================
def run_scoring_pipeline(df, weight_e, weight_s, weight_g):
    if df.empty:
        return df

    max_e = df['Scope_1_2_Emissions_kt'].max()
    min_e = df['Scope_1_2_Emissions_kt'].min()
    if max_e == min_e:
        df['Scope_Score'] = 1.0
    else:
        df['Scope_Score'] = (max_e - df['Scope_1_2_Emissions_kt']) / (max_e - min_e)

    def min_max_scale(series):
        if series.max() == series.min():
            return series * 0 + 0.5
        return (series - series.min()) / (series.max() - series.min())

    df['Renewable_Score'] = min_max_scale(df['Renewable_Energy_Pct'])
    df['Female_Score'] = min_max_scale(df['Female_Workforce_Pct'])
    df['Training_Score'] = min_max_scale(df['Avg_Training_Hours'])
    df['Board_Ind_Score'] = min_max_scale(df['Board_Independence_Pct'])
    df['Women_Board_Score'] = min_max_scale(df['Women_On_Board_Pct'])

    df['E_Score'] = (df['Scope_Score'] + df['Renewable_Score']) / 2
    df['S_Score'] = (df['Female_Score'] + df['Training_Score']) / 2
    df['G_Score'] = (df['Board_Ind_Score'] + df['Women_Board_Score']) / 2

    total_weights = weight_e + weight_s + weight_g if (weight_e + weight_s + weight_g) > 0 else 1.0
    
    df['Composite_ESG_Score'] = (
        (weight_e * df['E_Score'] + weight_s * df['S_Score'] + weight_g * df['G_Score']) / total_weights
    ) * 100

    return df.sort_values(by='Composite_ESG_Score', ascending=False).reset_index(drop=True)

scored_df = run_scoring_pipeline(active_df, w_e, w_s, w_g)

# ==============================================================================
# 5. DASHBOARD UI & TABS
# ==============================================================================
st.title("🌿 ESG & Sustainability Analytics Terminal")
st.markdown("Benchmarking **SEBI BRSR Disclosures** across Top Indian Listed Entities & Modeling Decarbonization Pathways.")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Sector Leaderboard", 
    "📊 Peer Benchmarking", 
    "⚡ Decarbonization Simulator", 
    "🤖 Gemini AI ESG Strategy & Roadmap"
])

# --- TAB 1: SECTOR LEADERBOARD ---
with tab1:
    st.subheader("Sector Rankings by Composite ESG Score")
    if scored_df.empty:
        st.warning("No companies match the current filter selection.")
    else:
        fig_rank = px.bar(
            scored_df, 
            x='Company', 
            y='Composite_ESG_Score',
            color='Composite_ESG_Score',
            color_continuous_scale='Greens',
            labels={'Composite_ESG_Score': 'ESG Score (0-100)'},
            text_auto='.1f',
            hover_data=['Sector', 'E_Score', 'S_Score', 'G_Score']
        )
        fig_rank.update_layout(xaxis_title="", yaxis_range=[0, 100], plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_rank, use_container_width=True)

        st.markdown("### Detailed Score Breakdown")
        display_cols = scored_df[['Company', 'Sector', 'Composite_ESG_Score', 'E_Score', 'S_Score', 'G_Score']].copy()
        display_cols[['E_Score', 'S_Score', 'G_Score']] = display_cols[['E_Score', 'S_Score', 'G_Score']].round(2)
        display_cols['Composite_ESG_Score'] = display_cols['Composite_ESG_Score'].round(1)
        st.dataframe(display_cols, use_container_width=True, hide_index=True)

# --- TAB 2: PEER BENCHMARKING ---
with tab2:
    st.subheader("Company Deep-Dive vs Sector Median")
    if scored_df.empty:
        st.warning("No data available.")
    else:
        selected_company = st.selectbox("Select Target Company:", scored_df['Company'].unique(), key="bench_select_company")
        comp_data = scored_df[scored_df['Company'] == selected_company].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall ESG Score", f"{comp_data['Composite_ESG_Score']:.1f} / 100")
        col2.metric("Environmental (E)", f"{comp_data['E_Score']:.2f}")
        col3.metric("Social (S)", f"{comp_data['S_Score']:.2f}")
        col4.metric("Governance (G)", f"{comp_data['G_Score']:.2f}")
        
        st.divider()
        
        metrics_categories = ['Scope_Score', 'Renewable_Score', 'Female_Score', 'Training_Score', 'Board_Ind_Score', 'Women_Board_Score']
        comp_values = [comp_data[c] for c in metrics_categories]
        median_values = [scored_df[c].median() for c in metrics_categories]
        
        fig_radar = px.line_polar(
            r=comp_values + [comp_values[0]],
            theta=metrics_categories + [metrics_categories[0]],
            line_close=True,
            title=f"{selected_company} Metrics vs. Industry Median",
            color_discrete_sequence=['#2e7d32']
        )
        fig_radar.add_trace(
            px.line_polar(
                r=median_values + [median_values[0]], 
                theta=metrics_categories + [metrics_categories[0]]
            ).data[0]
        )
        fig_radar.data[1].name = "Industry Median"
        fig_radar.data[1].line.dash = "dash"
        fig_radar.data[1].line.color = "#757575"
        st.plotly_chart(fig_radar, use_container_width=True)

# --- TAB 3: DECARBONIZATION SIMULATOR ---
with tab3:
    st.subheader("Scope 2 Decarbonization & Financial Simulator")
    if scored_df.empty:
        st.warning("No data available.")
    else:
        col_input, col_output = st.columns([1, 2])
        
        with col_input:
            sim_company = st.selectbox("Select Target Company:", scored_df['Company'].unique(), key="sim_select_company")
            reduction_target_pct = st.slider("Scope 2 Target Reduction (%)", min_value=5, max_value=100, value=25, step=5, key="sim_slider_target")
            
            st.markdown("### Cost & Transition Assumptions")
            capex_rate = st.slider("Capex Investment ($M per kt CO2e)", 0.5, 2.5, 1.2, 0.1, key="sim_slider_capex")
            savings_rate = st.slider("Annual Savings ($M per kt CO2e)", 0.05, 0.5, 0.15, 0.01, key="sim_slider_savings")
            
        with col_output:
            match_sim = scored_df[scored_df['Company'] == sim_company]
            if not match_sim.empty:
                baseline_emissions = match_sim['Scope_1_2_Emissions_kt'].values[0]
                
                scope2_baseline = baseline_emissions * 0.9
                emissions_avoided = scope2_baseline * (reduction_target_pct / 100.0)
                required_capex = emissions_avoided * capex_rate
                annual_savings = emissions_avoided * savings_rate
                payback_years = (required_capex / annual_savings) if annual_savings > 0 else 0
                
                st.markdown(f"### Financial & Carbon Results: **{sim_company}**")
                
                r1, r2 = st.columns(2)
                r1.metric("Carbon Avoided", f"{emissions_avoided:.2f} kt CO2e")
                r1.metric("Required Capex Investment", f"${required_capex:.2f} Million")
                
                r2.metric("Estimated Annual Savings", f"${annual_savings:.2f} Million")
                r2.metric("Payback Period", f"{payback_years:.1f} Years")
            else:
                st.info("Select a valid company to view simulation results.")

# --- TAB 4: GEMINI AI & RULE-BASED STRATEGY ROADMAP ---
with tab4:
    st.subheader("🤖 Strategic Transformation Roadmap & Advisory")
    st.markdown("Tailored framework mapping out diagnostic findings, interim actions, and long-term targets.")
    
    if scored_df.empty:
        st.warning("No data available.")
    else:
        roadmap_company = st.selectbox("Select Target Company for Roadmap:", scored_df['Company'].unique(), key="ai_roadmap_select_company")
        r_data_match = scored_df[scored_df['Company'] == roadmap_company]
        
        if not r_data_match.empty:
            r_data = r_data_match.iloc[0]
            
            med_emissions = scored_df['Scope_1_2_Emissions_kt'].median()
            med_renewable = scored_df['Renewable_Energy_Pct'].median()
            med_female = scored_df['Female_Workforce_Pct'].median()
            med_board = scored_df['Board_Independence_Pct'].median()

            # Check if user triggered Gemini AI generation
            if gemini_api_key and HAS_GEMINI:
                if st.button("✨ Generate Live Gemini AI Consulting Report", key="btn_generate_gemini"):
                    with st.spinner("Calling Google Gemini for deep strategic analysis..."):
                        try:
                            genai.configure(api_key=gemini_api_key)
                            model = genai.GenerativeModel('gemini-1.5-flash')
                            prompt = (
                                f"Act as a Senior ESG Strategy Partner specializing in Indian SEBI BRSR Core. "
                                f"Provide a 4-phase transformation strategy for {r_data['Company']} ({r_data['Sector']} sector).\n"
                                f"Metrics:\n"
                                f"- Scope 1 & 2 Emissions: {r_data['Scope_1_2_Emissions_kt']} kt CO2e (Median: {med_emissions:.1f})\n"
                                f"- Renewable Energy Share: {r_data['Renewable_Energy_Pct']}% (Median: {med_renewable}%)\n"
                                f"- Female Workforce: {r_data['Female_Workforce_Pct']}% (Median: {med_female}%)\n"
                                f"- Board Independence: {r_data['Board_Independence_Pct']}% (Median: {med_board}%)\n\n"
                                f"Format cleanly under these exact 4 headings using markdown:\n"
                                f"1. Phase 1: Diagnostic & Gap Analysis\n"
                                f"2. Phase 2: Immediate Interventions (Months 0-6)\n"
                                f"3. Phase 3: Structural Capital Allocation (Months 6-18)\n"
                                f"4. Phase 4: Long-Term Governance & Target Maturity"
                            )
                            response = model.generate_content(prompt)
                            st.markdown(response.text)
                        except Exception as e:
                            st.error(f"Gemini API Error: {e}. Falling back to structured view.")
            
            # Default structured card layout view
            st.markdown(f"### 📍 Executive Action Plan: **{roadmap_company}**")
            
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                st.markdown(f"""
                    <div class="phase-card">
                        <div class="phase-title">🔍 Phase 1: Diagnostic & Audit</div>
                        <ul>
                            <li><b>Emissions Baseline:</b> Scope 1 & 2 stands at {r_data['Scope_1_2_Emissions_kt']} kt CO2e (Peer Median: {med_emissions:.1f} kt).</li>
                            <li><b>Action:</b> Conduct asset-level energy mapping to isolate high-emission intensity centers.</li>
                        </ul>
                    </div>
                    <div class="phase-card">
                        <div class="phase-title">⚡ Phase 2: Quick Wins (Months 0–6)</div>
                        <ul>
                            <li><b>Efficiency:</b> {'Deploy intelligent smart-metering and HVAC retrofits.' if r_data['Scope_1_2_Emissions_kt'] > med_emissions else 'Lock in low-emission baselines across active assets.'}</li>
                            <li><b>Workforce Diversity:</b> Launch structured DEI hiring frameworks to improve female representation ({r_data['Female_Workforce_Pct']}% current).</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
                
            with col_c2:
                st.markdown(f"""
                    <div class="phase-card">
                        <div class="phase-title">🏗️ Phase 3: Structural Transition (Months 6–18)</div>
                        <ul>
                            <li><b>Renewable Scaling:</b> {'Execute Open Access Green Power PPAs to scale clean share beyond ' + str(r_data['Renewable_Energy_Pct']) + '%.' if r_data['Renewable_Energy_Pct'] < med_renewable else 'Scale captive microgrids for 24/7 carbon-free energy.'}</li>
                            <li><b>Supply Chain:</b> Embed SEBI BRSR vendor screening scorecards into procurement workflows.</li>
                        </ul>
                    </div>
                    <div class="phase-card">
                        <div class="phase-title">🏛️ Phase 4: Governance Maturity (Long-Term)</div>
                        <ul>
                            <li><b>Board Diversification:</b> {'Appoint non-executive independent directors to boost board independence (' + str(r_data['Board_Independence_Pct']) + '%).' if r_data['Board_Independence_Pct'] < med_board else 'Maintain top-decile corporate governance standing.'}</li>
                            <li><b>Assurance:</b> Prepare for third-party reasonable assurance under SEBI BRSR Core.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)