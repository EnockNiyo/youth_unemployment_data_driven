"""
Interactive Visualizations for Real World Bank Data
Professional, interactive charts and insights
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from data_sources import RealDataFetcher, DataQualityChecker
from datetime import datetime
import numpy as np


def display_interactive_real_data_dashboard():
    """Display comprehensive interactive dashboard for real World Bank data"""

    st.title("🌍 Real-Time Global Unemployment Data")
    st.markdown("**Live data from World Bank Open Data API**")
    st.markdown("---")

    # Initialize data fetcher
    fetcher = RealDataFetcher()
    checker = DataQualityChecker()

    # Fetch all data
    with st.spinner("🔄 Fetching latest data from World Bank..."):
        all_data = fetcher.combine_all_sources()

    if 'world_bank' not in all_data or not all_data['world_bank']:
        st.error("❌ Unable to fetch World Bank data. Please check your internet connection.")
        return

    wb_data = all_data['world_bank']

    # Data freshness indicator
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if 'youth_unemployment' in wb_data and not wb_data['youth_unemployment'].empty:
            latest_year = wb_data['youth_unemployment']['year'].max()
            st.metric("Latest Data Year", latest_year, help="Most recent data available")
        else:
            st.metric("Latest Data Year", "N/A")

    with col2:
        total_indicators = len([k for k, v in wb_data.items() if v is not None and not v.empty])
        st.metric("Active Indicators", total_indicators, help="Number of available indicators")

    with col3:
        if 'youth_unemployment' in wb_data and not wb_data['youth_unemployment'].empty:
            data_points = sum(len(v) for v in wb_data.values() if v is not None)
            st.metric("Total Data Points", data_points, help="Total records fetched")
        else:
            st.metric("Total Data Points", "N/A")

    with col4:
        freshness = checker.check_data_freshness(
            wb_data.get('youth_unemployment', pd.DataFrame()))
        status_color = "🟢" if freshness['status'] == 'Fresh' else "🟡" if freshness['status'] == 'Recent' else "🔴"
        st.metric("Data Status", f"{status_color} {freshness['status']}")

    st.markdown("---")

    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Trend Analysis",
        "⚖️ Gender Comparison",
        "🌍 Regional Context",
        "📊 Indicator Correlation",
        "🔍 Detailed Data"
    ])

    # TAB 1: Trend Analysis
    with tab1:
        display_trend_analysis(wb_data)

    # TAB 2: Gender Comparison
    with tab2:
        display_gender_comparison(wb_data)

    # TAB 3: Regional Context
    with tab3:
        display_regional_context(wb_data)

    # TAB 4: Indicator Correlation
    with tab4:
        display_indicator_correlation(wb_data)

    # TAB 5: Detailed Data
    with tab5:
        display_detailed_data(wb_data)


def display_trend_analysis(wb_data):
    """Display interactive trend analysis"""
    st.subheader("📈 Youth Unemployment Trend Analysis")

    if 'youth_unemployment' not in wb_data or wb_data['youth_unemployment'].empty:
        st.warning("⚠️ Youth unemployment data not available")
        return

    df = wb_data['youth_unemployment'].copy()

    # Interactive line chart with multiple indicators
    st.markdown("#### Historical Trends (2010-2024)")

    # Combine all unemployment indicators
    combined_df = pd.DataFrame()

    if 'youth_unemployment' in wb_data and not wb_data['youth_unemployment'].empty:
        combined_df['Year'] = wb_data['youth_unemployment']['year']
        combined_df['Youth (15-24)'] = wb_data['youth_unemployment']['value']

    if 'youth_unemployment_male' in wb_data and not wb_data['youth_unemployment_male'].empty:
        male_data = wb_data['youth_unemployment_male'].set_index('year')['value']
        combined_df = combined_df.set_index('Year')
        combined_df['Male Youth'] = male_data
        combined_df = combined_df.reset_index()

    if 'youth_unemployment_female' in wb_data and not wb_data['youth_unemployment_female'].empty:
        female_data = wb_data['youth_unemployment_female'].set_index('year')['value']
        combined_df = combined_df.set_index('Year')
        combined_df['Female Youth'] = female_data
        combined_df = combined_df.reset_index()

    if 'unemployment_total' in wb_data and not wb_data['unemployment_total'].empty:
        total_data = wb_data['unemployment_total'].set_index('year')['value']
        combined_df = combined_df.set_index('Year')
        combined_df['Total (All Ages)'] = total_data
        combined_df = combined_df.reset_index()

    # Create interactive line chart
    fig = go.Figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, col in enumerate(combined_df.columns[1:]):
        fig.add_trace(go.Scatter(
            x=combined_df['Year'],
            y=combined_df[col],
            mode='lines+markers',
            name=col,
            line=dict(width=3, color=colors[i % len(colors)]),
            marker=dict(size=8),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Year: %{x}<br>' +
                         'Rate: %{y:.1f}%<br>' +
                         '<extra></extra>'
        ))

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Unemployment Rate (%)',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Key statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        if not df.empty:
            latest_rate = df['value'].iloc[0]
            st.metric(
                "Latest Rate (Youth)",
                f"{latest_rate:.1f}%",
                help="Most recent youth unemployment rate"
            )

    with col2:
        if len(df) > 1:
            change = df['value'].iloc[0] - df['value'].iloc[-1]
            st.metric(
                "Change Since 2010",
                f"{abs(change):.1f}%",
                delta=f"{change:+.1f}%",
                delta_color="inverse"
            )

    with col3:
        if not df.empty:
            avg_rate = df['value'].mean()
            st.metric(
                "Average Rate",
                f"{avg_rate:.1f}%",
                help="Average over all years"
            )


def display_gender_comparison(wb_data):
    """Display gender-based comparison"""
    st.subheader("⚖️ Gender-Based Unemployment Analysis")

    # Check if gender data is available
    has_male = 'youth_unemployment_male' in wb_data and not wb_data['youth_unemployment_male'].empty
    has_female = 'youth_unemployment_female' in wb_data and not wb_data['youth_unemployment_female'].empty

    if not (has_male and has_female):
        st.warning("⚠️ Gender-disaggregated data not available")
        return

    # Prepare data
    male_df = wb_data['youth_unemployment_male'].copy()
    female_df = wb_data['youth_unemployment_female'].copy()

    # Side-by-side comparison
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👨 Male Youth Unemployment")

        fig_male = go.Figure()
        fig_male.add_trace(go.Scatter(
            x=male_df['year'],
            y=male_df['value'],
            mode='lines+markers',
            fill='tozeroy',
            name='Male',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))

        fig_male.update_layout(
            xaxis_title='Year',
            yaxis_title='Unemployment Rate (%)',
            height=350
        )

        st.plotly_chart(fig_male, use_container_width=True)

        if not male_df.empty:
            latest_male = male_df['value'].iloc[0]
            st.metric("Latest Rate", f"{latest_male:.1f}%")

    with col2:
        st.markdown("#### 👩 Female Youth Unemployment")

        fig_female = go.Figure()
        fig_female.add_trace(go.Scatter(
            x=female_df['year'],
            y=female_df['value'],
            mode='lines+markers',
            fill='tozeroy',
            name='Female',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=8)
        ))

        fig_female.update_layout(
            xaxis_title='Year',
            yaxis_title='Unemployment Rate (%)',
            height=350
        )

        st.plotly_chart(fig_female, use_container_width=True)

        if not female_df.empty:
            latest_female = female_df['value'].iloc[0]
            st.metric("Latest Rate", f"{latest_female:.1f}%")

    # Gender gap analysis
    st.markdown("---")
    st.markdown("#### 📊 Gender Gap Analysis")

    # Merge data
    merged = pd.merge(
        male_df[['year', 'value']],
        female_df[['year', 'value']],
        on='year',
        suffixes=('_male', '_female')
    )
    merged['gap'] = merged['value_female'] - merged['value_male']

    # Gap chart
    fig_gap = go.Figure()

    fig_gap.add_trace(go.Bar(
        x=merged['year'],
        y=merged['gap'],
        name='Gender Gap',
        marker_color=np.where(merged['gap'] > 0, '#ff7f0e', '#1f77b4'),
        hovertemplate='Year: %{x}<br>Gap: %{y:.1f}pp<extra></extra>'
    ))

    fig_gap.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Parity"
    )

    fig_gap.update_layout(
        xaxis_title='Year',
        yaxis_title='Gender Gap (Percentage Points)',
        height=350,
        showlegend=False
    )

    st.plotly_chart(fig_gap, use_container_width=True)

    # Insights
    if not merged.empty:
        latest_gap = merged['gap'].iloc[0]
        avg_gap = merged['gap'].mean()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Latest Gap", f"{abs(latest_gap):.1f}pp",
                     help="Percentage point difference (Female - Male)")

        with col2:
            st.metric("Average Gap", f"{abs(avg_gap):.1f}pp",
                     help="Average over all years")

        with col3:
            higher = "Female" if latest_gap > 0 else "Male" if latest_gap < 0 else "Equal"
            st.metric("Higher Rate", higher)


def display_regional_context(wb_data):
    """Display regional/comparative context"""
    st.subheader("🌍 Rwanda in Regional Context")

    st.info("📍 **Coming Soon**: Comparative analysis with East African countries")

    # For now, show Rwanda's indicators
    if 'youth_unemployment' in wb_data and not wb_data['youth_unemployment'].empty:
        df = wb_data['youth_unemployment']

        # Circular/radar chart for different indicators
        st.markdown("#### Rwanda's Unemployment Indicators")

        indicators = []
        values = []

        for key, data in wb_data.items():
            if data is not None and not data.empty:
                indicator_name = key.replace('_', ' ').title()
                latest_value = data['value'].iloc[0]
                indicators.append(indicator_name)
                values.append(latest_value)

        if indicators:
            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=indicators,
                fill='toself',
                name='Rwanda',
                line_color='#1f77b4'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(values) * 1.2]
                    )
                ),
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)


def display_indicator_correlation(wb_data):
    """Display correlation between indicators"""
    st.subheader("📊 Indicator Relationships")

    # GDP vs Unemployment
    if 'gdp_growth' in wb_data and 'youth_unemployment' in wb_data:
        if not wb_data['gdp_growth'].empty and not wb_data['youth_unemployment'].empty:

            gdp_df = wb_data['gdp_growth'][['year', 'value']].rename(columns={'value': 'gdp_growth'})
            unemp_df = wb_data['youth_unemployment'][['year', 'value']].rename(columns={'value': 'unemployment'})

            merged = pd.merge(gdp_df, unemp_df, on='year')

            # Scatter plot
            fig = px.scatter(
                merged,
                x='gdp_growth',
                y='unemployment',
                text='year',
                title='GDP Growth vs Youth Unemployment',
                labels={
                    'gdp_growth': 'GDP Growth Rate (%)',
                    'unemployment': 'Youth Unemployment Rate (%)'
                },
                height=500
            )

            fig.update_traces(
                textposition='top center',
                marker=dict(size=12, color='#1f77b4')
            )

            st.plotly_chart(fig, use_container_width=True)

            # Calculate correlation
            if len(merged) > 2:
                corr = merged['gdp_growth'].corr(merged['unemployment'])
                st.metric(
                    "Correlation Coefficient",
                    f"{corr:.3f}",
                    help="Correlation between GDP growth and unemployment"
                )


def display_detailed_data(wb_data):
    """Display detailed data tables"""
    st.subheader("🔍 Detailed Data Explorer")

    # Indicator selector
    available_indicators = {k: v for k, v in wb_data.items() if v is not None and not v.empty}

    if not available_indicators:
        st.warning("⚠️ No data available to display")
        return

    selected_indicator = st.selectbox(
        "Select Indicator",
        options=list(available_indicators.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )

    if selected_indicator:
        df = available_indicators[selected_indicator].copy()

        # Display data table
        st.dataframe(
            df.style.format({'value': '{:.2f}%'}),
            use_container_width=True,
            hide_index=True
        )

        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"{selected_indicator}_rwanda.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    st.set_page_config(page_title="Real Data Viz", layout="wide")
    display_interactive_real_data_dashboard()
