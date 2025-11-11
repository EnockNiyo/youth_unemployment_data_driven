"""
Demo Application - Enhanced Features
Test the new features independently before integrating into main app
"""

import streamlit as st
from app_integration import integrate_with_existing_app
import pandas as pd

# Page config
st.set_page_config(
    page_title="Youth Unemployment Analytics - Enhanced Demo",
    page_icon="📊",
    layout="wide"
)

# Load CSS if available
try:
    with open('style.css') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# Title
st.markdown(
    "<h1 style='text-align: center; color: #302;'>Youth Unemployment Analytics - Enhanced Demo</h1>",
    unsafe_allow_html=True
)

# Initialize and get selected page
selected_page = integrate_with_existing_app()

# Display demo content for original pages
if selected_page in ["🏠 Home", "📊 Dashboard"]:
    st.markdown("---")

    st.header("📊 Demo Dashboard")

    st.write("""
    Welcome to the **Enhanced Youth Unemployment Analytics Platform**!

    This demo showcases the new features:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("### 🔐 User Authentication\n\n"
                "- Secure login/register\n"
                "- User profiles\n"
                "- Role-based access")

    with col2:
        st.success("### 🌍 Real-Time Data\n\n"
                   "- World Bank API\n"
                   "- NISR data\n"
                   "- Live statistics")

    with col3:
        st.warning("### 📝 Data Contribution\n\n"
                   "- Submit employment data\n"
                   "- Post job opportunities\n"
                   "- Share success stories")

    st.markdown("---")

    # Demo statistics
    st.subheader("🎯 Platform Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Users", "0", help="Register to start!")

    with col2:
        st.metric("Data Sources", "3", help="World Bank, NISR, User Contributions")

    with col3:
        st.metric("Contributions", "0", help="Login and contribute data")

    with col4:
        st.metric("Predictions", "Enabled", help="ML-powered forecasting")

    st.markdown("---")

    # Getting started guide
    st.subheader("🚀 Getting Started")

    with st.expander("1️⃣ Create an Account", expanded=True):
        st.write("""
        1. Click **'Login/Register'** in the sidebar
        2. Fill in the registration form
        3. Choose your role (user, researcher, policymaker, admin)
        4. Submit and login
        """)

    with st.expander("2️⃣ Explore Real-Time Data"):
        st.write("""
        1. Navigate to **'Real-Time Data'** in sidebar
        2. View latest statistics from World Bank
        3. Compare Rwanda's youth unemployment trends
        4. Check data freshness and quality
        """)

    with st.expander("3️⃣ Contribute Your Data"):
        st.write("""
        1. Login to your account
        2. Go to **'Contribute Data'**
        3. Choose contribution type:
           - Personal employment status
           - Job opportunity
           - Skills gap report
           - Training feedback
           - Success story
        4. Fill and submit the form
        """)

    with st.expander("4️⃣ View Predictions"):
        st.write("""
        1. Navigate to **'Future Predictions'**
        2. Use filters to refine predictions
        3. View forecasts by region and demographics
        4. Export results for policy decisions
        """)

    st.markdown("---")

    # Features comparison
    st.subheader("✨ New vs Original Features")

    comparison_data = {
        "Feature": [
            "Data Sources",
            "User Management",
            "Data Contribution",
            "Real-Time Updates",
            "Role-Based Access",
            "API Integration"
        ],
        "Original": [
            "Static CSV files",
            "❌ None",
            "❌ None",
            "❌ None",
            "❌ None",
            "❌ None"
        ],
        "Enhanced": [
            "World Bank API + NISR + User Data",
            "✅ Full authentication system",
            "✅ 5 contribution types",
            "✅ Live data from World Bank",
            "✅ User/Researcher/Policymaker/Admin",
            "✅ World Bank Open Data API"
        ]
    }

    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Quick stats
    st.subheader("📈 Demo Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Data Sources Available:**")
        st.write("- 🌍 World Bank Open Data API")
        st.write("- 🇷🇼 NISR Labor Force Survey")
        st.write("- 👥 User Contributions")
        st.write("- 📊 ILO Employment Database (coming soon)")

    with col2:
        st.write("**Contribution Types:**")
        st.write("- 💼 Personal Employment Status")
        st.write("- 📋 Job Opportunity Postings")
        st.write("- 🎯 Skills Gap Reports")
        st.write("- ⭐ Training Program Feedback")
        st.write("- 🏆 Success Stories")

    st.markdown("---")

    # Call to action
    st.info("👉 **Ready to get started?** Click 'Login/Register' in the sidebar to create your account!")

elif selected_page == "🔮 Future Predictions":
    st.header("🔮 Future Predictions")
    st.info("This page will show the original future predictions functionality from home.py")
    st.write("The predictions model uses historical data to forecast youth unemployment trends.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Youth Unemployment Prediction Platform - Enhanced Version</strong></p>
    <p>Powered by NexGen Solution | Data from World Bank, NISR, and User Contributions</p>
    <p>© 2025 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
