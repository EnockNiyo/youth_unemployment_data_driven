"""
Integration Module for Enhanced Features
This module adds real data sources and user management to the application
"""

import streamlit as st
import sys
import os

# Import our new modules
from data_sources import RealDataFetcher, DataQualityChecker, display_real_data_dashboard
from user_management import UserDatabase, AuthenticationUI, display_user_sidebar
from data_contribution import DataContribution


def initialize_app():
    """Initialize the application with enhanced features"""

    # Initialize database
    if 'db' not in st.session_state:
        st.session_state.db = UserDatabase()

    # Initialize authentication UI
    if 'auth_ui' not in st.session_state:
        st.session_state.auth_ui = AuthenticationUI(st.session_state.db)

    # Initialize data fetcher
    if 'data_fetcher' not in st.session_state:
        st.session_state.data_fetcher = RealDataFetcher()

    # Initialize data contribution
    if 'data_contribution' not in st.session_state:
        st.session_state.data_contribution = DataContribution(st.session_state.db)


def display_enhanced_sidebar():
    """Enhanced sidebar with authentication and navigation"""

    st.sidebar.title("🌟 Navigation")

    # Check authentication status
    is_authenticated = st.session_state.get('authenticated', False)

    if is_authenticated:
        user = st.session_state.user
        st.sidebar.success(f"👤 Welcome, {user.get('full_name', user['username'])}")

        # Enhanced menu options
        menu_options = [
            "🏠 Home",
            "📊 Dashboard",
            "🔮 Future Predictions",
            "🌍 Real-Time Data",
            "📝 Contribute Data",
            "👤 My Profile",
            "📋 My Contributions",
        ]

        # Admin options
        if user.get('role') == 'admin':
            menu_options.extend(["⚙️ Admin Panel", "📈 User Statistics"])

    else:
        menu_options = [
            "🏠 Home",
            "📊 Dashboard",
            "🔮 Future Predictions",
            "🌍 Real-Time Data",
            "🔐 Login/Register"
        ]

    selected = st.sidebar.selectbox("Select Page", menu_options)

    # Display user info
    display_user_sidebar()

    return selected


def display_real_data_page():
    """Display real-time data from international sources"""
    st.title("🌍 Real-Time Data Sources")
    st.markdown("---")

    display_real_data_dashboard()


def display_contribution_page():
    """Display data contribution page"""
    if not st.session_state.get('authenticated', False):
        st.warning("🔒 Please login to contribute data")
        st.info("Go to Login/Register page to create an account")
        return

    st.session_state.data_contribution.display_contribution_form()

    st.markdown("---")

    st.session_state.data_contribution.display_user_contributions()


def display_profile_page():
    """Display user profile page"""
    if not st.session_state.get('authenticated', False):
        st.warning("🔒 Please login to view your profile")
        return

    st.session_state.auth_ui.display_user_profile()


def display_my_contributions_page():
    """Display user's contributions"""
    if not st.session_state.get('authenticated', False):
        st.warning("🔒 Please login to view your contributions")
        return

    st.title("📋 My Contributions")
    st.session_state.data_contribution.display_user_contributions()


def display_login_page():
    """Display login/register page"""
    st.session_state.auth_ui.display_auth_page()


def display_admin_panel():
    """Display admin panel (admin only)"""
    if not st.session_state.get('authenticated', False):
        st.warning("🔒 Please login to access admin panel")
        return

    user = st.session_state.user
    if user.get('role') != 'admin':
        st.error("❌ Access denied. Admin privileges required.")
        return

    st.title("⚙️ Admin Panel")

    # Get user statistics
    stats = st.session_state.db.get_user_statistics()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Users", stats.get('total_users', 0))

    with col2:
        st.metric("Total Contributions", stats.get('total_contributions', 0))

    with col3:
        users_by_role = stats.get('users_by_role', {})
        st.metric("Roles", len(users_by_role))

    # Display users by role
    st.subheader("Users by Role")
    if users_by_role:
        import pandas as pd
        role_df = pd.DataFrame(list(users_by_role.items()), columns=['Role', 'Count'])
        st.bar_chart(role_df.set_index('Role'))


def display_user_statistics():
    """Display user statistics dashboard"""
    if not st.session_state.get('authenticated', False):
        st.warning("🔒 Please login to view statistics")
        return

    user = st.session_state.user
    if user.get('role') not in ['admin', 'researcher', 'policymaker']:
        st.error("❌ Access denied. Insufficient privileges.")
        return

    st.title("📈 User Statistics")
    st.write("Comprehensive overview of platform usage and contributions")

    # Get statistics
    stats = st.session_state.db.get_user_statistics()

    # Display metrics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Platform Overview")
        st.write(f"**Total Users:** {stats.get('total_users', 0)}")
        st.write(f"**Total Contributions:** {stats.get('total_contributions', 0)}")

    with col2:
        st.subheader("Users by Role")
        users_by_role = stats.get('users_by_role', {})
        for role, count in users_by_role.items():
            st.write(f"**{role.title()}:** {count}")


def get_enhanced_menu_function(selected_page: str):
    """Route to the appropriate page function"""

    routing = {
        "🏠 Home": None,  # Return None to show original dashboard
        "📊 Dashboard": None,
        "🔮 Future Predictions": "future_predictions",
        "🌍 Real-Time Data": display_real_data_page,
        "📝 Contribute Data": display_contribution_page,
        "👤 My Profile": display_profile_page,
        "📋 My Contributions": display_my_contributions_page,
        "🔐 Login/Register": display_login_page,
        "⚙️ Admin Panel": display_admin_panel,
        "📈 User Statistics": display_user_statistics,
    }

    return routing.get(selected_page)


# Quick integration function for home.py
def integrate_with_existing_app():
    """
    Quick integration with existing home.py
    Add this at the top of home.py after imports
    """

    initialize_app()

    # Display enhanced sidebar
    selected_page = display_enhanced_sidebar()

    # Get page function
    page_function = get_enhanced_menu_function(selected_page)

    # If special page is selected, display it instead of dashboard
    if page_function is not None:
        if callable(page_function):
            page_function()
            st.stop()  # Stop execution of original dashboard
        elif page_function == "future_predictions":
            # Let the original code handle this
            pass

    # Otherwise, continue with original dashboard
    return selected_page


if __name__ == "__main__":
    integrate_with_existing_app()
