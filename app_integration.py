"""
Integration Module for Enhanced Features
This module adds real data sources and user management to the application
"""

import streamlit as st
import sys
import os
import json
import pandas as pd

# Compatibility helper for Streamlit rerun (some Streamlit versions removed experimental_rerun)
def safe_rerun():
    try:
        # Preferred for older/newer versions if available
        if hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()
            return
        if hasattr(st, 'rerun'):
            st.rerun()
            return
    except Exception:
        pass

    # Final fallback: stop execution (user will need to interact to rerun)
    try:
        st.stop()
    except Exception:
        return

# Import our new modules
from data_sources import RealDataFetcher, DataQualityChecker
from user_management import UserDatabase, AuthenticationUI, display_user_sidebar
from data_contribution import DataContribution
from interactive_real_data_viz import display_interactive_real_data_dashboard


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
    """Display real-time data from international sources with interactive visualizations"""
    display_interactive_real_data_dashboard()


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

    st.markdown("---")
    st.subheader("User Management")

    # Fetch all users and allow admin to export
    users = st.session_state.db.get_all_users()
    if users:
        users_df = pd.DataFrame(users)

        # Filters
        roles = sorted(users_df['role'].dropna().unique().tolist())
        selected_roles = st.multiselect("Filter by role", options=roles, default=roles)
        active_only = st.checkbox("Active users only", value=True)

        filtered = users_df[users_df['role'].isin(selected_roles)]
        if active_only:
            filtered = filtered[filtered['is_active'] == True]

        st.dataframe(filtered.drop(columns=['is_active']))

        csv = filtered.to_csv(index=False).encode('utf-8')
        st.download_button("Export Users CSV", data=csv, file_name="users_report.csv", mime="text/csv")
    else:
        st.info("No users found in the database.")

    st.markdown("---")
    st.subheader("Contributions")

    contributions = st.session_state.db.get_all_contributions()
    if contributions:
        contrib_df = pd.DataFrame(contributions)

        # Show data column as JSON string for table display
        contrib_df_display = contrib_df.copy()
        contrib_df_display['data'] = contrib_df_display['data'].apply(lambda x: json.dumps(x, ensure_ascii=False))

        status_opts = sorted(contrib_df_display['status'].dropna().unique().tolist())
        sel_status = st.multiselect("Filter by status", options=status_opts, default=status_opts)

        filtered_contrib = contrib_df_display[contrib_df_display['status'].isin(sel_status)]

        st.dataframe(filtered_contrib)

        csvc = filtered_contrib.to_csv(index=False).encode('utf-8')
        st.download_button("Export Contributions CSV", data=csvc, file_name="contributions_report.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("Manage Contribution Status")

        # Provide a simple editor: choose an ID, view details, change status
        try:
            contrib_ids = contrib_df['id'].tolist()
        except Exception:
            contrib_ids = []

        if contrib_ids:
            selected_id = st.selectbox("Select Contribution ID to edit", options=contrib_ids)

            # Find the original row (with parsed data)
            selected_row = None
            try:
                selected_row = contrib_df[contrib_df['id'] == selected_id].iloc[0]
            except Exception:
                selected_row = None

            if selected_row is not None:
                st.write("**Submitted by:**", selected_row.get('username'))
                st.write("**Full name:**", selected_row.get('full_name'))
                st.write("**Type:**", selected_row.get('type'))
                st.write("**Submitted at:**", selected_row.get('submitted_at'))
                st.write("**Current status:**", selected_row.get('status'))
                st.write("**Data:**")
                st.json(selected_row.get('data'))

                status_options = ["pending", "approved", "rejected"]
                current_index = 0
                try:
                    current_index = status_options.index(selected_row.get('status'))
                except Exception:
                    current_index = 0

                new_status = st.selectbox("New status", options=status_options, index=current_index)

                if st.button("Update Status"):
                    ok = st.session_state.db.update_contribution_status(selected_id, new_status)
                    if ok:
                        st.success(f"Contribution {selected_id} status updated to {new_status}")
                        safe_rerun()
                    else:
                        st.error("Failed to update contribution status")
        else:
            st.info("No contribution IDs available to edit.")
    else:
        st.info("No contributions found.")

    st.markdown("---")
    st.subheader("Manage Users")

    all_users = st.session_state.db.get_all_users()
    if all_users:
        users_df = pd.DataFrame(all_users)

        try:
            user_ids = users_df['id'].tolist()
        except Exception:
            user_ids = []

        if user_ids:
            selected_user_id = st.selectbox("Select User ID to edit", options=user_ids)
            selected_user = None
            try:
                selected_user = users_df[users_df['id'] == selected_user_id].iloc[0].to_dict()
            except Exception:
                selected_user = None

            if selected_user is not None:
                st.write("**Username:**", selected_user.get('username'))
                st.write("**Created At:**", selected_user.get('created_at'))

                with st.form("edit_user_form"):
                    full_name = st.text_input("Full Name", value=selected_user.get('full_name') or "")
                    email = st.text_input("Email", value=selected_user.get('email') or "")
                    role = st.selectbox("Role", options=["user", "researcher", "policymaker", "admin", "Local government official", "Private sector (Employer)", "Education (Vocational officer)", "NGO representative"], index=0 if not selected_user.get('role') else ["user", "researcher", "policymaker", "admin", "Local government official", "Private sector (Employer)", "Education (Vocational officer)", "NGO representative"].index(selected_user.get('role')) if selected_user.get('role') in ["user", "researcher", "policymaker", "admin", "Local government official", "Private sector (Employer)", "Education (Vocational officer)", "NGO representative"] else 0)
                    organization = st.text_input("Organization", value=selected_user.get('organization') or "")
                    phone = st.text_input("Phone", value=selected_user.get('phone') or "")
                    region = st.selectbox("Region", options=["", "Kigali", "Northern", "Southern", "Eastern", "Western"], index=0 if not selected_user.get('region') else ["", "Kigali", "Northern", "Southern", "Eastern", "Western"].index(selected_user.get('region')) if selected_user.get('region') in ["", "Kigali", "Northern", "Southern", "Eastern", "Western"] else 0)
                    is_active = st.checkbox("Active", value=bool(selected_user.get('is_active')))

                    submit_user = st.form_submit_button("Update User")

                    if submit_user:
                        updates = {
                            'full_name': full_name,
                            'email': email,
                            'role': role,
                            'organization': organization,
                            'phone': phone,
                            'region': region,
                            'is_active': 1 if is_active else 0
                        }

                        ok = st.session_state.db.update_user_profile(selected_user_id, updates)
                        if ok:
                            st.success(f"User {selected_user.get('username')} updated")
                            safe_rerun()
                        else:
                            st.error("Failed to update user")
        else:
            st.info("No user IDs available to edit.")
    else:
        st.info("No users found to manage.")


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
