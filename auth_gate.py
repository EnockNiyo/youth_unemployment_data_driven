"""
Authentication Gate Module
Ensures users must login before accessing dashboard
"""

import streamlit as st
from user_management import UserDatabase, AuthenticationUI


def require_authentication():
    """
    Enforce authentication before allowing dashboard access
    Returns True if authenticated, False otherwise
    """
    # Initialize database and auth UI
    if 'db' not in st.session_state:
        st.session_state.db = UserDatabase()

    if 'auth_ui' not in st.session_state:
        from user_management import AuthenticationUI
        st.session_state.auth_ui = AuthenticationUI(st.session_state.db)

    # Check if user is authenticated
    if not st.session_state.get('authenticated', False):
        display_login_gate()
        return False

    return True


def display_login_gate():
    """Display login/register gate page"""

    # Custom styling for login page
    st.markdown("""
        <style>
        .login-container {
            max-width: 500px;
            margin: 0 auto;
            padding: 2rem;
        }
        .login-header {
            text-align: center;
            color: #1f77b4;
            margin-bottom: 2rem;
        }
        .feature-box {
            background: #f0f2f6;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .feature-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div class="login-header">
            <h1>📊 Youth Unemployment Analytics</h1>
            <p style="font-size: 1.2rem; color: #666;">
                Data-Driven Insights for Rwanda's Future
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Welcome message
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.info("🔐 **Authentication Required** - Please login or register to access the dashboard")

    st.markdown("---")

    # Features showcase
    st.markdown("### ✨ Platform Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="feature-box">
                <div class="feature-icon">📈</div>
                <h4>Real-Time Data</h4>
                <p>Access live unemployment statistics from World Bank and NISR</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="feature-box">
                <div class="feature-icon">🤖</div>
                <h4>AI Forecasting</h4>
                <p>Professional ML models predict future trends with confidence intervals</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="feature-box">
                <div class="feature-icon">📊</div>
                <h4>Interactive Analytics</h4>
                <p>Explore data with powerful visualizations and filters</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Login/Register Section
    st.markdown("### 🔑 Access Dashboard")

    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

    with tab1:
        display_login_form()

    with tab2:
        display_register_form()

    st.markdown("---")

    # Footer
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p><strong>Powered by NexGen Solution</strong></p>
            <p>🇷🇼 Contributing to Rwanda's Vision 2050</p>
        </div>
    """, unsafe_allow_html=True)


def display_login_form():
    """Display login form"""
    st.markdown("#### Welcome Back!")

    with st.form("login_form_gate"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit = st.form_submit_button("🚀 Login", use_container_width=True)

        if submit:
            if username and password:
                user = st.session_state.db.authenticate_user(username, password)

                if user:
                    st.session_state.user = user
                    st.session_state.authenticated = True
                    st.success(f"✅ Welcome back, {user['full_name'] or user['username']}!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
            else:
                st.warning("⚠️ Please enter both username and password")


def display_register_form():
    """Display registration form"""
    st.markdown("#### Create Your Account")

    with st.form("register_form_gate"):
        col1, col2 = st.columns(2)

        with col1:
            username = st.text_input("Username*", placeholder="Choose a username")
            email = st.text_input("Email*", placeholder="your.email@example.com")
            password = st.text_input("Password*", type="password", placeholder="Min. 6 characters")
            full_name = st.text_input("Full Name", placeholder="Your full name")

        with col2:
            confirm_password = st.text_input("Confirm Password*", type="password", placeholder="Re-enter password")
            organization = st.text_input("Organization", placeholder="Your organization (optional)")
            phone = st.text_input("Phone Number", placeholder="+250 XXX XXX XXX")
            region = st.selectbox("Region", ["", "Kigali", "Northern", "Southern", "Eastern", "Western"])

        role = st.selectbox(
            "I am a...",
            ["user", "researcher", "policymaker"],
            help="Select your role (admin accounts require special approval)"
        )

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit = st.form_submit_button("📝 Register", use_container_width=True)

        if submit:
            # Validation
            if not all([username, email, password, confirm_password]):
                st.error("❌ Please fill in all required fields (*)")
            elif password != confirm_password:
                st.error("❌ Passwords do not match")
            elif len(password) < 6:
                st.error("❌ Password must be at least 6 characters")
            elif '@' not in email:
                st.error("❌ Please enter a valid email address")
            else:
                # Create user
                success = st.session_state.db.create_user(
                    username=username,
                    email=email,
                    password=password,
                    full_name=full_name,
                    role=role,
                    organization=organization,
                    phone=phone,
                    region=region
                )

                if success:
                    st.success("✅ Account created successfully! Please login above.")
                    st.balloons()
                else:
                    st.error("❌ Username or email already exists. Please choose different credentials.")


def display_logout_button():
    """Display logout button in sidebar"""
    if st.session_state.get('authenticated', False):
        with st.sidebar:
            st.markdown("---")
            user = st.session_state.user
            st.markdown(f"**👤 {user.get('full_name', user['username'])}**")
            st.caption(f"Role: {user['role'].title()}")

            if st.button("🚪 Logout", use_container_width=True, type="secondary"):
                st.session_state.db.log_activity(
                    user['id'],
                    "logout",
                    "User logged out"
                )
                st.session_state.user = None
                st.session_state.authenticated = False
                st.success("👋 Logged out successfully!")
                st.rerun()


if __name__ == "__main__":
    # Test the authentication gate
    st.set_page_config(page_title="Auth Test", layout="wide")

    if require_authentication():
        st.success("✅ Authenticated! Dashboard would load here.")
        display_logout_button()
    else:
        st.stop()
