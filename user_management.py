"""
User Management System for Youth Unemployment Platform
Includes authentication, profiles, and data contribution
"""

import streamlit as st
import pandas as pd
import json
import hashlib
import os
from datetime import datetime
from typing import Optional, Dict, List
import sqlite3

class UserDatabase:
    """SQLite database for user management"""

    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                role TEXT DEFAULT 'user',
                organization TEXT,
                phone TEXT,
                region TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')

        # User contributions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_contributions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                contribution_type TEXT,
                data_json TEXT,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending',
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # User activity log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                activity_type TEXT,
                description TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        conn.commit()
        conn.close()

    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def create_user(self, username: str, email: str, password: str,
                    full_name: str = "", role: str = "user",
                    organization: str = "", phone: str = "",
                    region: str = "") -> bool:
        """Create a new user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            password_hash = self.hash_password(password)

            cursor.execute('''
                INSERT INTO users (username, email, password_hash, full_name,
                                   role, organization, phone, region)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (username, email, password_hash, full_name, role,
                  organization, phone, region))

            conn.commit()
            conn.close()

            return True

        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            st.error(f"Error creating user: {e}")
            return False

    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user and return user data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            password_hash = self.hash_password(password)

            cursor.execute('''
                SELECT id, username, email, full_name, role, organization,
                       phone, region, created_at
                FROM users
                WHERE username = ? AND password_hash = ? AND is_active = 1
            ''', (username, password_hash))

            result = cursor.fetchone()

            if result:
                # Update last login
                cursor.execute('''
                    UPDATE users SET last_login = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (result[0],))
                conn.commit()

                # Log activity
                self.log_activity(result[0], "login", "User logged in")

                user_data = {
                    'id': result[0],
                    'username': result[1],
                    'email': result[2],
                    'full_name': result[3],
                    'role': result[4],
                    'organization': result[5],
                    'phone': result[6],
                    'region': result[7],
                    'created_at': result[8]
                }

                conn.close()
                return user_data

            conn.close()
            return None

        except Exception as e:
            st.error(f"Authentication error: {e}")
            return None

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Get user by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id, username, email, full_name, role, organization,
                       phone, region, created_at, last_login
                FROM users
                WHERE id = ?
            ''', (user_id,))

            result = cursor.fetchone()
            conn.close()

            if result:
                return {
                    'id': result[0],
                    'username': result[1],
                    'email': result[2],
                    'full_name': result[3],
                    'role': result[4],
                    'organization': result[5],
                    'phone': result[6],
                    'region': result[7],
                    'created_at': result[8],
                    'last_login': result[9]
                }
            return None

        except Exception as e:
            st.error(f"Error fetching user: {e}")
            return None

    def update_user_profile(self, user_id: int, updates: Dict) -> bool:
        """Update user profile"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Build dynamic update query
            update_fields = []
            values = []

            for key, value in updates.items():
                if key not in ['id', 'username', 'password_hash', 'created_at']:
                    update_fields.append(f"{key} = ?")
                    values.append(value)

            values.append(user_id)

            query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?"
            cursor.execute(query, values)

            conn.commit()
            conn.close()

            return True

        except Exception as e:
            st.error(f"Error updating profile: {e}")
            return False

    def add_contribution(self, user_id: int, contribution_type: str,
                        data: Dict) -> bool:
        """Add user contribution"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            data_json = json.dumps(data)

            cursor.execute('''
                INSERT INTO user_contributions (user_id, contribution_type, data_json)
                VALUES (?, ?, ?)
            ''', (user_id, contribution_type, data_json))

            conn.commit()
            conn.close()

            self.log_activity(user_id, "contribution", f"Added {contribution_type} contribution")

            return True

        except Exception as e:
            st.error(f"Error adding contribution: {e}")
            return False

    def get_user_contributions(self, user_id: int) -> List[Dict]:
        """Get all contributions by user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id, contribution_type, data_json, submitted_at, status
                FROM user_contributions
                WHERE user_id = ?
                ORDER BY submitted_at DESC
            ''', (user_id,))

            results = cursor.fetchall()
            conn.close()

            contributions = []
            for row in results:
                contributions.append({
                    'id': row[0],
                    'type': row[1],
                    'data': json.loads(row[2]),
                    'submitted_at': row[3],
                    'status': row[4]
                })

            return contributions

        except Exception as e:
            st.error(f"Error fetching contributions: {e}")
            return []

    def log_activity(self, user_id: int, activity_type: str, description: str):
        """Log user activity"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO activity_log (user_id, activity_type, description)
                VALUES (?, ?, ?)
            ''', (user_id, activity_type, description))

            conn.commit()
            conn.close()

        except Exception:
            pass  # Silently fail for activity logging

    def get_user_statistics(self) -> Dict:
        """Get overall user statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Total users
            cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = 1')
            total_users = cursor.fetchone()[0]

            # Total contributions
            cursor.execute('SELECT COUNT(*) FROM user_contributions')
            total_contributions = cursor.fetchone()[0]

            # Users by role
            cursor.execute('SELECT role, COUNT(*) FROM users GROUP BY role')
            users_by_role = dict(cursor.fetchall())

            conn.close()

            return {
                'total_users': total_users,
                'total_contributions': total_contributions,
                'users_by_role': users_by_role
            }

        except Exception as e:
            st.error(f"Error fetching statistics: {e}")
            return {}


class AuthenticationUI:
    """Streamlit UI components for authentication"""

    def __init__(self, db: UserDatabase):
        self.db = db

        # Initialize session state
        if 'user' not in st.session_state:
            st.session_state.user = None
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False

    def login_form(self):
        """Display login form"""
        st.subheader("🔐 Login")

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                if username and password:
                    user = self.db.authenticate_user(username, password)

                    if user:
                        st.session_state.user = user
                        st.session_state.authenticated = True
                        st.success(f"Welcome back, {user['full_name'] or user['username']}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Please enter both username and password")

    def register_form(self):
        """Display registration form"""
        st.subheader("📝 Register New Account")

        with st.form("register_form"):
            col1, col2 = st.columns(2)

            with col1:
                username = st.text_input("Username*")
                email = st.text_input("Email*")
                password = st.text_input("Password*", type="password")
                confirm_password = st.text_input("Confirm Password*", type="password")

            with col2:
                full_name = st.text_input("Full Name")
                organization = st.text_input("Organization")
                phone = st.text_input("Phone Number")
                region = st.selectbox("Region", ["", "Kigali", "Northern", "Southern",
                                                  "Eastern", "Western"])

            role = st.selectbox("Role", ["user", "researcher", "policymaker", "admin"])

            submit = st.form_submit_button("Register")

            if submit:
                if not all([username, email, password, confirm_password]):
                    st.error("Please fill in all required fields (*)")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    success = self.db.create_user(
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
                        st.success("Account created successfully! Please login.")
                        return True
                    else:
                        st.error("Username or email already exists")

        return False

    def logout(self):
        """Logout user"""
        if st.session_state.user:
            self.db.log_activity(
                st.session_state.user['id'],
                "logout",
                "User logged out"
            )

        st.session_state.user = None
        st.session_state.authenticated = False
        st.rerun()

    def display_user_profile(self):
        """Display user profile"""
        if not st.session_state.authenticated:
            st.warning("Please login to view your profile")
            return

        user = st.session_state.user
        st.subheader("👤 User Profile")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image("https://via.placeholder.com/150", width=150)

        with col2:
            st.write(f"**Username:** {user['username']}")
            st.write(f"**Email:** {user['email']}")
            st.write(f"**Full Name:** {user['full_name']}")
            st.write(f"**Role:** {user['role'].title()}")
            st.write(f"**Organization:** {user['organization']}")
            st.write(f"**Region:** {user['region']}")
            st.write(f"**Member Since:** {user['created_at']}")

        # Edit profile section
        with st.expander("Edit Profile"):
            with st.form("edit_profile"):
                full_name = st.text_input("Full Name", value=user['full_name'] or "")
                organization = st.text_input("Organization", value=user['organization'] or "")
                phone = st.text_input("Phone", value=user['phone'] or "")
                region = st.selectbox("Region",
                                     ["Kigali", "Northern", "Southern", "Eastern", "Western"],
                                     index=["Kigali", "Northern", "Southern", "Eastern", "Western"].index(user['region']) if user['region'] else 0)

                submit = st.form_submit_button("Update Profile")

                if submit:
                    updates = {
                        'full_name': full_name,
                        'organization': organization,
                        'phone': phone,
                        'region': region
                    }

                    if self.db.update_user_profile(user['id'], updates):
                        st.success("Profile updated successfully!")
                        # Update session state
                        user.update(updates)
                        st.session_state.user = user
                        st.rerun()

    def display_auth_page(self):
        """Display authentication page"""
        st.title("🔐 Authentication")

        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            self.login_form()

        with tab2:
            self.register_form()


def display_user_sidebar():
    """Display user info in sidebar"""
    if st.session_state.get('authenticated', False):
        user = st.session_state.user

        with st.sidebar:
            st.markdown("---")
            st.markdown("### 👤 User Info")
            st.write(f"**{user['full_name'] or user['username']}**")
            st.write(f"Role: {user['role'].title()}")

            if st.button("🚪 Logout", use_container_width=True):
                db = UserDatabase()
                auth_ui = AuthenticationUI(db)
                auth_ui.logout()
    else:
        with st.sidebar:
            st.markdown("---")
            st.info("Please login to access all features")


if __name__ == "__main__":
    # Test the module
    db = UserDatabase()
    auth_ui = AuthenticationUI(db)
    auth_ui.display_auth_page()
