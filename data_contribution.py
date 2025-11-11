"""
Data Contribution Module
Allows users to contribute unemployment data and experiences
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from user_management import UserDatabase

class DataContribution:
    """Handle user data contributions"""

    def __init__(self, db: UserDatabase):
        self.db = db

    def display_contribution_form(self):
        """Display form for users to contribute data"""
        if not st.session_state.get('authenticated', False):
            st.warning("🔒 Please login to contribute data")
            return

        user = st.session_state.user

        st.header("📊 Contribute Your Data")
        st.write("Help us build a comprehensive database by sharing your employment information")

        contribution_type = st.selectbox(
            "What would you like to contribute?",
            ["Personal Employment Status", "Job Opportunity", "Skills Gap Report",
             "Training Program Feedback", "Success Story"]
        )

        if contribution_type == "Personal Employment Status":
            self._employment_status_form(user)
        elif contribution_type == "Job Opportunity":
            self._job_opportunity_form(user)
        elif contribution_type == "Skills Gap Report":
            self._skills_gap_form(user)
        elif contribution_type == "Training Program Feedback":
            self._training_feedback_form(user)
        elif contribution_type == "Success Story":
            self._success_story_form(user)

    def _employment_status_form(self, user: Dict):
        """Form for personal employment status"""
        st.subheader("Personal Employment Status")

        with st.form("employment_status_form"):
            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input("Age", min_value=16, max_value=30, value=20)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                education_level = st.selectbox("Education Level",
                    ["Primary or Less", "Lower Secondary", "Upper Secondary",
                     "TVET", "Bachelor's", "Master's or Higher"])
                region = st.selectbox("Region",
                    ["Kigali", "Northern", "Southern", "Eastern", "Western"])

            with col2:
                employment_status = st.selectbox("Current Employment Status",
                    ["Employed Full-time", "Employed Part-time", "Self-employed",
                     "Unemployed - Actively Seeking", "Unemployed - Not Seeking",
                     "Student", "Other"])

                if employment_status.startswith("Employed") or employment_status == "Self-employed":
                    sector = st.selectbox("Employment Sector",
                        ["Agriculture", "Construction", "Education", "Healthcare",
                         "IT/Technology", "Manufacturing", "Retail", "Services",
                         "Tourism", "Other"])
                    monthly_income = st.number_input("Monthly Income (RWF)",
                                                    min_value=0, value=0, step=10000)
                    formal_informal = st.radio("Employment Type",
                                              ["Formal", "Informal"])
                else:
                    sector = None
                    monthly_income = 0
                    formal_informal = None

                    if "Unemployed" in employment_status:
                        unemployment_duration = st.number_input(
                            "Duration of Unemployment (months)",
                            min_value=0, value=0, step=1)
                    else:
                        unemployment_duration = 0

            # Skills section
            st.write("**Skills**")
            col3, col4 = st.columns(2)

            with col3:
                digital_skills = st.multiselect("Digital Skills",
                    ["Basic Computer", "Microsoft Office", "Programming",
                     "Data Analysis", "Graphic Design", "Social Media",
                     "Digital Marketing", "None"])

            with col4:
                technical_skills = st.multiselect("Technical Skills",
                    ["Carpentry", "Plumbing", "Electrical", "Welding",
                     "Tailoring", "Mechanics", "Agriculture", "Construction",
                     "Cooking", "Hairdressing", "None"])

            # Training and programs
            training_participated = st.checkbox("Have you participated in any training programs?")
            if training_participated:
                training_program = st.text_input("Training Program Name")
                training_effectiveness = st.select_slider(
                    "How effective was the training?",
                    ["Not Effective", "Slightly Effective", "Moderately Effective",
                     "Very Effective", "Extremely Effective"])
            else:
                training_program = None
                training_effectiveness = None

            # Additional information
            job_seeking_challenges = st.multiselect("Main Challenges in Finding Employment",
                ["Lack of Experience", "Skills Mismatch", "Limited Opportunities",
                 "Geographic Location", "No Transportation", "Family Responsibilities",
                 "Lack of Information", "Low Wages", "Other"])

            additional_comments = st.text_area("Additional Comments (Optional)")

            consent = st.checkbox("I consent to share this data for research and policy purposes")

            submit = st.form_submit_button("Submit Contribution")

            if submit:
                if not consent:
                    st.error("Please provide consent to submit your data")
                    return

                # Prepare contribution data
                contribution_data = {
                    'age': age,
                    'gender': gender,
                    'education_level': education_level,
                    'region': region,
                    'employment_status': employment_status,
                    'sector': sector,
                    'monthly_income': monthly_income,
                    'formal_informal': formal_informal,
                    'unemployment_duration': unemployment_duration if "Unemployed" in employment_status else None,
                    'digital_skills': digital_skills,
                    'technical_skills': technical_skills,
                    'training_participated': training_participated,
                    'training_program': training_program,
                    'training_effectiveness': training_effectiveness,
                    'job_seeking_challenges': job_seeking_challenges,
                    'additional_comments': additional_comments,
                    'submitted_by': user['username'],
                    'submission_date': datetime.now().isoformat()
                }

                # Save to database
                if self.db.add_contribution(user['id'], "Employment Status", contribution_data):
                    st.success("✅ Thank you for your contribution!")
                    st.balloons()
                else:
                    st.error("Failed to submit contribution. Please try again.")

    def _job_opportunity_form(self, user: Dict):
        """Form for job opportunity posting"""
        st.subheader("Post a Job Opportunity")

        with st.form("job_opportunity_form"):
            col1, col2 = st.columns(2)

            with col1:
                job_title = st.text_input("Job Title*")
                company_name = st.text_input("Company/Organization Name*")
                job_location = st.selectbox("Job Location*",
                    ["Kigali", "Northern Province", "Southern Province",
                     "Eastern Province", "Western Province", "Remote"])
                sector = st.selectbox("Sector*",
                    ["Agriculture", "Construction", "Education", "Healthcare",
                     "IT/Technology", "Manufacturing", "Retail", "Services",
                     "Tourism", "NGO", "Government", "Other"])

            with col2:
                employment_type = st.selectbox("Employment Type*",
                    ["Full-time", "Part-time", "Contract", "Internship",
                     "Temporary", "Volunteer"])
                experience_required = st.selectbox("Experience Required",
                    ["No experience", "0-1 years", "1-3 years", "3-5 years", "5+ years"])
                education_required = st.selectbox("Minimum Education",
                    ["None", "Primary", "Secondary", "TVET", "Bachelor's", "Master's"])
                salary_range = st.text_input("Salary Range (Optional)")

            job_description = st.text_area("Job Description*")
            required_skills = st.text_area("Required Skills (comma-separated)")
            application_instructions = st.text_area("How to Apply*")
            application_deadline = st.date_input("Application Deadline")

            contact_email = st.text_input("Contact Email")
            contact_phone = st.text_input("Contact Phone (Optional)")

            submit = st.form_submit_button("Post Job Opportunity")

            if submit:
                if not all([job_title, company_name, job_location, sector,
                           employment_type, job_description, application_instructions]):
                    st.error("Please fill in all required fields (*)")
                    return

                contribution_data = {
                    'job_title': job_title,
                    'company_name': company_name,
                    'job_location': job_location,
                    'sector': sector,
                    'employment_type': employment_type,
                    'experience_required': experience_required,
                    'education_required': education_required,
                    'salary_range': salary_range,
                    'job_description': job_description,
                    'required_skills': required_skills,
                    'application_instructions': application_instructions,
                    'application_deadline': str(application_deadline),
                    'contact_email': contact_email,
                    'contact_phone': contact_phone,
                    'posted_by': user['username'],
                    'posted_date': datetime.now().isoformat()
                }

                if self.db.add_contribution(user['id'], "Job Opportunity", contribution_data):
                    st.success("✅ Job opportunity posted successfully!")
                else:
                    st.error("Failed to post job opportunity. Please try again.")

    def _skills_gap_form(self, user: Dict):
        """Form for skills gap reporting"""
        st.subheader("Report Skills Gap")
        st.write("Help identify skill mismatches in the job market")

        with st.form("skills_gap_form"):
            sector = st.selectbox("Sector/Industry*",
                ["Agriculture", "Construction", "Education", "Healthcare",
                 "IT/Technology", "Manufacturing", "Retail", "Services",
                 "Tourism", "Other"])

            region = st.selectbox("Region*",
                ["Kigali", "Northern", "Southern", "Eastern", "Western", "All Regions"])

            missing_skills = st.text_area(
                "What skills are lacking in youth?*",
                help="List skills that young people need but don't have")

            in_demand_skills = st.text_area(
                "What skills are most in demand?*",
                help="List skills that employers are looking for")

            training_suggestions = st.text_area(
                "Training Program Suggestions",
                help="What training programs would help bridge this gap?")

            urgency = st.select_slider(
                "How urgent is this skills gap?",
                ["Low", "Medium", "High", "Critical"])

            additional_notes = st.text_area("Additional Notes")

            submit = st.form_submit_button("Submit Skills Gap Report")

            if submit:
                if not all([sector, region, missing_skills, in_demand_skills]):
                    st.error("Please fill in all required fields (*)")
                    return

                contribution_data = {
                    'sector': sector,
                    'region': region,
                    'missing_skills': missing_skills,
                    'in_demand_skills': in_demand_skills,
                    'training_suggestions': training_suggestions,
                    'urgency': urgency,
                    'additional_notes': additional_notes,
                    'reported_by': user['username'],
                    'report_date': datetime.now().isoformat()
                }

                if self.db.add_contribution(user['id'], "Skills Gap Report", contribution_data):
                    st.success("✅ Skills gap report submitted successfully!")
                else:
                    st.error("Failed to submit report. Please try again.")

    def _training_feedback_form(self, user: Dict):
        """Form for training program feedback"""
        st.subheader("Training Program Feedback")

        with st.form("training_feedback_form"):
            program_name = st.text_input("Training Program Name*")
            organization = st.text_input("Training Organization*")
            program_type = st.selectbox("Program Type",
                ["Technical Skills", "Digital Skills", "Business Development",
                 "Entrepreneurship", "Soft Skills", "Language", "Other"])

            duration = st.text_input("Duration (e.g., 3 months)")
            completion_date = st.date_input("When did you complete it?")

            col1, col2 = st.columns(2)

            with col1:
                content_rating = st.slider("Content Quality", 1, 5, 3)
                instructor_rating = st.slider("Instructor Quality", 1, 5, 3)
                facilities_rating = st.slider("Facilities Rating", 1, 5, 3)

            with col2:
                relevance_rating = st.slider("Job Market Relevance", 1, 5, 3)
                satisfaction_rating = st.slider("Overall Satisfaction", 1, 5, 3)

            employment_after = st.radio(
                "Did you find employment after this training?",
                ["Yes, in related field", "Yes, in unrelated field", "No, still searching",
                 "No, not currently seeking"])

            skills_gained = st.text_area("What skills did you gain?*")
            recommendations = st.text_area("Recommendations for improvement")

            would_recommend = st.checkbox("Would you recommend this program to others?")

            submit = st.form_submit_button("Submit Feedback")

            if submit:
                if not all([program_name, organization, skills_gained]):
                    st.error("Please fill in all required fields (*)")
                    return

                contribution_data = {
                    'program_name': program_name,
                    'organization': organization,
                    'program_type': program_type,
                    'duration': duration,
                    'completion_date': str(completion_date),
                    'content_rating': content_rating,
                    'instructor_rating': instructor_rating,
                    'facilities_rating': facilities_rating,
                    'relevance_rating': relevance_rating,
                    'satisfaction_rating': satisfaction_rating,
                    'employment_after': employment_after,
                    'skills_gained': skills_gained,
                    'recommendations': recommendations,
                    'would_recommend': would_recommend,
                    'submitted_by': user['username'],
                    'submission_date': datetime.now().isoformat()
                }

                if self.db.add_contribution(user['id'], "Training Feedback", contribution_data):
                    st.success("✅ Thank you for your feedback!")
                else:
                    st.error("Failed to submit feedback. Please try again.")

    def _success_story_form(self, user: Dict):
        """Form for success stories"""
        st.subheader("Share Your Success Story")
        st.write("Inspire others by sharing your journey to employment!")

        with st.form("success_story_form"):
            story_title = st.text_input("Story Title*")

            background = st.text_area(
                "Your Background*",
                help="Tell us about your situation before finding employment")

            challenges = st.text_area(
                "Challenges You Faced*",
                help="What obstacles did you overcome?")

            turning_point = st.text_area(
                "What Was Your Turning Point?*",
                help="What helped you succeed?")

            current_situation = st.text_area(
                "Where Are You Now?*",
                help="Describe your current employment and achievements")

            advice = st.text_area(
                "Advice for Others*",
                help="What advice would you give to other youth?")

            allow_contact = st.checkbox("Allow others to contact me for mentorship")

            submit = st.form_submit_button("Share Story")

            if submit:
                if not all([story_title, background, challenges, turning_point,
                           current_situation, advice]):
                    st.error("Please fill in all required fields (*)")
                    return

                contribution_data = {
                    'story_title': story_title,
                    'background': background,
                    'challenges': challenges,
                    'turning_point': turning_point,
                    'current_situation': current_situation,
                    'advice': advice,
                    'allow_contact': allow_contact,
                    'author': user['username'],
                    'submission_date': datetime.now().isoformat()
                }

                if self.db.add_contribution(user['id'], "Success Story", contribution_data):
                    st.success("✅ Thank you for sharing your inspiring story!")
                    st.balloons()
                else:
                    st.error("Failed to submit story. Please try again.")

    def display_user_contributions(self):
        """Display user's past contributions"""
        if not st.session_state.get('authenticated', False):
            st.warning("🔒 Please login to view your contributions")
            return

        user = st.session_state.user
        contributions = self.db.get_user_contributions(user['id'])

        st.subheader("📋 Your Contributions")

        if not contributions:
            st.info("You haven't made any contributions yet. Start contributing data above!")
            return

        for i, contrib in enumerate(contributions):
            with st.expander(f"{contrib['type']} - {contrib['submitted_at'][:10]}"):
                st.write(f"**Status:** {contrib['status'].title()}")
                st.write(f"**Submitted:** {contrib['submitted_at']}")

                st.json(contrib['data'])


if __name__ == "__main__":
    # Test the module
    db = UserDatabase()
    contribution = DataContribution(db)
    contribution.display_contribution_form()
