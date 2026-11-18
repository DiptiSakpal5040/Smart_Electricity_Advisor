
import os

# ======================================================
# MEMORY OPTIMIZATION — SET BEFORE NUMERICAL IMPORTS
# ======================================================

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import json
import hashlib
import re
from pathlib import Path
from datetime import datetime, date
from collections import Counter

import streamlit as st
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits


# ======================================================
# PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="Smart Electricity Advisor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ======================================================
# FILE PATHS
# ======================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR.parent / "models"

USER_FILE = BASE_DIR / "users.json"
ELECTRICITY_FILE = BASE_DIR / "electricity_data.json"
HISTORY_FILE = BASE_DIR / "prediction_history.json"


# ======================================================
# GENERAL STORAGE FUNCTIONS
# ======================================================

def load_json(path, default=None):
    if default is None:
        default = {}

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError):
            return default

    return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, default=str)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def load_users():
    return load_json(USER_FILE, {})


def save_users(users):
    save_json(USER_FILE, users)


def load_electricity_data():
    return load_json(ELECTRICITY_FILE, {})


def save_electricity_data(data):
    save_json(ELECTRICITY_FILE, data)


def load_prediction_history():
    return load_json(HISTORY_FILE, {})


def save_prediction_history(data):
    save_json(HISTORY_FILE, data)


# ======================================================
# MODEL LOADING
# ======================================================

def load_model(filename):
    model_path = MODEL_DIR / filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find model file: {model_path}"
        )

    return joblib.load(model_path)


# ======================================================
# SESSION STATE
# ======================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "full_name" not in st.session_state:
    st.session_state.full_name = ""

if "email" not in st.session_state:
    st.session_state.email = ""


# ======================================================
# THEME — PREMIUM TEAL, NAVY AND WHITE
# ======================================================

st.markdown("""
<style>

/* Main application background */

.stApp {
    background: #F3F7F8;
    color: #1E293B;
}

/* Header */

header[data-testid="stHeader"] {
    background: transparent;
}

footer {
    visibility: hidden;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: #102A43;
    border-right: 1px solid #1E405B;
}

[data-testid="stSidebar"] * {
    color: #EAF4F4 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #C3D5DF !important;
}

[data-testid="stSidebar"] hr {
    border-color: #31516A;
}

/* Main headings */

.main-title {
    color: #123B4A;
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 4px;
}

.subtitle {
    color: #647B87;
    font-size: 15px;
    margin-bottom: 20px;
}

.section-title {
    color: #164E63;
    font-size: 21px;
    font-weight: 750;
    margin-top: 20px;
    margin-bottom: 12px;
}

/* Cards */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #DCE8E9;
    border-radius: 15px;
    box-shadow: 0 4px 14px rgba(16, 42, 67, 0.04);
}

/* Login card */

div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stForm"]) {
    background: #FFFFFF !important;
    border: 1px solid #DCE8E9 !important;
    border-radius: 22px !important;
    padding: 22px !important;
    box-shadow: 0 12px 35px rgba(16, 42, 67, 0.10) !important;
}

/* Input labels */

.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stDateInput label,
.stTimeInput label {
    color: #164E63;
    font-weight: 600;
}

/* Input fields */

.stTextInput input,
.stNumberInput input {
    background: #F8FBFC;
    color: #183B4A;
    border: 1px solid #C9DADD;
    border-radius: 9px;
}

/* Select boxes */

.stSelectbox div[data-baseweb="select"] > div {
    background: #F8FBFC;
    border-color: #C9DADD;
    border-radius: 9px;
}

/* Buttons */

.stButton button,
.stFormSubmitButton button {
    background: #087E8B;
    color: white;
    border: none;
    border-radius: 9px;
    min-height: 42px;
    font-weight: 650;
    transition: 0.2s;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #05616C;
    color: white;
    border: none;
}

/* Tabs */

.stTabs [data-baseweb="tab"] {
    color: #52717C;
    font-weight: 650;
}

.stTabs [aria-selected="true"] {
    color: #087E8B !important;
}

/* Metrics */

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DCE8E9;
    padding: 16px;
    border-radius: 13px;
}

/* Footer */

.footer {
    text-align: center;
    color: #82949C;
    font-size: 12px;
    margin-top: 25px;
}

/* Small labels */

.small-note {
    color: #718792;
    font-size: 12px;
}

</style>
""", unsafe_allow_html=True)


# ======================================================
# LOGIN AND REGISTRATION
# ======================================================

if not st.session_state.logged_in:

    st.markdown(
        '<div class="main-title" style="text-align:center;'
        'font-size:36px;margin-top:35px;">'
        '⚡ Smart Electricity Advisor</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle" style="text-align:center;">'
        'Monitor • Predict • Save Energy'
        '</div>',
        unsafe_allow_html=True
    )

    left, center, right = st.columns([1, 1.15, 1])

    with center:

        with st.container(border=True):

            st.markdown(
                "<h2 style='text-align:center;color:#164E63;'>"
                "Welcome!</h2>",
                unsafe_allow_html=True
            )

            st.markdown(
                "<p style='text-align:center;color:#718792;'>"
                "Sign in to manage your household electricity"
                "</p>",
                unsafe_allow_html=True
            )

            login_tab, register_tab = st.tabs(
                ["Login", "Create Account"]
            )

            # LOGIN

            with login_tab:

                with st.form("login_form"):

                    login_email = st.text_input(
                        "Email Address",
                        placeholder="Enter your email"
                    )

                    login_password = st.text_input(
                        "Password",
                        type="password",
                        placeholder="Enter your password"
                    )

                    login_button = st.form_submit_button(
                        "Login",
                        use_container_width=True
                    )

                if login_button:

                    users = load_users()
                    email = login_email.strip().lower()

                    if (
                        email in users
                        and users[email]["password"]
                        == hash_password(login_password)
                    ):

                        st.session_state.logged_in = True
                        st.session_state.full_name = (
                            users[email]["full_name"]
                        )
                        st.session_state.email = email

                        st.rerun()

                    else:
                        st.error(
                            "Invalid email address or password."
                        )

            # REGISTRATION

            with register_tab:

                with st.form("register_form"):

                    new_name = st.text_input(
                        "Full Name",
                        placeholder="Enter your full name"
                    )

                    new_email = st.text_input(
                        "Email Address",
                        placeholder="Enter your email"
                    )

                    new_password = st.text_input(
                        "Password",
                        type="password",
                        placeholder="Create a password"
                    )

                    register_button = st.form_submit_button(
                        "Create Account",
                        use_container_width=True
                    )

                if register_button:

                    users = load_users()

                    name = new_name.strip()
                    email = new_email.strip().lower()

                    if not name or not email or not new_password:
                        st.error("Please fill in all fields.")

                    elif not re.fullmatch(
                        r"[^@\s]+@[^@\s]+\.[^@\s]+",
                        email
                    ):
                        st.error("Enter a valid email address.")

                    elif len(new_password) < 6:
                        st.error(
                            "Password must contain at least 6 characters."
                        )

                    elif email in users:
                        st.error(
                            "An account with this email already exists."
                        )

                    else:

                        users[email] = {
                            "full_name": name,
                            "email": email,
                            "password": hash_password(new_password)
                        }

                        save_users(users)

                        st.success(
                            "Account created successfully! "
                            "You can now log in."
                        )

    st.markdown(
        '<div class="footer">'
        'Smart Electricity Consumption & Saving Advisor'
        '</div>',
        unsafe_allow_html=True
    )


# ======================================================
# MAIN APPLICATION AFTER LOGIN
# ======================================================

else:

    user_email = st.session_state.email

    electricity_data = load_electricity_data()
    user_data = electricity_data.get(user_email, {})

    all_history = load_prediction_history()
    user_history = all_history.get(user_email, [])

    # --------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------

    with st.sidebar:

        st.markdown("## ⚡ Smart Advisor")

        st.caption("Your Personal Energy Dashboard")

        st.divider()

        st.markdown("### USER PROFILE")

        st.write(f"Welcome, **{st.session_state.full_name}**")
        st.caption(user_email)

        st.divider()

        st.markdown("### MAIN MENU")

        selected_page = st.radio(
            "Navigate",
            [
                "Dashboard",
                "My Electricity Details",
                "Consumption Prediction",
                "Model Comparison",
                "Electricity Saving Advisor",
                "Electricity Bill Estimator",
                "Consumption Trends",
                "Reports & History",
                "My Profile",
                "About Project"
            ],
            label_visibility="collapsed"
        )

        st.divider()

        if st.button("🚪 Logout", use_container_width=True):

            st.session_state.logged_in = False
            st.session_state.full_name = ""
            st.session_state.email = ""

            st.rerun()

        st.caption("Smart Electricity Advisor | Version 2.0")


    # ==================================================
    # DASHBOARD
    # ==================================================

    if selected_page == "Dashboard":

        st.markdown(
            '<div class="main-title">My Electricity Dashboard ⚡</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="subtitle">Welcome back, '
            f'{st.session_state.full_name}! '
            'Here is your household electricity overview.</div>',
            unsafe_allow_html=True
        )

        if not user_data:

            st.info(
                "Your electricity profile is not set up yet."
            )

            st.warning(
                "Go to My Electricity Details and enter your "
                "household information to personalize this dashboard."
            )

        else:

            monthly_units = float(
                user_data.get("monthly_units", 0)
            )

            monthly_bill = float(
                user_data.get("monthly_bill", 0)
            )

            family_members = int(
                user_data.get("family_members", 1)
            )

            tariff = float(
                user_data.get("tariff", 0)
            )

            household_type = user_data.get(
                "household_type", "Residential"
            )

            daily_units = monthly_units / 30
            per_person_units = monthly_units / max(
                family_members, 1
            )

            estimated_bill = monthly_units * tariff

            st.markdown(
                '<div class="section-title">'
                'Your Electricity Statistics'
                '</div>',
                unsafe_allow_html=True
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Monthly Consumption",
                f"{monthly_units:,.1f} Units"
            )

            col2.metric(
                "Monthly Bill",
                f"₹{monthly_bill:,.2f}"
            )

            col3.metric(
                "Daily Average",
                f"{daily_units:,.2f} Units"
            )

            col4.metric(
                "Family Members",
                family_members
            )

            st.markdown(
                '<div class="section-title">'
                'Household Overview'
                '</div>',
                unsafe_allow_html=True
            )

            with st.container(border=True):

                info1, info2, info3 = st.columns(3)

                info1.markdown("#### 🏠 Household Type")
                info1.write(household_type)

                info2.markdown("#### ⚡ Tariff")
                info2.write(f"₹{tariff:.2f} per unit")

                info3.markdown("#### 👨‍👩‍👧 Units per Person")
                info3.write(
                    f"{per_person_units:,.2f} units/month"
                )

                st.caption(
                    "The daily average uses a 30-day calculation. "
                    "The approximate bill is monthly units multiplied "
                    "by your entered tariff."
                )

            st.markdown(
                '<div class="section-title">'
                'Your Electricity Summary'
                '</div>',
                unsafe_allow_html=True
            )

            with st.container(border=True):

                st.write(
                    f"Your household consumes approximately "
                    f"**{monthly_units:,.1f} units per month**."
                )

                st.write(
                    f"Your estimated daily consumption is "
                    f"**{daily_units:,.2f} units**."
                )

                st.write(
                    f"Your household's average consumption is "
                    f"**{per_person_units:,.2f} units per person** "
                    f"per month."
                )

                st.write(
                    f"At your entered tariff, the calculated bill "
                    f"is approximately **₹{estimated_bill:,.2f}**."
                )

        # RECENT PREDICTION

        st.markdown(
            '<div class="section-title">Latest ML Prediction</div>',
            unsafe_allow_html=True
        )

        if user_history:

            latest = user_history[-1]

            with st.container(border=True):

                st.write(
                    f"**Prediction:** {latest.get('overall', 'N/A')}"
                )

                st.write(
                    f"**Date:** {latest.get('date', 'N/A')}"
                )

                st.write(
                    f"**Random Forest:** "
                    f"{latest.get('random_forest', 'N/A')} "
                    f" | **SVM:** {latest.get('svm', 'N/A')} "
                    f" | **XGBoost:** {latest.get('xgboost', 'N/A')}"
                )

        else:

            st.info(
                "You have not made a prediction yet. "
                "Visit Consumption Prediction to try the three ML models."
            )

        # CONSUMPTION CATEGORIES

        st.markdown(
            '<div class="section-title">'
            'Consumption Categories'
            '</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            with st.container(border=True):
                st.markdown("### 🟢 Low")
                st.write("Lower relative consumption.")

        with c2:
            with st.container(border=True):
                st.markdown("### 🟡 Medium")
                st.write("Moderate relative consumption.")

        with c3:
            with st.container(border=True):
                st.markdown("### 🔴 High")
                st.write("Higher relative consumption.")

        st.caption(
            "These are the classification labels used by the trained "
            "machine learning models."
        )

        # QUICK ACCESS

        st.markdown(
            '<div class="section-title">Explore Your Energy Tools</div>',
            unsafe_allow_html=True
        )

        quick_features = [
            (
                "📊 Consumption Prediction",
                "Classify electricity consumption using three ML models."
            ),
            (
                "🤖 Model Comparison",
                "Compare Random Forest, SVM and XGBoost."
            ),
            (
                "💡 Electricity Saving Advisor",
                "Explore practical ways to reduce electricity usage."
            ),
            (
                "🧾 Electricity Bill Estimator",
                "Estimate your electricity bill using your tariff."
            )
        ]

        for i in range(0, len(quick_features), 2):

            q1, q2 = st.columns(2)

            for column, item in zip(
                [q1, q2],
                quick_features[i:i + 2]
            ):

                with column:

                    with st.container(border=True):

                        st.markdown(f"### {item[0]}")
                        st.write(item[1])


    # ==================================================
    # MY ELECTRICITY DETAILS
    # ==================================================

    elif selected_page == "My Electricity Details":

        st.title("⚡ My Electricity Details")

        st.write(
            "Enter your household information to personalize "
            "your electricity dashboard."
        )

        existing_data = load_electricity_data().get(
            user_email, {}
        )

        with st.container(border=True):

            st.subheader("Household Information")

            with st.form("electricity_details_form"):

                col1, col2 = st.columns(2)

                with col1:

                    monthly_units = st.number_input(
                        "Monthly Electricity Consumption (Units)",
                        min_value=0.0,
                        value=float(
                            existing_data.get("monthly_units", 0.0)
                        ),
                        step=10.0
                    )

                    monthly_bill = st.number_input(
                        "Monthly Electricity Bill (₹)",
                        min_value=0.0,
                        value=float(
                            existing_data.get("monthly_bill", 0.0)
                        ),
                        step=100.0
                    )

                    family_members = st.number_input(
                        "Number of Family Members",
                        min_value=1,
                        max_value=30,
                        value=int(
                            existing_data.get("family_members", 1)
                        ),
                        step=1
                    )

                with col2:

                    tariff = st.number_input(
                        "Electricity Tariff (₹ per Unit)",
                        min_value=0.0,
                        value=float(
                            existing_data.get("tariff", 0.0)
                        ),
                        step=0.5
                    )

                    household_options = [
                        "Residential",
                        "Apartment",
                        "Other"
                    ]

                    saved_type = existing_data.get(
                        "household_type", "Residential"
                    )

                    type_index = (
                        household_options.index(saved_type)
                        if saved_type in household_options
                        else 0
                    )

                    household_type = st.selectbox(
                        "Type of Household",
                        household_options,
                        index=type_index
                    )

                submitted = st.form_submit_button(
                    "Save Electricity Details",
                    use_container_width=True
                )

                if submitted:

                    data = load_electricity_data()

                    old_data = data.get(user_email, {})

                    data[user_email] = {
                        "monthly_units": monthly_units,
                        "monthly_bill": monthly_bill,
                        "family_members": family_members,
                        "tariff": tariff,
                        "household_type": household_type,
                        "monthly_records": old_data.get(
                            "monthly_records", []
                        )
                    }

                    save_electricity_data(data)

                    st.success(
                        "Your electricity details have been saved!"
                    )

                    st.rerun()

        if existing_data:
            st.success("Your household details are already saved.")
        else:
            st.info(
                "Complete the form above to set up your electricity profile."
            )


    # ==================================================
    # CONSUMPTION PREDICTION
    # ==================================================

    elif selected_page == "Consumption Prediction":

        st.title("📊 Consumption Prediction")

        st.write(
            "Enter electricity readings and compare the predictions "
            "of Random Forest, SVM and XGBoost."
        )

        st.info(
            "The models classify consumption into Low, Medium or High. "
            "Enter values in the same units and format used during training."
        )

        st.markdown("### ⚡ Enter Electricity Readings")

        with st.form("prediction_form"):

            col1, col2 = st.columns(2)

            with col1:

                reactive_power = st.number_input(
                    "Global Reactive Power (kW)",
                    min_value=0.0,
                    value=0.12,
                    step=0.01
                )

                voltage = st.number_input(
                    "Voltage (V)",
                    min_value=0.0,
                    value=240.0,
                    step=1.0
                )

                sub_metering_1 = st.number_input(
                    "Kitchen Sub-metering (Wh)",
                    min_value=0.0,
                    value=1.12,
                    step=0.1
                )

                sub_metering_2 = st.number_input(
                    "Laundry Sub-metering (Wh)",
                    min_value=0.0,
                    value=1.30,
                    step=0.1
                )

                sub_metering_3 = st.number_input(
                    "Heating / AC Sub-metering (Wh)",
                    min_value=0.0,
                    value=6.46,
                    step=0.1
                )

            with col2:

                reading_date = st.date_input(
                    "Reading Date",
                    value=date.today()
                )

                reading_time = st.time_input(
                    "Reading Time",
                    value=datetime.now().time().replace(
                        second=0,
                        microsecond=0
                    )
                )

                st.caption(
                    "Hour, day, month and weekday are calculated automatically."
                )

            predict_button = st.form_submit_button(
                "⚡ Predict Consumption",
                use_container_width=True
            )

        if predict_button:

            input_data = pd.DataFrame([{

                "Global_reactive_power": reactive_power,
                "Voltage": voltage,
                "Sub_metering_1": sub_metering_1,
                "Sub_metering_2": sub_metering_2,
                "Sub_metering_3": sub_metering_3,
                "Hour": reading_time.hour,
                "Day": reading_date.day,
                "Month": reading_date.month,
                "DayOfWeek": reading_date.weekday()

            }])

            predictions = {}

            try:

                with st.spinner(
                    "Running all three machine learning models..."
                ):

                    with threadpool_limits(limits=1):

                        # RANDOM FOREST

                        # RANDOM FOREST

                        rf_data = load_model("random_forest.pkl")

# Extract the actual trained model
                        rf_model = rf_data["model"]

# Verify feature order
                        rf_features = rf_data["feature_names"]

                        rf_input = input_data[rf_features]

# Make prediction
                        predictions["Random Forest"] = str(
                        rf_model.predict(rf_input)[0]
                        )

                        del rf_model
                        del rf_data 
                        del rf_input
                        gc.collect()

                        # SVM

                        
                        # SVM

                        svm_data = load_model("svm_model.pkl")

                        # Extract the trained model and scaler
                        svm_model = svm_data["model"]
                        svm_scaler = svm_data["scaler"]

                        # Use the exact feature order from training
                        svm_features = svm_data["feature_names"]
                        svm_input_data = input_data[svm_features]

                        # Scale the input readings
                        svm_input = svm_scaler.transform(svm_input_data)

                        # Make prediction
                        predictions["SVM"] = str(
                        svm_model.predict(svm_input)[0]
                        )
 
                        del svm_model
                        del svm_scaler
                        del svm_data
                        del svm_input
                        del svm_input_data
                        gc.collect()


                        # XGBOOST

                        
                    # XGBOOST

                        xgb_data = load_model("xgboost_model.pkl")

                    # Extract trained model and label encoder
                        xgb_model = xgb_data["model"]
                        xgb_encoder = xgb_data["label_encoder"]

                    # Use exact training feature order
                        xgb_features = xgb_data["feature_names"]
                        xgb_input = input_data[xgb_features]

                     # Predict encoded class
                        xgb_encoded = int(
                        xgb_model.predict(xgb_input)[0]
                        )

                      # Convert encoded class into original category
                        predictions["XGBoost"] = str(
    xgb_encoder.inverse_transform(
        [xgb_encoded]
    )[0]
)

                        del xgb_model
                        del xgb_encoder
                        del xgb_data
                        del xgb_input
                        gc.collect()


                # INDIVIDUAL RESULTS

                st.markdown("---")

                st.markdown("### 📈 Individual Model Results")

                r1, r2, r3 = st.columns(3)

                r1.metric(
                    "🌳 Random Forest",
                    predictions["Random Forest"]
                )

                r2.metric(
                    "📊 SVM",
                    predictions["SVM"]
                )

                r3.metric(
                    "🚀 XGBoost",
                    predictions["XGBoost"]
                )

                # MAJORITY VOTE

                counts = Counter(predictions.values())

                final_prediction, votes = counts.most_common(1)[0]

                st.markdown("### 🎯 Overall Prediction")

                if votes == 1:

                    st.warning(
                        "All three models produced different results. "
                        "There is no majority prediction."
                    )

                    final_prediction = "No Majority"

                elif final_prediction == "High":

                    st.error(
                        "High Consumption — consider reducing "
                        "unnecessary electricity usage."
                    )

                elif final_prediction == "Medium":

                    st.warning(
                        "Medium Consumption — monitor your usage "
                        "and look for opportunities to save energy."
                    )

                else:

                    st.success(
                        "Low Consumption — the majority of models "
                        "classified this reading as Low."
                    )

                st.write(
                    f"**Overall result:** {final_prediction}"
                )

                st.write(
                    f"**Votes:** {votes} out of 3 models"
                )

                st.caption(
                    "The overall result uses majority voting. "
                    "This is a classification, not an exact kWh forecast."
                )

                # SAVE HISTORY

                history_data = load_prediction_history()

                current_user_history = history_data.get(
                    user_email, []
                )

                current_user_history.append({

                    "date": reading_date.strftime("%Y-%m-%d"),
                    "time": reading_time.strftime("%H:%M"),
                    "random_forest": predictions["Random Forest"],
                    "svm": predictions["SVM"],
                    "xgboost": predictions["XGBoost"],
                    "overall": final_prediction,
                    "votes": votes,
                    "reactive_power": reactive_power,
                    "voltage": voltage,
                    "sub_metering_1": sub_metering_1,
                    "sub_metering_2": sub_metering_2,
                    "sub_metering_3": sub_metering_3

                })

                history_data[user_email] = current_user_history

                save_prediction_history(history_data)

                st.success(
                    "Your prediction has been saved to Consumption History."
                )

            except FileNotFoundError as e:

                gc.collect()

                st.error(str(e))

            except Exception as e:

                gc.collect()

                st.error(f"Prediction error: {e}")
                st.exception(e)


    # ==================================================
    # MODEL COMPARISON
    # ==================================================

    elif selected_page == "Model Comparison":

        st.title("🤖 Machine Learning Model Comparison")

        st.write(
            "Compare the performance of the three trained algorithms."
        )

        model_data = pd.DataFrame({

            "Algorithm": [
                "Random Forest",
                "SVM",
                "XGBoost"
            ],

            "Accuracy (%)": [
                77.726,
                69.202,
                77.600
            ],

            "Precision": [
                0.783,
                0.699,
                0.782
            ],

            "Recall": [
                0.793,
                0.723,
                0.792
            ],

            "F1-Score": [
                0.782,
                0.687,
                0.780
            ]

        })

        st.markdown("### Model Evaluation Results")

        st.dataframe(
            model_data,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Accuracy Comparison")

        accuracy_chart = model_data.set_index("Algorithm")

        st.bar_chart(
            accuracy_chart[["Accuracy (%)"]],
            color="#087E8B"
        )

        st.markdown("### Precision, Recall and F1-Score")

        metric_chart = model_data.set_index("Algorithm")

        st.bar_chart(
            metric_chart[
                ["Precision", "Recall", "F1-Score"]
            ],
            color=["#087E8B", "#2A9D8F", "#E9A23B"]
        )

        st.markdown("### Model Performance Summary")

        st.write(
            "- **Random Forest:** Highest accuracy in the supplied evaluation."
        )

        st.write(
            "- **SVM:** Lower accuracy than the other two models "
            "in the supplied evaluation."
        )

        st.write(
            "- **XGBoost:** Performance is close to Random Forest."
        )

        st.caption(
            "These scores are the evaluation values supplied for this project. "
            "They should match the final test-set evaluation used in your report."
        )


    
    # ==================================================
    # ELECTRICITY SAVING ADVISOR
    # ==================================================

    elif selected_page == "Electricity Saving Advisor":

        st.title("💡 Electricity Saving Advisor")

        st.write(
            "Explore practical ways to reduce unnecessary "
            "electricity consumption at home."
        )

        # ---------------- LATEST PREDICTION ----------------

        latest_prediction = (
            user_history[-1].get("overall")
            if user_history
            else None
        )

        if latest_prediction == "High":

            st.error(
                "Your latest majority prediction was High. "
                "Pay attention to appliance usage and standby power."
            )

        elif latest_prediction == "Medium":

            st.warning(
                "Your latest majority prediction was Medium. "
                "Small changes in daily habits can help reduce usage."
            )

        elif latest_prediction == "Low":

            st.success(
                "Your latest majority prediction was Low. "
                "Continue following energy-efficient habits."
            )

        else:

            st.info(
                "Make a consumption prediction to see your latest "
                "consumption category here."
            )

        # ---------------- ENERGY SAVING TIPS ----------------

        tips = [

            (
                "💡 Use LED Lighting",
                "Replace traditional bulbs with energy-efficient LED lights. "
                "Switch off lights when leaving a room."
            ),

            (
                "🔌 Switch Off Unused Devices",
                "Turn off televisions, computers, fans and other appliances "
                "when they are not required."
            ),

            (
                "❄️ Use Air Conditioning Efficiently",
                "Choose a comfortable temperature, clean filters regularly "
                "and keep doors and windows closed while cooling."
            ),

            (
                "🧺 Use Washing Machines Efficiently",
                "Prefer full loads instead of running frequent small loads. "
                "Use energy-efficient settings where available."
            ),

            (
                "⚡ Avoid Standby Power",
                "Unplug chargers and switch off devices that continue "
                "using electricity in standby mode."
            ),

            (
                "🧊 Improve Refrigerator Efficiency",
                "Avoid opening the refrigerator frequently and ensure "
                "the door seal is working properly."
            ),

            (
                "☀️ Make Use of Natural Light",
                "Open curtains during the day and reduce unnecessary "
                "use of artificial lighting."
            ),

            (
                "📊 Monitor Your Electricity Usage",
                "Record monthly meter readings and compare them to "
                "identify changes in household consumption."
            )

        ]

        for title, description in tips:

            with st.container(border=True):

                st.markdown(f"### {title}")
                st.write(description)

        # ---------------- ENERGY SAVING CHECKLIST ----------------

        st.markdown("### Your Energy-Saving Checklist")

        checklist_items = [
            "Switch off unused lights",
            "Unplug unused chargers",
            "Use LED bulbs",
            "Avoid unnecessary AC usage",
            "Run washing machines with full loads",
            "Check monthly electricity readings"
        ]

        for item in checklist_items:

            st.checkbox(item, key=f"tip_{item}")

        # ---------------- POTENTIAL SAVINGS CALCULATOR ----------------

        st.markdown("---")

        st.markdown("### 💰 Potential Electricity Savings Calculator")

        st.write(
            "Choose a reduction target to estimate how much electricity "
            "and money your household could save."
        )

        # Get saved electricity details
        monthly_units = float(
            user_data.get("monthly_units", 0) or 0
        )

        tariff_rate = float(
            user_data.get("tariff", 0) or 0
        )

        if monthly_units > 0:

            reduction_percent = st.select_slider(
                "Select your electricity reduction target",
                options=[5, 10, 15],
                value=10,
                format_func=lambda x: f"{x}% reduction"
            )

            # Calculate estimated savings
            units_saved = (
                monthly_units * reduction_percent / 100
            )

            new_monthly_units = monthly_units - units_saved

            monthly_money_saved = units_saved * tariff_rate

            yearly_money_saved = monthly_money_saved * 12

            # Display savings
            st.markdown("#### 📊 Your Estimated Savings")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Current Monthly Usage",
                    f"{monthly_units:.2f} units"
                )

                st.metric(
                    "Electricity You Could Save",
                    f"{units_saved:.2f} units",
                    delta=f"{reduction_percent}% reduction"
                )

            with col2:

                st.metric(
                    "New Monthly Usage",
                    f"{new_monthly_units:.2f} units"
                )

                st.metric(
                    "Estimated Monthly Money Saved",
                    f"₹{monthly_money_saved:,.2f}"
                )

            st.success(
                f"🎉 By reducing electricity consumption by "
                f"{reduction_percent}%, you could save approximately "
                f"₹{yearly_money_saved:,.2f} per year."
            )

            st.caption(
                f"Calculation uses your saved tariff of "
                f"₹{tariff_rate:.2f}/unit. Actual bill savings may "
                "differ because electricity bills use progressive "
                "tariff slabs, fixed charges, FAC and taxes."
            )

        else:

            st.warning(
                "Please add your monthly electricity consumption "
                "in the My Electricity Details page to use this calculator."
            )


    
    # ==================================================
    # ELECTRICITY BILL ESTIMATOR
    # MSEDCL RESIDENTIAL TARIFF - FY 2026-27
    # ==================================================

    elif selected_page == "Electricity Bill Estimator":

        st.title("🧾 Electricity Bill Estimator")

        st.write(
            "Estimate your residential electricity bill using "
            "MSEDCL Maharashtra tariff slabs for FY 2026-27."
        )

        st.info(
            "This estimator uses MSEDCL LT-I(B) residential rates. "
            "It is suitable only for consumers billed by MSEDCL. "
            "Other Maharashtra electricity distributors may have "
            "different tariffs."
        )

        # ---------------- TARIFF CONFIGURATION ----------------

        ENERGY_SLABS = [
            (100, 3.96),
            (300, 10.80),
            (500, 15.03),
            (float("inf"), 17.53)
        ]

        WHEELING_RATE = 1.60
        SINGLE_PHASE_FIXED = 130.00
        THREE_PHASE_FIXED = 435.00
        ELECTRICITY_DUTY_RATE = 16.0
        MUNICIPAL_FIXED_CHARGE = 10.00

        # ---------------- SAVED USER DETAILS ----------------

        saved_units = max(
            0.0,
            float(user_data.get("monthly_units", 0) or 0)
        )

        # ---------------- RESET CALLBACK ----------------

        def reset_bill_estimator():

            st.session_state["bill_units"] = saved_units
            st.session_state["bill_phase"] = "Single Phase"
            st.session_state["bill_months"] = 1
            st.session_state["bill_municipal"] = False

            st.session_state["bill_fac_1"] = 0.0
            st.session_state["bill_fac_2"] = 0.0
            st.session_state["bill_fac_3"] = 0.0
            st.session_state["bill_fac_4"] = 0.0

        # ---------------- BILL INPUT FORM ----------------

        with st.container(border=True):

            st.markdown("### Enter Bill Details")

            with st.form("msedcl_bill_form"):

                col1, col2 = st.columns(2)

                with col1:

                    units = st.number_input(
                        "Total Electricity Consumption (Units)",
                        min_value=0.0,
                        max_value=100000.0,
                        value=float(saved_units),
                        step=10.0,
                        key="bill_units"
                    )

                    billing_months = st.selectbox(
                        "Billing Period",
                        options=[1, 2, 3],
                        format_func=lambda x: (
                            f"{x} Month{'s' if x > 1 else ''}"
                        ),
                        key="bill_months"
                    )

                with col2:

                    connection_type = st.selectbox(
                        "Connection Type",
                        ["Single Phase", "Three Phase"],
                        key="bill_phase"
                    )

                    municipal_area = st.checkbox(
                        "My connection is in a municipal corporation area",
                        value=False,
                        key="bill_municipal"
                    )

                st.markdown("---")

                st.markdown("### Monthly Fuel Adjustment Charges")

                st.caption(
                    "Enter the FAC rates from your electricity bill. "
                    "These rates can change monthly. Keep zero if unknown."
                )

                fac_col1, fac_col2 = st.columns(2)

                with fac_col1:

                    fac_1 = st.number_input(
                        "FAC: First 100 units (₹/unit)",
                        min_value=-5.0,
                        max_value=10.0,
                        value=0.0,
                        step=0.05,
                        key="bill_fac_1"
                    )

                    fac_2 = st.number_input(
                        "FAC: 101–300 units (₹/unit)",
                        min_value=-5.0,
                        max_value=10.0,
                        value=0.0,
                        step=0.05,
                        key="bill_fac_2"
                    )

                with fac_col2:

                    fac_3 = st.number_input(
                        "FAC: 301–500 units (₹/unit)",
                        min_value=-5.0,
                        max_value=10.0,
                        value=0.0,
                        step=0.05,
                        key="bill_fac_3"
                    )

                    fac_4 = st.number_input(
                        "FAC: Above 500 units (₹/unit)",
                        min_value=-5.0,
                        max_value=10.0,
                        value=0.0,
                        step=0.05,
                        key="bill_fac_4"
                    )

                calculate_bill = st.form_submit_button(
                    "⚡ Calculate Electricity Bill",
                    use_container_width=True
                )

            st.button(
                "↩️ Reset Bill Details",
                use_container_width=True,
                on_click=reset_bill_estimator
            )

        # ---------------- BILL CALCULATION ----------------

        if calculate_bill:

            # Scale slab limits according to billing period.
            slab_limits = [
                (100 * billing_months, 3.96),
                (300 * billing_months, 10.80),
                (500 * billing_months, 15.03),
                (float("inf"), 17.53)
            ]

            fac_rates = [
                fac_1,
                fac_2,
                fac_3,
                fac_4
            ]

            remaining_units = units
            previous_limit = 0.0

            energy_charges = 0.0
            fac_charges = 0.0
            slab_details = []

            # Calculate progressive slab-wise charges.
            for index, (limit, rate) in enumerate(slab_limits):

                slab_units = min(
                    remaining_units,
                    limit - previous_limit
                )

                if slab_units > 0:

                    energy_amount = slab_units * rate

                    fac_amount = (
                        slab_units * fac_rates[index]
                    )

                    energy_charges += energy_amount
                    fac_charges += fac_amount

                    slab_details.append({
                        "Slab": (
                            "First 100 units"
                            if index == 0
                            else
                            "101–300 units"
                            if index == 1
                            else
                            "301–500 units"
                            if index == 2
                            else
                            "Above 500 units"
                        ),
                        "Units": slab_units,
                        "Rate (₹)": rate,
                        "Energy Charges (₹)": energy_amount
                    })

                    remaining_units -= slab_units

                previous_limit = limit

                if remaining_units <= 0:
                    break

            # Wheeling charge on all consumed units.
            wheeling_charges = units * WHEELING_RATE

            # Fixed charge according to connection type.
            if connection_type == "Single Phase":

                fixed_charges = (
                    SINGLE_PHASE_FIXED * billing_months
                )

            else:

                fixed_charges = (
                    THREE_PHASE_FIXED * billing_months
                )

            # Optional municipal corporation fixed charge.
            municipal_charges = (
                MUNICIPAL_FIXED_CHARGE * billing_months
                if municipal_area
                else 0.0
            )

            # Electricity duty.
            duty_base = (
                fixed_charges
                + municipal_charges
                + energy_charges
                + wheeling_charges
                + fac_charges
            )

            electricity_duty = (
                duty_base * ELECTRICITY_DUTY_RATE / 100
            )

            # Final estimated bill.
            total_bill = (
                duty_base + electricity_duty
            )

            # ---------------- DISPLAY BILL ----------------

            st.markdown("---")

            st.markdown("## 📊 Your Electricity Bill")

            st.caption(
                f"Billing period: {billing_months} month(s) | "
                f"Consumption: {units:,.2f} units | "
                f"Connection: {connection_type}"
            )

            st.markdown("### Slab-wise Energy Charges")

            if slab_details:

                slab_df = pd.DataFrame(slab_details)

                st.dataframe(
                    slab_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info("No electricity consumption entered.")

            st.markdown("### Detailed Bill Breakdown")

            bill_breakdown = pd.DataFrame({
                "Bill Component": [
                    "Energy Charges",
                    "Wheeling Charges",
                    "Fuel Adjustment Charges",
                    "Fixed Charges",
                    "Municipal Corporation Charges",
                    "Electricity Duty (16%)"
                ],
                "Amount (₹)": [
                    energy_charges,
                    wheeling_charges,
                    fac_charges,
                    fixed_charges,
                    municipal_charges,
                    electricity_duty
                ]
            })

            st.dataframe(
                bill_breakdown,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("---")

            result_col1, result_col2, result_col3 = st.columns(3)

            result_col1.metric(
                "Energy Charges",
                f"₹{energy_charges:,.2f}"
            )

            result_col2.metric(
                "Additional Charges",
                f"₹{(
                    wheeling_charges
                    + fac_charges
                    + fixed_charges
                    + municipal_charges
                ):,.2f}"
            )

            result_col3.metric(
                "Electricity Duty",
                f"₹{electricity_duty:,.2f}"
            )

            st.success(
                f"### Estimated Electricity Bill: ₹{total_bill:,.2f}"
            )

            st.caption(
                "Calculation uses progressive MSEDCL residential "
                "slabs, wheeling charges, fixed charges, entered FAC "
                "rates and 16% electricity duty. This is an estimate, "
                "not an official bill. Arrears, credits, rebates, "
                "interest, other adjustments and rounding may differ."
            )


    # ==================================================
    # CONSUMPTION TRENDS
    # ==================================================

    elif selected_page == "Consumption Trends":

        st.title("📈 Consumption Trends")

        st.write(
            "Record your monthly meter readings and view your "
            "household electricity consumption trend."
        )

        current_data = load_electricity_data()

        current_user_data = current_data.get(
            user_email, {}
        )

        monthly_records = current_user_data.get(
            "monthly_records", []
        )

        st.markdown("### Add Monthly Electricity Reading")

        with st.container(border=True):

            with st.form("monthly_reading_form"):

                reading_month = st.date_input(
                    "Select Month",
                    value=date.today()
                )

                monthly_reading_units = st.number_input(
                    "Monthly Consumption (Units)",
                    min_value=0.0,
                    value=0.0,
                    step=10.0
                )

                monthly_reading_bill = st.number_input(
                    "Monthly Bill (₹)",
                    min_value=0.0,
                    value=0.0,
                    step=100.0
                )

                save_reading = st.form_submit_button(
                    "Save Monthly Reading",
                    use_container_width=True
                )

                if save_reading:

                    month_key = reading_month.strftime("%Y-%m")

                    # Replace an existing reading for the same month.

                    monthly_records = [
                        record
                        for record in monthly_records
                        if record.get("month") != month_key
                    ]

                    monthly_records.append({

                        "month": month_key,
                        "units": monthly_reading_units,
                        "bill": monthly_reading_bill

                    })

                    monthly_records.sort(
                        key=lambda x: x["month"]
                    )

                    current_user_data["monthly_records"] = (
                        monthly_records
                    )

                    current_data[user_email] = current_user_data

                    save_electricity_data(current_data)

                    st.success(
                        "Monthly reading saved successfully!"
                    )

                    st.rerun()

        st.markdown("### Monthly Consumption Overview")

        if monthly_records:

            trend_df = pd.DataFrame(monthly_records)

            trend_df["month"] = trend_df["month"].astype(str)

            trend_df = trend_df.sort_values("month")

            trend_col1, trend_col2 = st.columns(2)

            trend_col1.metric(
                "Recorded Months",
                len(trend_df)
            )

            trend_col2.metric(
                "Average Monthly Units",
                f"{trend_df['units'].mean():,.2f}"
            )

            st.markdown("#### Monthly Electricity Consumption")

            chart_df = trend_df.set_index("month")

            st.bar_chart(
                chart_df[["units"]],
                color="#087E8B"
            )

            st.markdown("#### Monthly Electricity Bill")

            st.line_chart(
                chart_df[["bill"]],
                color="#D49A36"
            )

            st.markdown("#### Recorded Readings")

            st.dataframe(
                trend_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No monthly readings have been saved yet. "
                "Add your first reading above to start tracking trends."
            )

        st.markdown("### Prediction Category History")

        if user_history:

            prediction_df = pd.DataFrame(user_history)

            category_counts = (
                prediction_df["overall"]
                .value_counts()
                .rename_axis("Category")
                .reset_index(name="Number of Predictions")
            )

            st.bar_chart(
                category_counts.set_index("Category"),
                color="#2A9D8F"
            )

        else:

            st.info(
                "Prediction category trends will appear after "
                "you use the Consumption Prediction page."
            )

        st.caption(
            "Monthly charts use readings entered by you. "
            "Prediction category charts show saved model results, "
            "not actual electricity-unit measurements."
        )


    # ==================================================
    # REPORTS AND HISTORY
    # ==================================================

    elif selected_page == "Reports & History":

        st.title("📅 Reports & History")

        st.write(
            "View your previous machine learning predictions "
            "and download your records."
        )

        history_data = load_prediction_history()

        user_history = history_data.get(user_email, [])

        if user_history:

            history_df = pd.DataFrame(user_history)

            st.markdown("### Prediction Summary")

            total_predictions = len(history_df)

            category_counts = (
                history_df["overall"].value_counts()
            )

            h1, h2, h3 = st.columns(3)

            h1.metric(
                "Total Predictions",
                total_predictions
            )

            h2.metric(
                "High Results",
                int(category_counts.get("High", 0))
            )

            h3.metric(
                "Low Results",
                int(category_counts.get("Low", 0))
            )

            st.markdown("### Prediction Records")

            display_columns = [
                "date",
                "time",
                "random_forest",
                "svm",
                "xgboost",
                "overall",
                "votes"
            ]

            st.dataframe(
                history_df[display_columns],
                use_container_width=True,
                hide_index=True
            )

            csv_data = history_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Prediction History CSV",
                data=csv_data,
                file_name="electricity_prediction_history.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.markdown("### Prediction Category Distribution")

            st.bar_chart(
                history_df["overall"].value_counts(),
                color="#087E8B"
            )

        else:

            st.info(
                "No prediction history is available yet. "
                "Make a prediction to create your first record."
            )

        # MONTHLY READING EXPORT

        monthly_records = user_data.get(
            "monthly_records", []
        )

        st.markdown("### Monthly Electricity Records")

        if monthly_records:

            monthly_df = pd.DataFrame(monthly_records)

            st.dataframe(
                monthly_df,
                use_container_width=True,
                hide_index=True
            )

            monthly_csv = monthly_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Monthly Electricity Records",
                data=monthly_csv,
                file_name="monthly_electricity_records.csv",
                mime="text/csv",
                use_container_width=True
            )

        else:

            st.info(
                "No monthly electricity readings are saved yet."
            )


    # ==================================================
    # MY PROFILE
    # ==================================================

    elif selected_page == "My Profile":

        st.title("👤 My Profile")

        st.write(
            "View and update your account information."
        )

        users = load_users()

        current_user = users.get(user_email, {})

        with st.container(border=True):

            st.markdown("### Account Information")

            st.write(
                f"**Full Name:** {current_user.get('full_name', '')}"
            )

            st.write(
                f"**Email Address:** {user_email}"
            )

            st.write(
                "**Account Status:** Active"
            )

        st.markdown("### Update Your Name")

        with st.form("update_profile_form"):

            updated_name = st.text_input(
                "Full Name",
                value=current_user.get("full_name", "")
            )

            update_profile = st.form_submit_button(
                "Update Profile",
                use_container_width=True
            )

            if update_profile:

                if not updated_name.strip():

                    st.error("Name cannot be empty.")

                else:

                    users[user_email]["full_name"] = (
                        updated_name.strip()
                    )

                    save_users(users)

                    st.session_state.full_name = (
                        updated_name.strip()
                    )

                    st.success(
                        "Profile updated successfully!"
                    )

                    st.rerun()

        st.markdown("### Household Summary")

        if user_data:

            p1, p2, p3 = st.columns(3)

            p1.metric(
                "Monthly Units",
                f"{user_data.get('monthly_units', 0):,.1f}"
            )

            p2.metric(
                "Family Members",
                user_data.get("family_members", 1)
            )

            p3.metric(
                "Household Type",
                user_data.get("household_type", "Residential")
            )

        else:

            st.info(
                "Add your household details to view your summary."
            )


    # ==================================================
    # ABOUT PROJECT
    # ==================================================

    elif selected_page == "About Project":

        st.title("ℹ️ About Smart Electricity Advisor")

        st.write(
            "Smart Electricity Consumption and Saving Advisor "
            "is a machine learning-based application designed to "
            "analyze electricity readings, classify consumption "
            "levels and encourage energy-efficient habits."
        )

        st.markdown("### Project Objectives")

        st.markdown("""
        - Analyze household electricity readings.
        - Classify electricity consumption into Low, Medium and High.
        - Compare three machine learning algorithms.
        - Provide practical electricity-saving suggestions.
        - Help users estimate electricity bills.
        - Maintain user-specific prediction and monthly reading history.
        """)

        st.markdown("### Dataset")

        st.write(
            "Individual Household Electric Power Consumption dataset "
            "from the UCI Machine Learning Repository."
        )

        st.markdown("### Machine Learning Algorithms")

        with st.container(border=True):

            st.markdown("#### 🌳 Random Forest Classifier")

            st.write(
                "Uses multiple decision trees and combines their "
                "predictions to classify electricity consumption."
            )

            st.markdown("#### 📊 Support Vector Machine (SVM)")

            st.write(
                "Uses a scaled feature set to classify electricity "
                "consumption into different categories."
            )

            st.markdown("#### 🚀 XGBoost Classifier")

            st.write(
                "Uses gradient boosting to build an ensemble of "
                "decision trees for classification."
            )

        st.markdown("### Prediction Categories")

        category1, category2, category3 = st.columns(3)

        category1.success("🟢 Low")
        category2.warning("🟡 Medium")
        category3.error("🔴 High")

        st.markdown("### Technologies Used")

        st.markdown("""
        - Python
        - Streamlit
        - Pandas
        - Scikit-learn
        - XGBoost
        - Joblib
        - JSON-based local storage
        """)

        st.caption(
            "The application provides classification results and "
            "estimated bill calculations. It does not directly measure "
            "electricity consumption from a physical smart meter."
        )
