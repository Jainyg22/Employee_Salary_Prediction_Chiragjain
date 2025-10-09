import streamlit as st
import pandas as pd
import joblib


# Load encoders
workclass_encoder = joblib.load("workclass_encoder.pkl")
occupation_encoder = joblib.load("occupation_encoder.pkl")
relationship_encoder = joblib.load("relationship_encoder.pkl")
race_encoder =  joblib.load("race_encoder.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
native_encoder = joblib.load("native-country_encoder.pkl")
marital_encoder = joblib.load("marital-status_encoder.pkl")


# --- Load Trained Model ---
model = joblib.load("best_model.pkl")

# --- Page Config ---
st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼", layout="wide")

# --- Styling ---
st.markdown("""
    <style>
        .main { background-color: #f5f5f5; }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            font-weight: bold;
        }
        .stMarkdown {
            font-size: 16px;
        }
        .footer {
            font-size: 12px;
            text-align: center;
            margin-top: 50px;
            color: gray;
        }
        /* Attempt to center the header */
        .st-emotion-cache-vk3wp9 h1 {
            text-align: center;
        }

    </style>
""", unsafe_allow_html=True)

# --- Title ---
#! Centering the title using markdown for better control
st.markdown("<h1 style='text-align: center;'>💼 Employee Salary Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>🔍 Predict whether an employee earns >$50K or ≤$50K using ML!</h4>", unsafe_allow_html=True)


st.markdown("---")

# --- Sidebar Inputs ---
st.sidebar.header("📋 Input Employee Details")

# Mapping education levels to educational-num
education_mapping = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5,
    "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10,
    "Assoc-voc": 11, "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14,
    "Prof-school": 15, "Doctorate": 16
}

with st.sidebar.form("user_input_form"):
    age = st.slider("👤 Age", 18, 65, 30, help="Age of the employee")
    workclass = st.selectbox("🏢 Workclass", workclass_encoder.classes_, help="Type of employer")
    education = st.selectbox("🎓 Education Level", list(education_mapping.keys()), help="Highest level of education achieved")
    marital_status = st.selectbox("❤️ Marital Status", marital_encoder.classes_, help="Marital status of the employee")
    occupation = st.selectbox("💼 Job Role", occupation_encoder.classes_, help="Employee's occupation")
    relationship = st.selectbox("👨‍👩‍👧‍👦 Relationship", relationship_encoder.classes_, help="Employee's relationship status")
    race = st.selectbox("🌍 Race", race_encoder.classes_, help="Employee's race")
    gender = st.selectbox("🚻 Gender", gender_encoder.classes_, help="Employee's gender")
    capital_gain = st.slider("📈 Capital Gain", 0, 20000, 0, step=100, help="Capital gains from investments")
    capital_loss = st.slider("📉 Capital Loss", 0, 2500, 0, step=100, help="Capital losses from investments")
    hours_per_week = st.slider("⏱ Hours per Week", 20, 60, 40, help="Number of hours worked per week")
    native_country = st.selectbox("🌍 Native Country", native_encoder.classes_, help="Employee's country of origin")


    submitted = st.form_submit_button("🔮 Predict")


# --- Single Prediction ---
if submitted:
    input_df = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'educational-num': [education_mapping[education]], # Map education to educational-num
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship' : [relationship],
        'race':[race],
        'gender' :[gender],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week' : [hours_per_week],
        'native-country' : [native_country],
    })

    # Apply to input_df
    input_df_encoded = input_df.copy()
    input_df_encoded['workclass'] = workclass_encoder.transform(input_df_encoded['workclass'])
    input_df_encoded['occupation'] = occupation_encoder.transform(input_df_encoded['occupation'])
    input_df_encoded['relationship'] = relationship_encoder.transform(input_df_encoded['relationship'])
    input_df_encoded['race'] = race_encoder.transform(input_df_encoded['race'])
    input_df_encoded['gender'] = gender_encoder.transform(input_df_encoded['gender'])
    input_df_encoded['native-country'] = native_encoder.transform(input_df_encoded['native-country'])
    input_df_encoded['marital-status'] = marital_encoder.transform(input_df_encoded['marital-status'])


    st.markdown("### 🧾 Input Summary")
    #! Displaying dataframe without index and to full width
    st.dataframe(input_df, hide_index=True, use_container_width=True)

    # Prediction
    pred = model.predict(input_df_encoded)[0]
    label = ">50K" if pred == ">50K" else "<=50K" # Assuming model predicts string labels


    st.markdown("### 💡 Prediction Result")
    if label == ">50K":
        st.success("🎉 The predicted income is greater than $50K.")
    else:
        st.warning("💸 The predicted income is less than or equal to $50K.")


# --- Batch Prediction ---
st.markdown("---")
st.subheader("📂 Batch Prediction (Upload CSV)")

# Add a section to download a sample CSV
st.markdown("Download a sample CSV template for batch prediction:")
sample_data = pd.DataFrame(columns=['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'])
csv_template = sample_data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Download Sample CSV",
    data=csv_template,
    file_name='salary_prediction_template.csv',
    mime='text/csv',
)


uploaded_file = st.file_uploader("Upload CSV with the same columns: age, workclass, education, marital-status, occupation, relationship, race, gender, capital-gain, capital-loss, hours-per-week, native-country", type="csv")

if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded Data Preview:", batch_data.head())

    batch_data_encoded = batch_data.copy()

    # Map education to educational-num for batch data
    batch_data_encoded['educational-num'] = batch_data_encoded['education'].map(education_mapping)
    batch_data_encoded.drop(columns=['education'], errors='ignore', inplace=True) # Drop the original education column


    # Apply encoders to batch data
    # Handle potential unseen labels during transformation by replacing them with a placeholder or the most frequent category
    for col, encoder in zip(['workclass', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'marital-status'],
                             [workclass_encoder, occupation_encoder, relationship_encoder, race_encoder, gender_encoder, native_encoder, marital_encoder]):
        # Use a try-except block to handle unseen labels during transform
        try:
            batch_data_encoded[col] = encoder.transform(batch_data_encoded[col])
        except ValueError as e:
            st.warning(f"Warning: Unseen labels found in column '{col}' during encoding: {e}. Replacing with a placeholder.")
            # Replace unseen labels with a placeholder or handle as appropriate
            # One approach is to replace unseen values with a value that will result in a specific encoded value (e.g., the encoded value of 'Others' or a new value for 'Unknown')
            if 'Others' in encoder.classes_:
                 unseen_value_encoded = encoder.transform(['Others'])[0]
            else:
                # If 'Others' is not in classes, find a suitable default or handle differently
                unseen_value_encoded = -1 # Placeholder or a value that signifies 'Unknown'

            batch_data_encoded[col] = batch_data_encoded[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else unseen_value_encoded)



    batch_preds = model.predict(batch_data_encoded)
    batch_data['Predicted Income'] = [">50K" if p == ">50K" else "<=50K" for p in batch_preds] # Assuming model predicts string labels

    # Decode categorical columns back to original values, handling potential errors
    for col, encoder in zip(['workclass', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'marital-status'],
                             [workclass_encoder, occupation_encoder, relationship_encoder, race_encoder, gender_encoder, native_encoder, marital_encoder]):
        try:
            batch_data[col] = encoder.inverse_transform(batch_data_encoded[col])
        except ValueError as e:
             st.warning(f"Warning: Unseen labels found in column '{col}' during decoding: {e}. Cannot fully decode.")
             # If inverse_transform fails, the column will remain with encoded values or the placeholder if used during transform.
             # You might want to add a column indicating decoding issues or leave the encoded value.
             pass


    st.markdown("### ✅ Predictions:")
    #! Displaying dataframe without index and to full width
    st.dataframe(batch_data.head(), hide_index=True, use_container_width=True)

    # Download button
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Result CSV", csv, "salary_predictions.csv", "text/csv")

# --- Footer ---
st.markdown("""
    <div style='text-align: center; font-size: 14px; color: gray;'>
        <br><br>
        Made with ❤️ using Streamlit | Powered by a Machine Learning Classifier Algorithm<br>
        Trained on the Adult Employee Dataset<br>
        © 2025 Chirag Jain(NIT KKR)
    </div>
    """, unsafe_allow_html=True)
