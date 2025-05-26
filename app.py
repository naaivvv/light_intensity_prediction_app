
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json

st.set_page_config(page_title="Light Intensity (Is) & Pollution Level Predictor", layout="wide")

# --- Constants for file paths and value ranges ---
MODEL_PATH = 'light_pollution_model.keras'
SCALER_PATH = 'scaler.pkl'
FEATURES_PATH = 'feature_names.json'
CATEGORIES_PATH = 'original_dataset_categories.json'

# Define the 'Is' value range for pollution visualization
MIN_IS_VALUE_FOR_POLLUTION_RANGE = -2.5
MAX_IS_VALUE_FOR_POLLUTION_RANGE = -0.5

# --- Cached Resource Loading Functions ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Ensure '{MODEL_PATH}' is trained for 'Is' and is a valid Keras model file.")
        st.stop()

@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load(SCALER_PATH)
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler: {e}. Ensure '{SCALER_PATH}' is compatible with the features for 'Is' prediction.")
        st.stop()

@st.cache_data
def load_feature_info():
    try:
        with open(FEATURES_PATH, 'r') as f:
            all_feature_names = json.load(f)
        with open(CATEGORIES_PATH, 'r') as f:
            original_dataset_categories = json.load(f)
        return all_feature_names, original_dataset_categories
    except Exception as e:
        st.error(f"Error loading feature info: {e}. Ensure JSON files list correct features for 'Is' prediction.")
        st.stop()

# --- Load Resources ---
model = load_model()
scaler = load_scaler()
all_feature_names, original_dataset_categories = load_feature_info()

# --- Prediction Display Function ---
def display_prediction_with_pollution_level(is_value):
    is_value_pyfloat = float(is_value)

    st.metric(label="📊 Predicted Intensity (Is)", value=f"{is_value_pyfloat:.4f}")

    # --- Enhanced Progress Bar Section ---
    st.markdown("---")
    st.markdown("#### Intensity Visualization on Pollution Scale")
    st.caption(
        f"This bar shows where the predicted 'Is' value ({is_value_pyfloat:.4f}) falls within the "
        f"typical assessment range: **{MIN_IS_VALUE_FOR_POLLUTION_RANGE}** (lower pollution potential) "
        f"to **{MAX_IS_VALUE_FOR_POLLUTION_RANGE}** (higher pollution potential)."
    )

    progress_val_calculated = (is_value_pyfloat - MIN_IS_VALUE_FOR_POLLUTION_RANGE) / \
                              (MAX_IS_VALUE_FOR_POLLUTION_RANGE - MIN_IS_VALUE_FOR_POLLUTION_RANGE)
    clipped_progress_val = np.clip(float(progress_val_calculated), 0.0, 1.0)

    # Determine bar color based on 'Is' value relative to pollution thresholds
    bar_color = "#757575" # Default grey
    intensity_range_for_color = MAX_IS_VALUE_FOR_POLLUTION_RANGE - MIN_IS_VALUE_FOR_POLLUTION_RANGE

    if intensity_range_for_color > 0:
        if is_value_pyfloat < (MIN_IS_VALUE_FOR_POLLUTION_RANGE + intensity_range_for_color * 0.33):
            bar_color = "#66BB6A"  # Softer Green
        elif is_value_pyfloat < (MIN_IS_VALUE_FOR_POLLUTION_RANGE + intensity_range_for_color * 0.66):
            bar_color = "#FFA726"  # Softer Orange
        else:
            bar_color = "#EF5350"  # Softer Red

        # Adjust color if value is outside the defined range
        if is_value_pyfloat > MAX_IS_VALUE_FOR_POLLUTION_RANGE : bar_color = "#D32F2F" # Darker Red
        if is_value_pyfloat < MIN_IS_VALUE_FOR_POLLUTION_RANGE : bar_color = "#388E3C" # Darker Green

    progress_bar_html = f"""
    <div style="background-color: #e0e0e0; border-radius: 5px; padding: 3px; margin-top: 5px; margin-bottom: 5px;">
        <div style="width: {clipped_progress_val*100}%; background-color: {bar_color}; height: 24px; border-radius: 3px; text-align: center; color: white; font-weight: bold; line-height: 24px; font-size: 0.9em;">
            {clipped_progress_val*100:.0f}%
        </div>
    </div>
    """
    st.markdown(progress_bar_html, unsafe_allow_html=True)

    # Display min/max labels for the bar
    p_col1, p_col2 = st.columns([1,1])
    with p_col1:
        st.caption(f"Scale Min: {MIN_IS_VALUE_FOR_POLLUTION_RANGE}")
    with p_col2:
        st.caption(f"Scale Max: {MAX_IS_VALUE_FOR_POLLUTION_RANGE}")

    if is_value_pyfloat > MAX_IS_VALUE_FOR_POLLUTION_RANGE:
        st.warning(f"⚠️ Intensity ({is_value_pyfloat:.4f}) is **above** the defined maximum ({MAX_IS_VALUE_FOR_POLLUTION_RANGE}) for this visualization, suggesting very high pollution potential.", icon="📈")
    elif is_value_pyfloat < MIN_IS_VALUE_FOR_POLLUTION_RANGE:
        st.info(f"ℹ️ Intensity ({is_value_pyfloat:.4f}) is **below** the defined minimum ({MIN_IS_VALUE_FOR_POLLUTION_RANGE}) for this visualization, suggesting very low pollution potential.", icon="📉")

    st.markdown("---")

    # --- Estimated Pollution Level Section ---
    st.subheader("📝 Estimated Pollution Level based on Intensity (Is):")

    if intensity_range_for_color <= 0:
        st.warning("Min/Max 'Is' range for pollution interpretation is not correctly defined.")
        return

    # Define pollution levels based on 'Is' value
    if is_value_pyfloat < (MIN_IS_VALUE_FOR_POLLUTION_RANGE + intensity_range_for_color * 0.33):
        st.markdown("### <span style='color:#2E7D32;'>🌿 Low Pollution (Likely Darker/Less Populated Area)</span>", unsafe_allow_html=True)
        st.markdown("The predicted intensity suggests a lower level of artificial light, often found in areas with less human activity or effective light control.")
    elif is_value_pyfloat < (MIN_IS_VALUE_FOR_POLLUTION_RANGE + intensity_range_for_color * 0.66):
        st.markdown("### <span style='color:#FF8F00;'>🏙️ Moderate Pollution (Suburban/Transition Area)</span>", unsafe_allow_html=True)
        st.markdown("The predicted intensity indicates a noticeable presence of artificial light, typical of suburban areas or transitional zones between urban and rural environments.")
    else:
        st.markdown("### <span style='color:#C62828;'>🏭 High Pollution (Likely Densely Populated/Urban Area)</span>", unsafe_allow_html=True)
        st.markdown("The predicted intensity is high, suggesting significant artificial light sources, characteristic of densely populated urban centers or industrial areas.")

# --- Main Application ---
st.title("✨ Light Intensity (Is) & Pollution Level Predictor")
st.markdown("Input observational parameters to predict 'Is' (Intensity) and get an estimation of the associated light pollution level. Higher negative 'Is' values typically indicate less light pollution.")

if model and scaler and all_feature_names:
    st.sidebar.header("⚙️ Input Features:")
    inputs = {}

    inputs['Altitude_m'] = st.sidebar.slider("Altitude (m)", min_value=70.0, max_value=100.0, value=70.0, step=1.0, help="Altitude in meters.")
    inputs['Exposure_time_sec'] = st.sidebar.slider("Exposure Time (sec)", min_value=0.16, max_value=0.50, value=0.16, step=0.01, help="Camera exposure time in seconds.")
    inputs['NSB_mpsas'] = st.sidebar.number_input("NSB (mpsas)", min_value=13.0, max_value=18.0, value=14.5, format="%.2f", step=0.01, help="Natural Sky Brightness in magnitudes per square arcsecond.")

    st.sidebar.markdown("---") # Separator
    st.sidebar.markdown("##### Sensor Readings (Log Scale)")
    s_col1, s_col2 = st.sidebar.columns(2)
    inputs['Gs'] = s_col1.number_input("Gs (Green)", min_value=-3.0, max_value=-0.9, value=-1.66, format="%.3f", step=0.01, help="Green sensor reading (log scale).")
    inputs['Bs'] = s_col1.number_input("Bs (Blue)", min_value=-3.1, max_value=-1.1, value=-2.41, format="%.3f", step=0.01, help="Blue sensor reading (log scale).")
    inputs['Rs'] = s_col2.number_input("Rs (Red)", min_value=-2.7, max_value=-0.7, value=-1.34, format="%.3f", step=0.01, help="Red sensor reading (log scale).")
    st.sidebar.markdown("---") # Separator

    selected_dataset_category = None
    if original_dataset_categories:
        selected_dataset_category = st.sidebar.selectbox(
            "Dataset Category (Origin)",
            options=original_dataset_categories,
            index=0,
            help="Select the original dataset category if known, influences one-hot encoding."
        )
    else:
        st.sidebar.text("No 'dataset' categories found for selection.")

    if st.sidebar.button("🚀 Predict Intensity & Pollution Level", type="primary", use_container_width=True):
        input_df = pd.DataFrame(columns=all_feature_names)
        input_df.loc[0] = 0.0 # Initialize all features to 0.0

        # Assign numerical inputs
        if 'Altitude_m' in all_feature_names: input_df.at[0, 'Altitude_m'] = float(inputs['Altitude_m'])
        if 'Exposure_time_sec' in all_feature_names: input_df.at[0, 'Exposure_time_sec'] = float(inputs['Exposure_time_sec'])
        if 'NSB_mpsas' in all_feature_names: input_df.at[0, 'NSB_mpsas'] = float(inputs['NSB_mpsas'])
        if 'Rs' in all_feature_names: input_df.at[0, 'Rs'] = float(inputs['Rs'])
        if 'Gs' in all_feature_names: input_df.at[0, 'Gs'] = float(inputs['Gs'])
        if 'Bs' in all_feature_names: input_df.at[0, 'Bs'] = float(inputs['Bs'])

        # Handle one-hot encoded dataset category
        if selected_dataset_category:
            one_hot_col_name = f"dataset_{selected_dataset_category}"
            if one_hot_col_name in all_feature_names:
                input_df.at[0, one_hot_col_name] = 1.0 # Set selected category to 1.0
            else:
                st.warning(f"Warning: One-hot column '{one_hot_col_name}' not found in expected features: {all_feature_names}")

        # Ensure all columns are numeric and handle potential missing features not set by UI
        for col in input_df.columns:
            if col not in ['Altitude_m', 'Exposure_time_sec', 'NSB_mpsas', 'Rs', 'Gs', 'Bs'] and not col.startswith('dataset_'):
                 if input_df.at[0, col] == 0.0 and col in all_feature_names: # Check if it's still default
                    st.caption(f"Note: Feature '{col}' used default value 0.0 as it was not set via UI.")
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0.0)


        if input_df.isnull().values.any():
            st.error("Error: Input data contains NaNs after processing. Check feature preparation logic.")
        else:
            try:
                input_df_float = input_df.astype(float)
                scaled_input = scaler.transform(input_df_float)
                prediction = model.predict(scaled_input)
                predicted_is_val = prediction[0][0]

                st.subheader("📈 Prediction Result:")
                display_prediction_with_pollution_level(predicted_is_val)

            except ValueError as ve:
                st.error(f"ValueError during prediction or scaling: {ve}")
                st.error("This might be due to a mismatch in expected feature columns or data types.")
                st.dataframe(input_df_float.head())
                st.text(f"Input DataFrame dtypes:\n{input_df_float.dtypes}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.dataframe(input_df_float.head())
                st.text(f"Input DataFrame dtypes:\n{input_df_float.dtypes}")

    st.sidebar.markdown("---")
    st.sidebar.info("Adjust feature values using the controls above and click 'Predict' to see the results.")

else:
    st.error("🔴 App Critical Error: Essential resources (model, scaler, or feature info) failed to load. Please check file paths and configurations. Application cannot run.")

# --- Footer and Credits ---
st.markdown("<br><hr style='margin-top: 30px; margin-bottom: 30px;'><br>", unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align: center; color: #4A4A4A; font-size: 0.9em;'>
        <p>Developed by: <b>Edwin P. Bayog Jr.</b><br>
        <i>BSCpE 3-A</i></p>
        <p style='margin-top: 5px;'>Course: <b>CPE333-V Environmental Engineering</b></p>
    </div>
    <br>
    """,
    unsafe_allow_html=True
)
