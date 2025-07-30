import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Property Analysis & Prediction Engine", layout="wide")

# --- Data Cleaning and Preprocessing Function ---
def clean_and_prepare_data(df, original_df_ref):
    """
    Final script that cleans data and uses specific logical rules to remove outliers.
    """
    df_clean = df.copy()

    def convert_price(price_str):
        if not isinstance(price_str, str) or not re.search(r'\d', price_str): return None
        price_str = price_str.replace('Rp', '').replace(' ', '').replace(',', '.')
        try:
            if 'M' in price_str: num = float(price_str.replace('M', '')) * 1_000_000_000
            elif 'Jt' in price_str: num = float(price_str.replace('Jt', '')) * 1_000_000
            else: num = float(price_str)
            return int(num)
        except ValueError: return None

    df_clean['Price'] = df_clean['Price'].apply(convert_price)
    for col in ['Bedrooms', 'Building Area (m²)', 'Land Area (m²)']:
        df_clean[col] = df_clean[col].astype(str).str.extract(r'(\d+)').astype(float)

    df_clean.dropna(subset=['Price', 'Building Area (m²)', 'Land Area (m²)', 'Bedrooms'], inplace=True)
    df_clean = df_clean[(df_clean['Building Area (m²)'] > 0) & (df_clean['Land Area (m²)'] > 0)]
    df_clean['rejection_reason'] = ''

    # Outlier Removal Rules
    bed_density_mask = df_clean['Bedrooms'] / df_clean['Building Area (m²)'] >= 0.1
    df_clean.loc[bed_density_mask, 'rejection_reason'] += 'Impossible bedroom density; '

    price_land_mask = (df_clean['Price'] / df_clean['Land Area (m²)']) < 700_000
    df_clean.loc[price_land_mask, 'rejection_reason'] += 'Unrealistically low price for land area; '

    price_building_mask = (df_clean['Price'] / df_clean['Building Area (m²)']) > 150_000_000
    df_clean.loc[price_building_mask, 'rejection_reason'] += 'Unrealistically high price for building area; '
    
    building_ratio_mask = (df_clean['Building Area (m²)'] / df_clean['Land Area (m²)']) > 10
    df_clean.loc[building_ratio_mask, 'rejection_reason'] += 'Impossible building-to-land ratio; '

    # Separate good data from outliers
    cleaned_df = df_clean[df_clean['rejection_reason'] == ''].copy()
    dropped_df = original_df_ref.loc[df_clean[df_clean['rejection_reason'] != ''].index].copy()
    dropped_df['rejection_reason'] = df_clean[df_clean['rejection_reason'] != '']['rejection_reason']


    # Finalize the Clean DataFrame
    def extract_city(address_str):
        if not isinstance(address_str, str): return 'other'
        address_lower = address_str.lower()
        if 'jakarta' in address_lower: return 'jakarta'
        if 'tangerang' in address_lower: return 'tangerang'
        if 'bekasi' in address_lower: return 'bekasi'
        if 'depok' in address_lower: return 'depok'
        if 'bogor' in address_lower: return 'bogor'
        return 'other'

    cleaned_df.loc[:, 'City'] = original_df_ref['Address'].loc[cleaned_df.index].apply(extract_city)
    cleaned_df = cleaned_df.drop(columns=['Title', 'Address', 'rejection_reason'], errors='ignore')

    for col in ['Price', 'Bedrooms', 'Building Area (m²)', 'Land Area (m²)']:
         cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce').astype('Int64')

    final_cols = ['City', 'Bedrooms', 'Building Area (m²)', 'Land Area (m²)', 'Price']
    cleaned_df = cleaned_df[final_cols]

    return cleaned_df, dropped_df

# --- EDA and Analysis Functions (IMPROVED) ---
def perform_eda(df):
    """Generates and displays enhanced EDA plots in Streamlit."""
    st.header("Exploratory Data Analysis (EDA)")
    
    st.subheader("1. Price Distribution")
    use_log_scale = st.checkbox('Use Log Scale for Price Axis', value=True)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    price_data = df['Price'] / 1_000_000_000
    sns.histplot(price_data, kde=True, bins=50, ax=ax1, log_scale=use_log_scale, color='#3266CC')
    xlabel = 'Price (Miliar Rp)' + (' - Log Scale' if use_log_scale else '')
    ax1.set_xlabel(xlabel)
    median_price = price_data.median()
    ax1.axvline(median_price, color='red', linestyle='--', label=f'Median: {median_price:.2f} M')
    ax1.legend()
    st.pyplot(fig1)

    st.markdown("---")
    
    st.subheader("2. Price Distribution by City")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    city_order = df.groupby('City')['Price'].median().sort_values().index
    sns.violinplot(y='City', x=df['Price'] / 1_000_000_000, data=df, order=city_order, inner='quartile', ax=ax2)
    ax2.set_xlabel('Price (Miliar Rp)')
    st.pyplot(fig2)

    st.markdown("---")

    st.subheader("3. Average Price per m² by City")
    df['price_per_sqm'] = df['Price'] / df['Building Area (m²)']
    avg_price_per_sqm = df.groupby('City')['price_per_sqm'].mean().sort_values(ascending=False)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    avg_price_per_sqm.plot(kind='bar', ax=ax3)
    ax3.set_ylabel('Average Price / m² (in Juta Rp)')
    ax3.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig3)
    
    st.markdown("---")
    
    st.subheader("4. Correlation Between Features")
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    numerical_cols = ['Price', 'Bedrooms', 'Building Area (m²)', 'Land Area (m²)']
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax4)
    st.pyplot(fig4)
    
    st.markdown("---")
    
    st.subheader("5. Building Area vs. Price by City")
    fig5, ax5 = plt.subplots(figsize=(11, 7))
    sns.scatterplot(x='Building Area (m²)', y='Price', hue='City', data=df, alpha=0.7, ax=ax5)
    ax5.ticklabel_format(style='plain', axis='y')
    ax5.set_ylabel("Price (in Rp)")
    ax5.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    st.pyplot(fig5)

# --- Main App ---
st.title("All-in-One Property Analysis & Prediction App")

# Initialize session state
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'dropped_data' not in st.session_state:
    st.session_state.dropped_data = None
if 'model_data' not in st.session_state:
    st.session_state.model_data = None

# Create tabs
tab1, tab2, tab3 = st.tabs(["📂 Data Loader & EDA", "🛠️ Model Training", "🏠 Price Predictor"])

# --- Tab 1: Data Loader & EDA ---
with tab1:
    st.header("Upload and Process Your Data")
    uploaded_file = st.file_uploader("Upload your raw properti_data.csv file", type=["csv"])

    if uploaded_file is None:
        st.session_state.cleaned_data = None
        st.session_state.dropped_data = None
        st.session_state.model_data = None
        
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        
        # --- NEW: Beautiful Data Summary ---
        st.subheader("Uploaded Data Summary")
        summary_df_part1 = pd.DataFrame({
            "Metric": ["Number of Rows", "Number of Columns", "Duplicate Rows"],
            "Value": [raw_df.shape[0], raw_df.shape[1], raw_df.duplicated().sum()]
        })
        st.table(summary_df_part1)
        
        st.subheader("Column Data Types & Missing Values")
        summary_df_part2 = pd.DataFrame({
            "Column": raw_df.columns,
            "Data Type": raw_df.dtypes.values,
            "Missing Values": raw_df.isnull().sum().values
        })
        st.dataframe(summary_df_part2)

        with st.expander("Expand for Data Cleaning Options"):
            remove_duplicates = st.checkbox("Remove duplicate rows")
            if remove_duplicates:
                raw_df.drop_duplicates(inplace=True)
                st.info(f"Duplicate rows removed. Data now has {len(raw_df)} rows.")

            remove_missing = st.checkbox("Remove rows with any missing values")
            if remove_missing:
                raw_df.dropna(inplace=True)
                st.info(f"Rows with missing values removed. Data now has {len(raw_df)} rows.")

        if st.button("Process Data and Run EDA"):
            with st.spinner("Cleaning data, removing outliers, and generating plots..."):
                original_df = raw_df.copy()
                for col in original_df.select_dtypes(include=['object']).columns:
                    original_df[col] = original_df[col].str.replace('"', '', regex=False)

                cleaned_df, dropped_df = clean_and_prepare_data(original_df, original_df)
                st.session_state.cleaned_data = cleaned_df
                st.session_state.dropped_data = dropped_df

            st.success(f"Processing complete! Kept {len(cleaned_df)} rows and removed {len(dropped_df)} outliers.")

    if st.session_state.cleaned_data is not None:
        st.subheader("Cleaned Data Preview")
        st.dataframe(st.session_state.cleaned_data.head())

        if not st.session_state.dropped_data.empty:
            with st.expander("Expand to see Outlier Analysis"):
                st.subheader("Removed Outlier Data (with Reason)")
                st.dataframe(st.session_state.dropped_data)
        
        # Perform EDA on the cleaned data
        perform_eda(st.session_state.cleaned_data)

# --- Tab 2: Model Training ---
with tab2:
    st.header("Train the Prediction Model")
    if st.session_state.cleaned_data is None:
        st.warning("Please upload and process data in the 'Data Loader & EDA' tab first.")
    else:
        if st.button("Train RandomForest Model"):
            with st.spinner("Training model... This may take a moment."):
                df = st.session_state.cleaned_data
                dummies = pd.get_dummies(df['City'], prefix='City')
                df_model = pd.concat([df, dummies], axis=1)
                target = 'Price'
                features = ['Bedrooms', 'Building Area (m²)', 'Land Area (m²)'] + list(dummies.columns)
                X = df_model[features]
                y = df_model[target]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)

                predictions = model.predict(X_test)
                mae = mean_absolute_error(y_test, predictions)

                st.session_state.model_data = {'model': model, 'columns': features, 'mae': mae}

            st.success("Model training complete!")

    if st.session_state.model_data is not None:
        st.subheader("Model Performance")
        mae = st.session_state.model_data['mae']
        st.metric("Mean Absolute Error (MAE)", f"Rp {mae:,.0f}")
        st.info("This is the average error of the model's predictions on the test data.")


# --- Tab 3: Price Predictor ---
with tab3:
    st.header("Get a Price Prediction")
    if st.session_state.model_data is None:
        st.warning("Please train a model in the 'Model Training' tab first.")
    else:
        model = st.session_state.model_data['model']
        model_columns = st.session_state.model_data['columns']
        city_list = sorted([col.replace('City_', '') for col in model_columns if 'City_' in col])

        city = st.selectbox('City', options=city_list, key='pred_city')
        bedrooms = st.slider('Number of Bedrooms', min_value=1, max_value=15, value=3, key='pred_beds')
        building_area = st.number_input('Building Area (m²)', min_value=30, max_value=2000, value=150, key='pred_build')
        land_area = st.number_input('Land Area (m²)', min_value=40, max_value=5000, value=200, key='pred_land')

        if st.button("Predict Price"):
            # Sanity Check for Inputs
            is_unusual = False
            warning_messages = []
            if bedrooms / building_area >= 0.1:
                is_unusual = True
                warning_messages.append(f"- Having {bedrooms} bedrooms in a {building_area}m² building is highly unusual.")
            if building_area > land_area and (building_area / land_area > 10):
                is_unusual = True
                warning_messages.append(f"- A building area of {building_area}m² on a {land_area}m² plot is very rare.")

            if is_unusual:
                st.warning("Warning: The features you entered are unrealistic and may lead to an unreliable prediction.", icon="⚠️")
                for msg in warning_messages: st.write(msg)
                st.write("---")

            # Prepare Input and Predict
            input_data = {col: 0 for col in model_columns}
            input_data['Bedrooms'] = bedrooms
            input_data['Building Area (m²)'] = building_area
            input_data['Land Area (m²)'] = land_area
            if f'City_{city}' in input_data:
                input_data[f'City_{city}'] = 1

            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]

            # Calculate Confidence Score
            try:
                individual_tree_predictions = [tree.predict(input_df) for tree in model.estimators_]
                std_dev = np.std(individual_tree_predictions)
                lower_bound = prediction - (1.96 * std_dev)
                upper_bound = prediction + (1.96 * std_dev)
                uncertainty_percentage = (std_dev * 1.96 * 2) / prediction * 100
                confidence_percentage = max(0, 100 - uncertainty_percentage)
            except Exception:
                confidence_percentage, lower_bound, upper_bound = "N/A", "N/A", "N/A"

            # Display Results
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Estimated Property Price", value=f"Rp {prediction:,.0f}")
            with col2:
                st.metric(label="Confidence Score", value=f"{confidence_percentage:.1f} %" if isinstance(confidence_percentage, float) else "N/A")

            if isinstance(lower_bound, float):
                st.success(f"The model predicts the price is likely between **Rp {lower_bound:,.0f}** and **Rp {upper_bound:,.0f}**.")