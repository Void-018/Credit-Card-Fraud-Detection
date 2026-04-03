import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and scaler
@st.cache_resource
def load_model_scaler():
    try:
        scaler = joblib.load('scaler.pkl')
        model = joblib.load('model.pkl')
        st.success("Model and scaler loaded!")
        return scaler, model
    except FileNotFoundError:
        st.error("Model or scaler not found! Run: python train_model.py")
        return None, None

scaler, model = load_model_scaler()
if scaler is None or model is None:
    st.stop()

st.title('🕵️ Credit Card Fraud Detection')
st.info('This app predicts whether a transaction is fraudulent using ML anomaly detection (IsolationForest).')

# Option 1: File upload
uploaded_file = st.file_uploader('Choose CSV file (creditcard format)', type='csv')
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write('Raw shape:', df.shape)
        
        # Auto-fix: Drop Class if present, ensure 30 feats
        if 'Class' in df.columns:
            df = df.drop('Class', axis=1)
            st.info('Dropped Class column.')
        
        # Expected columns
        expected_cols = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']
        
        if set(df.columns) & set(expected_cols) == set(expected_cols[:len(df.columns)]):
            # Reorder to standard
            df = df.reindex(columns=expected_cols[:df.shape[1]], fill_value=0)
        elif df.shape[1] > 30:
            df = df.iloc[:, :30]
            st.warning('Truncated to first 30 columns.')
        elif df.shape[1] < 30:
            # Pad missing cols with 0
            for i in range(df.shape[1], 30):
                df[f'feat_{i}'] = 0
        
        if df.shape[1] != 30:
            st.error(f'Need exactly 30 numeric columns after fix. Got {df.shape[1]}.')
            st.stop()
        
        # Validate numeric, no NaN
        df = df.select_dtypes(include=[np.number]).fillna(0)
        if df.isna().any().any():
            st.error('NaNs found after fill.')
            st.stop()
        
        st.success(f'Processed to {df.shape} - ready!')
        
        scaled = scaler.transform(df)
        preds_raw = model.predict(scaled)
        preds = np.where(preds_raw == -1, 1, 0)
        decision_funcs = model.decision_function(scaled)
        probs = np.clip((-decision_funcs + 0.5), 0, 1)
        df = df.copy()
        df['fraud_pred'] = preds
        df['fraud_prob'] = probs
        
        # Summary
        total = len(df)
        fraud_count = preds.sum()
        fraud_pct = fraud_count / total * 100
        col1, col2, col3 = st.columns(3)
        col1.metric('Total', total)
        col2.metric('Frauds', fraud_count)
        col3.metric('Fraud %', f'{fraud_pct:.2f}%')
        
        st.dataframe(df.head(100))
        st.download_button('Download predictions CSV', df.to_csv(index=False), 'predictions.csv')
        
        # Plot
        import altair as alt
        plot_data = df['fraud_pred'].value_counts().reset_index()
        plot_data.columns = ['pred', 'count']
        st.altair_chart(alt.Chart(plot_data).mark_bar().encode(x='pred:N', y='count:Q', color='pred:N'), use_container_width=True)
        
    except Exception as e:
        st.error(f'Upload error: {str(e)}')

# Option 2: Manual input (demo first row)
st.subheader('Manual Demo')
if 'data' not in st.session_state:
    st.session_state.data = [0.0] * 30
data = st.session_state.data.copy()

mode = st.radio('Mode:', ['Basic', 'Advanced'])

col1, col2 = st.columns(2)
if col1.button('Non-Fraud Demo'):
    st.session_state.data = [0.0] * 30
    st.rerun()
if col2.button('Fraud Demo'):
    np.random.seed(42)
    st.session_state.data = list(np.random.normal(3, 2, 30))
    st.rerun()

feature_names = [f'V{i}' for i in range(1,29)] + ['Amount', 'Time']
feat_desc = {
    'V1': 'Aggregated distance signals',
    'V2': 'Anonymized PCA2',
    'V3': 'PCA3',
    'V4': 'PCA4',
    'Amount': 'Transaction Amount',
    'Time': 'Elapsed Time'
}

n_sliders = 4 if mode == 'Basic' else 10
st.write(f'Adjust first {n_sliders} features:')
for i in range(n_sliders):
    nm = feature_names[i]
    desc = feat_desc.get(nm, 'PCA feature')
    data[i] = st.slider(f'{nm} ({desc})', -5.0, 5.0, data[i], 0.1, help=desc)
st.session_state.data = data

if st.button('Predict', type='primary'):
    # Validate
    try:
        data_array = np.array(data)
        if not np.issubdtype(data_array.dtype, np.number):
            st.error('All features must be numeric.')
            st.stop()
        input_array = data_array.reshape(1, -1)
        scaled_input = scaler.transform(input_array)
    except:
        st.error('Invalid input data.')
        st.stop()
    
    pred_raw = model.predict(scaled_input)[0]
    pred = 1 if pred_raw == -1 else 0
    decision_func = model.decision_function(scaled_input)[0]
    prob = float(np.clip((-decision_func + 0.5), 0, 1))
    
    risk_level = 'High' if pred == 1 else 'Low'
    st.metric('Risk Level', risk_level + ' Risk 🚨' if pred == 1 else ' Risk ✅', delta=None)
    st.metric('Fraud Probability', f'{prob:.1%}')
    
    st.progress(prob)
    if pred == 1:
        st.error('🚨 High Risk: Fraud likely!')
    else:
        st.success('✅ Low Risk: Safe.')


