import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import gradio as gr
import os

# Check if model files exist, if not, train the model
if not all(os.path.exists(f) for f in ["model.pkl", "scaler.pkl", "features.pkl"]):
    print("Model files not found. Training model...")
    
    # Load training and test sets
    train_df = pd.read_csv("unzipped/DIA_trainingset_RDKit_descriptors.csv")
    test_df = pd.read_csv("unzipped/DIA_testset_RDKit_descriptors.csv")
    
    # View structure
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print(train_df.head())
    
    # Drop 'SMILES' column and prepare features
    X_train = train_df.drop(columns=["Label", "SMILES"])
    y_train = train_df["Label"]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Train model
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_scaled, y_train)
    
    # Save artifacts
    joblib.dump(clf, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(X_train.columns.tolist(), "features.pkl")
    
    print("Model training completed and artifacts saved!")

# Load saved artifacts
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")  # list of descriptor names

# Select top features for manual input
top_features = features[:15]  # Using first 15 features

### --- Single Molecule Prediction ---
def predict_single(*args):
    try:
        # Convert args to numpy array and handle missing values
        input_values = []
        for i, arg in enumerate(args):
            if arg is None or arg == "":
                return f"❌ Error: Please fill in all {len(top_features)} descriptor values."
            try:
                input_values.append(float(arg))
            except (ValueError, TypeError):
                return f"❌ Error: Invalid value for {top_features[i]}. Please enter a numeric value."
        
        # Create full feature array with zeros for missing features
        full_array = np.zeros(len(features))
        
        # Fill in the top features that were provided
        for i, val in enumerate(input_values):
            feature_idx = features.index(top_features[i])
            full_array[feature_idx] = val
        
        # Reshape and scale
        arr = full_array.reshape(1, -1)
        arr_scaled = scaler.transform(arr)
        
        # Make prediction
        pred = model.predict(arr_scaled)[0]
        prob = model.predict_proba(arr_scaled)[0]
        
        # Get probability for positive class (risk)
        prob_risk = prob[1] if len(prob) > 1 else prob[0]
        
        label = "⚠️ Autoimmunity Risk" if pred == 1 else "✅ No Risk"
        return f"{label} (Confidence: {prob_risk:.2%})"
        
    except Exception as e:
        return f"❌ Error in prediction: {str(e)}"

### --- CSV Batch Upload Prediction ---
def predict_batch(file):
    try:
        if file is None:
            return "❌ Please upload a CSV file."
        
        # Read the uploaded file
        df = pd.read_csv(file.name)
        
        if df.empty:
            return "❌ The uploaded file is empty."
        
        # Check for required columns
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            return f"❌ Missing required columns: {missing_features[:10]}..." if len(missing_features) > 10 else f"❌ Missing required columns: {missing_features}"
        
        # Reorder columns to match training features
        df_ordered = df[features]
        
        # Handle missing values
        if df_ordered.isnull().any().any():
            df_ordered = df_ordered.fillna(0)  # or use df_ordered.dropna()
        
        # Scale features
        scaled = scaler.transform(df_ordered)
        
        # Make predictions
        preds = model.predict(scaled)
        probs = model.predict_proba(scaled)
        
        # Add predictions to dataframe
        result_df = df.copy()
        result_df["Autoimmunity_Risk"] = ["Yes" if p == 1 else "No" for p in preds]
        result_df["Risk_Probability"] = [f"{pr[1]:.2%}" for pr in probs]
        
        return result_df
        
    except Exception as e:
        return f"❌ Error processing file: {str(e)}"

# Create the tabbed interface
with gr.Blocks(title="Drug Autoimmunity Risk Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧬 Drug Autoimmunity Risk Predictor")
    gr.Markdown("Predict autoimmune risk for drug compounds using RDKit molecular descriptors.")
    
    with gr.Tabs():
        with gr.TabItem("🔬 Single Prediction"):
            gr.Markdown("### Enter RDKit descriptors for a single drug compound")
            gr.Markdown(f"**Note:** Using top {len(top_features)} most important descriptors. Other descriptors will be set to 0.")
            
            with gr.Row():
                with gr.Column():
                    # Create input fields directly in the interface
                    manual_inputs = []
                    for feat in top_features:
                        input_field = gr.Number(
                            label=f"{feat}",
                            placeholder="Enter numeric value",
                            value=0.0,
                            interactive=True
                        )
                        manual_inputs.append(input_field)
                
                with gr.Column():
                    single_output = gr.Textbox(label="Prediction Result", lines=3)
                    single_btn = gr.Button("🔍 Predict Risk", variant="primary")
            
            single_btn.click(
                fn=predict_single,
                inputs=manual_inputs,
                outputs=single_output
            )
            
        with gr.TabItem("📁 Batch Prediction"):
            gr.Markdown("### Upload CSV file with RDKit descriptors")
            gr.Markdown(f"**Requirements:** CSV must contain all {len(features)} RDKit descriptor columns.")
            
            file_input = gr.File(
                label="Upload CSV File",
                file_types=[".csv"],
                file_count="single"
            )
            batch_output = gr.Dataframe(label="Prediction Results")
            batch_btn = gr.Button("📊 Process Batch", variant="primary")
            
            batch_btn.click(
                fn=predict_batch,
                inputs=file_input,
                outputs=batch_output
            )
    
    # Add footer information
    gr.Markdown("---")
    gr.Markdown("**Model Information:** Random Forest Classifier trained on RDKit molecular descriptors for autoimmunity risk prediction.")

# Launch the interface
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )