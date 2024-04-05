import gradio as gr
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the XGBoost model
xgboost_model = joblib.load('xgboost_model.pkl')

# Load the TF-IDF vectorizer
(vectorizer, _) = joblib.load('tfidf_xgboost.pkl')

# # Function to make predictions
# def predict_label(input_text):
#     # Preprocess the input text
#     processed_input_text = input_text

#     # Convert the processed input text to TF-IDF vector
#     input_vector = vectorizer.transform([processed_input_text])

#     # Use the XGBoost model to make predictions
#     predicted_label = xgboost_model.predict(input_vector)[0]
#     if predicted_label == 0:
#         return 'skill/competence'
#     elif predicted_label == 1:
#         return 'knowledge'

# # Create a Gradio interface
# iface = gr.Interface(
#     fn=predict_label,
#     inputs=gr.components.Textbox(lines=5, label="Enter the text:"),
#     outputs=gr.components.Label(),
#     title="ESCO Classifier"
# )

# # Launch the Gradio interface
# iface.launch()


# Function to make predictions
def predict_label(input_text):
    # Preprocess the input text
    processed_input_text = input_text

    # Convert the processed input text to TF-IDF vector
    input_vector = vectorizer.transform([processed_input_text])

    # Use the XGBoost model to make predictions
    predicted_probs = xgboost_model.predict_proba(input_vector)[0]
    predicted_label = xgboost_model.predict(input_vector)[0]

    if predicted_label == 0:
        label = 'skill/competence'
    else:
        label = 'knowledge'

    skill_competence_prob = predicted_probs[0] * 100
    knowledge_prob = predicted_probs[1] * 100

    return label, skill_competence_prob, knowledge_prob


# Create a Gradio interface
iface = gr.Interface(
    fn=predict_label,
    inputs=gr.components.Textbox(lines=5, label="Enter the text:"),
    outputs=[
        gr.components.Label(label="Predicted Label"),
        gr.components.Label(label="Skill/Competence Probability (%)"),
        gr.components.Label(label="Knowledge Probability (%)"),
    ],
    title="ESCO Classifier"
)

# Launch the Gradio interface
iface.launch(share=True)