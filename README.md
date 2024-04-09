# ESCO Text Classification: Skill/Competence or Knowledge

This is a simple Gradio app that accepts a single text input and predicts whether the input text is considered a skill/competence or knowledge. The app uses a machine learning model trained on the ESCO Skills database to make the classification.

## Usage

1. Clone the repository or download the source code.
2. Install the required dependencies by running `pip3 install -r requirements.txt`.
3. Run the app by executing `python3 main.py` in your terminal.
4. The Gradio app will open in your default web browser.
5. Enter your text input in the provided text box and click the "Submit" button.
6. The app will display the predicted classification: either "Skill/Competence" or "Knowledge".

## Model Details

The machine learning model used in this app is an XGBoost classifier. It was trained on the ESCO Skills database, which provides a comprehensive list of skills, competencies, and knowledge descriptions.

The ESCO Skills database was preprocessed, and the text data was transformed into numerical features suitable for training the XGBoost model. The model was then trained to learn the patterns and characteristics that distinguish skills/competencies from knowledge descriptions.

## Dependencies

The app requires the following Python libraries:

- Gradio
- XGBoost
- Pandas
- Scikit-learn

You can install the required dependencies by running `pip3 install -r requirements.txt`.
