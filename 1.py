# Step 1: Dataset Creation (Assuming data is already created)

# Step 2: Model Training
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV

# Example dataset
X_train = ["Hi", "How are you?", "Hello", "Goodbye", "See you later", "Take care", "What's the weather like today?", "Can you tell me the time?", "Where is the nearest restaurant?"]
y_train = ["Greet", "Greet", "Greet", "Farewell", "Farewell", "Farewell", "Inquiry", "Inquiry", "Inquiry"]

# Text preprocessing
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)

# Model training
classifier = SVC(probability=True)  # Support Vector Machine classifier
calibrated_classifier = CalibratedClassifierCV(classifier)  # Calibrated classifier for probability estimation
calibrated_classifier.fit(X_train_vectors, y_train)

# Step 3: Intent Classification Function
def classify_intent(text):
    # Preprocess text
    text_vector = vectorizer.transform([text])
    
    # Predict probabilities
    confidence = calibrated_classifier.predict_proba(text_vector).max()
    predicted_intent = calibrated_classifier.predict(text_vector)[0]
    
    return predicted_intent, confidence

# Step 4: Fallback Mechanism
CONFIDENCE_THRESHOLD = 0.7
FALLBACK_RESPONSE = "NLU fallback: Intent could not be confidently determined"

def classify_intent_with_fallback(text):
    intent, confidence = classify_intent(text)
    
    if confidence >= CONFIDENCE_THRESHOLD:
        return intent, confidence
    else:
        return FALLBACK_RESPONSE, confidence

# Step 5: Testing (Assuming random test inputs)
test_inputs = ["Hi there!", "How's the weather?", "Good afternoon", "Where can I find a good restaurant?"]

for input_text in test_inputs:
    intent, confidence = classify_intent_with_fallback(input_text)
    print(f"Input: '{input_text}', Predicted Intent: '{intent}', Confidence: {confidence}")

