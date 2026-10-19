# import joblib
# import pandas as pd

# MODEL_PATH = "outputs/final_model.pkl"

# artifact = joblib.load(MODEL_PATH)

# preprocessor = artifact["preprocessor"]
# encoder = artifact["target_encoder"]
# selected_indices = artifact["selected_indices"]
# model = artifact["model"]

# def predict_student(student_dict):
#     df = pd.DataFrame([student_dict])

#     X = preprocessor.transform(df)
#     X = X[:, selected_indices]

#     pred = model.predict(X)[0]
#     label = encoder.inverse_transform([pred])[0]

#     result = {
#         "predicted_class": str(label)
#     }

#     if hasattr(model, "predict_proba"):
#         probs = model.predict_proba(X)[0]
#         result["confidence"] = float(max(probs))

#     return result


# # Example:
# # Change the keys to match your dataset columns.
# #
# # student = {
# #     "Attendance": 85,
# #     "StudyHours": 5,
# #     "PreviousScore": 72
# # }
# #
# # print(predict_student(student))


import joblib
import pandas as pd

MODEL_PATH = "outputs/final_model.pkl"

# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------
artifact = joblib.load(MODEL_PATH)

preprocessor = artifact["preprocessor"]
encoder = artifact["target_encoder"]
model = artifact["model"]

# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------
def predict_student(student_dict):

    # Convert input dictionary to DataFrame
    df = pd.DataFrame([student_dict])

    # Preprocess input
    X = preprocessor.transform(df)

    # IMPORTANT:
    # The current best model is Baseline Random Forest.
    # It was trained using all 17 preprocessed features.
    # Therefore, DO NOT apply selected_indices here.
    
    pred = model.predict(X)[0]

    # Convert numerical prediction back to class label
    label = encoder.inverse_transform([pred])[0]

    result = {
        "predicted_class": str(label)
    }

    # Get prediction confidence
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        result["confidence"] = float(max(probabilities))

    return result


# ---------------------------------------------------------
# Example prediction
# ---------------------------------------------------------
if __name__ == "__main__":

    student = {
        "gender": "female",
        "race/ethnicity": "group C",
        "parental level of education": "bachelor's degree",
        "lunch": "standard",
        "test preparation course": "completed"
    }

    result = predict_student(student)

    print("\n===== STUDENT PERFORMANCE PREDICTION =====")
    print("Predicted Performance:", result["predicted_class"])
    print("Confidence:", f"{result['confidence']:.2%}")