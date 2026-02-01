from spam_classifier import train_and_evaluate, predict_email

# Train and evaluate model
model, vectorizer = train_and_evaluate("data/spam.csv")

# Test prediction
sample_email = "Congratulations! You have won a free lottery ticket. Claim now."
print("\nSample Email:", sample_email)
print("Prediction:", predict_email(sample_email, model, vectorizer))