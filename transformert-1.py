from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love Hugging Face! This is amazing.")
print(result)
