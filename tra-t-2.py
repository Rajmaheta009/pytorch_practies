from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
text = generator("raj maheta is,", max_length=40, num_return_sequences=1)
print(text[0]["generated_text"])
