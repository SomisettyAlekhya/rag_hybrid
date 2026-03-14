from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="gpt2"
)

def generate_answer(query, context):

    prompt = f"""
Context: {context}
Question: {query}
Answer:
"""

    result = generator(
        prompt,
        max_new_tokens=40,
        do_sample=False
    )

    text = result[0]["generated_text"]

    # Remove prompt from output
    answer = text.replace(prompt, "").strip()

    # Remove unwanted words
    answer = answer.replace("Context:", "")
    answer = answer.replace("Question:", "")
    answer = answer.replace("Answer:", "")

    # Keep only first sentence
    answer = answer.split(".")[0] + "."

    return answer