
import ollama

print("🤖 My Simple AI Chatbot")
print("Type 'exit' to stop chatting.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("AI: Goodbye! Have a nice day.")
        break

    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )

    print("AI:", response["message"]["content"])
    print()
