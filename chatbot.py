import aiml
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("sales-dataset.csv")
print("Available columns:", data.columns.tolist())  # Debug
data.columns = data.columns.str.strip()
print(data.head())  # Show first 5 rows

# Initialize AIML kernel
# Initialize AIML kernel
kernel = aiml.Kernel()
kernel.learn("chatbot.aiml")   # ✅ Correct path

print("Chatbot is ready! Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    # Get response from AIML
    response = kernel.respond(user_input.upper())
    print("Bot:", response)

    # Check which query AIML set
    query_type = kernel.getPredicate("query")

    if query_type == "total_revenue":
        total = data['Revenue'].sum()
        print(f"Total Revenue: ${total}")
        kernel.setPredicate("query", "")  # Reset query

    elif query_type == "best_selling":
        best_product = data.groupby('Product')['Units_Sold'].sum().idxmax()
        print(f"Best Selling Product: {best_product}")
        kernel.setPredicate("query", "")

    elif query_type == "sales_trend":
        trend = data.groupby('Date')['Revenue'].sum()
        trend.plot(kind='line', marker='o')
        plt.title("Revenue Trend Over Time")
        plt.xlabel("Date")
        plt.ylabel("Revenue")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        kernel.setPredicate("query", "")
    
  