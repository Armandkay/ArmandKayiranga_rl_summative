def render_transaction(transaction, action):
    """
    Prints transaction info and model prediction.
    """
    label = int(transaction[-1])
    action_str = "Fraud" if action == 1 else "Legit"
    correct = "✅" if action == label else "❌"
    print(f"Transaction: {transaction[:-1]}, Predicted: {action_str}, True: {label} {correct}")
