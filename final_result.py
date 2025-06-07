import matplotlib.pyplot as plt


def compare_prices_with_graph(y_test, y_pred):
    plt.figure()
    plt.plot(y_test, label="real Close")
    plt.plot(y_pred, label="predicted Close")
    plt.title("Actual vs Predicted Close (Scaled)")
    plt.legend()
    plt.show()
