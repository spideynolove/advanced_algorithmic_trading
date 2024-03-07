import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from random import randint


if __name__ == "__main__":
    # Create a list of the number of coin tosses ("Bernoulli trials")
    # numbers = (randint(0, 500) for _ in range(6))
    numbers = [0, 2, 10, 20, 50, 500]   # trials

    # Random variates: "prior" | fairness
    data = stats.bernoulli.rvs(0.5, size=numbers[-1])

    # make x-axis 100 separate plotting points
    x = np.linspace(0, 1, 100)
    for i, N in enumerate(numbers):
        heads = data[:N].sum()

        # Create an axes subplot
        ax = plt.subplot(len(numbers) / 2, 2, i + 1)
        ax.set_title(f"{N} trials, {heads} heads")

        # Add labels to both axes
        plt.xlabel("$P(H)$, Probability of Heads")
        plt.ylabel("Density")
        if i == 0:
            plt.ylim([0.0, 2.0])
        plt.setp(ax.get_yticklabels(), visible=False)

        # Beta distribution "posterior"
        y = stats.beta.pdf(x, 1 + heads, 1 + N - heads)
        plt.plot(x, y, label=f"observe {N} tosses,\n {heads} heads")
        plt.fill_between(x, 0, y, color="#aaaadd", alpha=0.5)

    # show plot
    plt.tight_layout()
    plt.show()
