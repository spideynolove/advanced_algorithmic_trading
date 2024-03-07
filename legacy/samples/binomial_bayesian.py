
# --------------- beta distribution ploting ------------------
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    # ???
    sns.set_palette("deep", desat=.6)
    sns.set_context(rc={"figure.figsize": (8, 4)})
    # len = 100
    x = np.linspace(0, 1, 100)

    # P(o|a, b) = (o^(a-1))((1 - o)^(b-1))/B(a, b)
    # larger a -> probability distribution move to 1, heads more often
    # larger b -> probability distribution move to 0, tails more often
    # both a and b increase -> narrow distribution
    # distribution peak when a and b increase equally

    params = [(0.5, 0.5), (1, 1), (4, 3), (2, 5), (6, 6)]
    for p in params:
        y = beta.pdf(x, p[0], p[1])
        plt.plot(x, y, label=f"$\\alpha={p}$, $\\beta={p}$")
    plt.xlabel("$\\theta$, Fairness")
    plt.ylabel("Density")
    plt.legend(title="Parameters")
    plt.show()

# -------------------------------------------------------------------------
# prior. posterior. likelihood. evidence.
# https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
