import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pymc3


if __name__ == '__main__':
    plt.style.use("ggplot")
    n = 50
    z = 10

    # what are these params
    alpha = 12
    beta = 12
    alpha_post = 22
    beta_post = 52

    # iterations = 100000
    iterations = 100
    basic_model = pymc3.Model()
    with basic_model:
        # Define "conjugate prior" belief about the fairness
        theta = pymc3.Beta("theta", alpha=alpha, beta=beta)

        # Define the Bernoulli likelihood function
        y = pymc3.Binomial("y", n=n, p=theta, observed=z)

        # Use Maximum A Posteriori optimisation as initial value for MCMC
        start = pymc3.find_MAP()    # ???

        # Use the Metropolis algorithm
        step = pymc3.Metropolis()

        # Calculate the trace
        trace = pymc3.sample(iterations, step, start,
                             random_seed=1, progressbar=True)

        # Plot the posterior histogram from MCMC analysis
        plt.hist(trace["theta"], bins=50, histtype="step",
                 density=True, label="Posterior (MCMC)", color="red")

        # Plot the analytic prior and posterior beta distributions
        x = np.linspace(0, 1, 100)

        plt.plot(x, stats.beta.pdf(x, alpha, beta),
                 "--", label="Prior", color="blue")

        plt.plot(x, stats.beta.pdf(x, alpha_post, beta_post),
                 label='Posterior(Analytic)', color="green")

        # Update the graph labels
        plt.legend(title="Parameters", loc="best")
        plt.xlabel("$\\theta$, Fairness")
        plt.ylabel("Density")
        # traceplot method to plot kernel density estimate (KDE) and trace
        pymc3.traceplot(trace)
        plt.show()
