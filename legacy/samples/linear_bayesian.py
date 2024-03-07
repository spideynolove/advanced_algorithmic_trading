# 1. frequentist (classical) approach to multiple linear regression
# best linear fit to the data

# noisy linear data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pymc3 as pm
import pymc3.plots.posteriorplot as pp

# print(f"Running on PyMC3 v{pm.__version__}")

sns.set(style='darkgrid', palette='muted')


def simulate_ln_data(N, beta_0, beta_1, eps_sigma_sq):
    # print(list(map(lambda x: float(x)/100.0, np.arange(N))))
    df = pd.DataFrame(
        # choice using list not map so need to convert
        {"x": np.random.RandomState(42).choice(
            list(map(lambda x: float(x)/100.0, np.arange(N))),
            N, replace=False)}
    )
    eps_mean = 0.0
    df['y'] = beta_0 + beta_1*df['x'] + \
        np.random.RandomState(42).normal(eps_mean, eps_sigma_sq, N)
    # print(df)
    return df


def glm_mcmc_inference(df, iteration=5000):
    basic_model = pm.Model()
    with basic_model:
        # use a Normal distribution for the likelihood
        pm.glm.GLM.from_formula("y ~ x", df, family=pm.glm.families.Normal())

        # initial value for MCMC, use Maximum A Posteriori (MAP) optimisation
        start = pm.find_MAP()

        # Use the No-U-Turn Sampler
        step = pm.NUTS()
        # step = pm.Metropolis()

        # Calculate the trace
        trace = pm.sample(iteration, step, start,
                          random_seed=42, progressbar=True)
        return trace


if __name__ == "__main__":
    beta_0 = 1.0
    beta_1 = 2.0
    # Simulate 100 data points, with a variance of 0.5
    N = 100
    eps_sigma_sq = 0.5

    # Simulate the "linear" data
    df = simulate_ln_data(N, beta_0, beta_1, eps_sigma_sq)

    # Plot the data, and a frequentist linear regression
    sns.lmplot(x='x', y='y', data=df, height=6)
    plt.xlim(0.0, 1.0)

    trace = glm_mcmc_inference(df, iteration=5000)
    pm.traceplot(trace[500:])
    plt.show()

    # Plot a sample of posterior regression lines
    sns.lmplot(x='x', y='y', data=df, height=6, fit_reg=False)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 4.0)

    # now change to this: https://docs.pymc.io/notebooks/GLM-linear.html
    pp.plot_posterior_predictive_glm(trace, samples=100)

    x = np.linspace(0, 1, N)
    y = beta_0 + beta_1*x
    plt.plot(x, y, label='True Regression Line', lw=3, c='green')
    plt.legend(loc=0)
    plt.show()
