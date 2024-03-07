# ---------------------- ----------------------
from scipy import stats
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
import pandas as pd
from theano import shared
from sklearn import preprocessing
import pickle

if __name__ == "__main__":
    '''
    # https://towardsdatascience.com/hands-on-bayesian-statistics-with-python-pymc3-arviz-499db9a59501

    '''
    # ########### ############### Part 1 ############### ###############
    # ''' not useful line
    # # print(f'Runnning on PyMC3 v{pm.__version__}')
    # '''
    # # -------------- read data from file
    # data = pd.read_csv('data/renfe_small.csv')
    # ''' # error code
    # # data.drop('Unnamed: 0', axis=1, inplace=True)
    # '''
    # # -------------- random sample
    # data = data.sample(frac=0.01, random_state=99)
    # ''' # check percent of values in price are missing
    # # print(data.isnull().sum()/len(data))    # nearly 9.7
    # # print(data.tail())
    # '''
    # # -------------- fill missing data
    # '''
    # # Turista is the mode
    # print(data['train_class'].describe())
    # '''
    # data['train_class'] = data['train_class'].fillna(
    #     data['train_class'].mode().iloc[0])
    # data['fare'] = data['fare'].fillna(data['fare'].mode().iloc[0])
    # '''
    # # groupby
    # # transform
    # # lambda : shorten function. Eg: print((lambda x: x ** 2)(2))
    # # fillna
    # '''
    # data['price'] = data.groupby('fare').transform(
    #     lambda x: x.fillna(x.mean()))
    # # -------------- assume Gaussian dist is properly dist
    # '''
    # # kernel density estimate
    # az.plot_kde(data['price'].values, rug=True)
    # plt.yticks([0], alpha=0)
    # plt.show()
    # '''
    # # -------------- Ticket price data Model
    # with pm.Model() as model:
    #     # ------------------- priors
    #     '''
    #     # 1. population mean - Normal distribution
    #     # 2. ticket price - uniform distribution (boundaries 0: 300)
    #     # 3. std population - HalfNormal distribution
    #     '''
    #     mean = pm.Uniform('mean', lower=0, upper=300)
    #     std = pm.HalfNormal('std', sd=10)
    #     # ------------------- likelihood
    #     '''
    #     1. y ??? - normal distribution: y(mean, std)
    #     2. Draw 1000 posterior samples using NUTS sampling
    #     '''
    #     y = pm.Normal('y', mu=mean, sd=std, observed=data['price'].values)
    #     ''' # for speed up purpose
    #     # trace = pm.sample(1000, tune=1000)
    #     # with open('data/02_bayes_trace.pickle', 'wb') as f:
    #     #     pickle.dump(trace, f, protocol=pickle.HIGHEST_PROTOCOL)
    #     '''
    #     with open('data/02_bayes_trace.pickle', 'rb') as f:
    #         trace = pickle.load(f)
    #     # ------------------- plot the trace
    #     '''
    #     notice about: converge and stationary
    #     how to read trace plot
    #     any correlation??? between mean vs std???
    #     '''
    #     # az.plot_trace(trace)
    #     # az.plot_pair(trace, kind='kde', fill_last=False)
    #     # plt.show()
    #     ''' # detailed summary
    #     '''
    #     # print(az.summary(trace))
    #     ''' # mean and Highest Posterior Density (HPD) of a distribution
    #     1. HPD intervals are not the same as confidence intervals
    #     2. interpret the distribution as there is 94% probability the belief
    #      is between 60 euro and 65 euro for the mean ticket price
    #     '''
    #     # az.plot_posterior(trace)
    #     # plt.show()
    #     ''' # verify the convergence using the Gelman Rubin
    #     # 1. Values close to 1.0 mean convergence
    #     '''
    #     # print(az.rhat(trace).values())
    #     ''' # compare:
    #     # 1. bfmi (Bayesian Fraction of Missing Information)
    #     # 2. Gelman-Rubin
    #     '''
    #     # bfmi = pm.bfmi(trace)
    #     # max_gr = max(np.max(gr_stats) for gr_stats in
    #     #              az.rhat(trace).values())
    #     # (pm.energyplot(trace, legend=False, figsize=(6, 4)).set_title(
    #     #     f"BFMI = {bfmi}\nGelman-Rubin = {max_gr}"))
    #     # plt.show()
    #     ''' # Posterior Predictive Checks (PPCs)
    #     1. purpose: validate a model
    #     2. simulation results -> predictions
    #     '''
    #     # ppc = pm.sample_posterior_predictive(trace,
    #     #                                      samples=1000, model=model)
    #     # # print(np.asarray(ppc['y']).shape)
    #     # # posterior plot
    #     # _, ax = plt.subplots(figsize=(10, 5))
    #     # ax.hist([y.mean() for y in ppc['y']], bins=19, alpha=0.5)
    #     # ax.axvline(data.price.mean())
    #     # ax.set(title='Posterior predictive of the mean',
    #     #        xlabel='mean(x)', ylabel='Frequency')
    #     # plt.show()

    # ########### ############### Part 2 ############### ###############
    # # -------------- Group Comparison
    # '''
    # 1. estimating effect size between price vs fare types
    # 2. quantifying the difference between two fare categories using mean
    #  of each fare type
    # 3. obtain a posterior distribution of the differences of means
    #   between fare categories
    # '''
    # data = pd.read_csv('data/renfe_small.csv')
    # data['train_class'] = data['train_class'].fillna(
    #     data['train_class'].mode().iloc[0])
    # data['fare'] = data['fare'].fillna(data['fare'].mode().iloc[0])
    # data['price'] = data.groupby('fare').transform(
    #     lambda x: x.fillna(x.mean()))
    # # ''' # error code
    # # data.drop('Unnamed: 0', axis=1, inplace=True)
    # price = data['price'].values
    # idx = pd.Categorical(data['fare'], categories=[
    #                      'Flexible', 'Promo', 'Promo +',
    #                      'Adulto ida', 'Mesa',
    #                      'Individual-Flexible']).codes

    # groups = len(np.unique(idx))
    # with pm.Model() as comparing_groups:
    #     μ = pm.Normal('μ', mu=0, sd=10, shape=groups)
    #     σ = pm.HalfNormal('σ', sd=10, shape=groups)
    #     y = pm.Normal('y', mu=μ[idx], sd=σ[idx], observed=price)
    #     '''
    #     # sampling
    #     trace_groups = pm.sample(5000, tune=5000, return_inferencedata=False)
    #     with open('data/02_bayes_trace_groups.pickle', 'wb') as f:
    #         pickle.dump(trace_groups, f, protocol=pickle.HIGHEST_PROTOCOL)
    #     '''
    #     with open('data/02_bayes_trace_groups.pickle', 'rb') as f:
    #         trace_groups = pickle.load(f)
    #     # create a summary table instead of trace plotting
    #     flat_fares = az.from_pymc3(trace=trace_groups)
    #     fares_gaussian = az.summary(flat_fares)
    #     '''
    #     fares_gaussian.to_csv('data/fares_gaussian.csv')
    #     # ------------------- get fares_gaussian from file
    #     fares_gaussian = pd.read_csv('data/fares_gaussian.csv')
    #     '''
    #     # -----------------------------------------
    #     '''
    #     # significant differences between groups on the mean
    #     # drop
    #     fares_gaussian.drop('Unnamed: 0', axis=1, inplace=True)
    #     # or set index
    #     fares_gaussian.set_index('Unnamed: 0', inplace=True)
    #     '''
    #     # print(fares_gaussian)
    #     '''
    #     1. Cohen’s d comparison
    #     2. probability of superiority (xác suất vượt trội)
    #     '''
    #     # -------------------------------------------------------
    #     dist = stats.norm()
    #     _, ax = plt.subplots(5, 2, figsize=(15, 9), constrained_layout=True)
    #     comparisons = [(i, j) for i in range(6) for j in range(i+1, 6)]
    #     pos = [(k, v) for k in range(5) for v in (0, 1)]
    #     for (i, j), (k, v) in zip(comparisons, pos):
    #         means_diff = trace_groups['μ'][:, i] - trace_groups['μ'][:, j]
    #         d_cohen = (means_diff / np.sqrt((trace_groups['σ'][:, i]**2
    #                                          + trace_groups['σ'][:, j]**2) /
    #                                         2)).mean()
    #         ps = dist.cdf(d_cohen/(2**0.5))
    #         az.plot_posterior(means_diff, ref_val=0, ax=ax[k, v])
    #         ax[k, v].set_title(f'μ{i}-μ{j}')
    #         ax[k, v].plot(
    #             0, label=f"Cohen= {d_cohen:.2f}\nProb= {ps:.2f}", alpha=0)
    #         ax[k, v].legend()
    #     plt.show()
    #     # -------------------------------------------------------
    # ########### ############### Part 3 ############### ###############
    # # Bayesian Hierarchical Linear Regression
    # # ----------- Encoding the categorical variable
    # def replace_fare(fare):

    #     if fare == 'Adulto ida':
    #         return 1
    #     elif fare == 'Promo +':
    #         return 2
    #     elif fare == 'Promo':
    #         return 3
    #     elif fare == 'Flexible':
    #         return 4
    #     elif fare == 'Individual-Flexible':
    #         return 5
    #     elif fare == 'Mesa':
    #         return 6

    # data = pd.read_csv('data/renfe_small.csv')
    # data['train_class'] = data['train_class'].fillna(
    #     data['train_class'].mode().iloc[0])
    # data['fare'] = data['fare'].fillna(data['fare'].mode().iloc[0])
    # data['price'] = data.groupby('fare').transform(
    #     lambda x: x.fillna(x.mean()))
    # data['fare_encode'] = data['fare'].apply(lambda x: replace_fare(x))
    # label_encoder = preprocessing.LabelEncoder()
    # data['train_type_encode'] = label_encoder.fit_transform(data['train_type'])
    # train_type_names = data.train_type.unique()
    # train_type_idx = data.train_type_encode.values
    # n_train_types = len(data.train_type.unique())
    # # print(data[['train_type', 'price', 'fare_encode']].tail())
    # with pm.Model() as hierarchical_model:
    #     # global model parameters
    #     α_μ_tmp = pm.Normal('α_μ_tmp', mu=0., sd=100)
    #     α_σ_tmp = pm.HalfNormal('α_σ_tmp', 5.)
    #     β_μ = pm.Normal('β_μ', mu=0., sd=100)
    #     β_σ = pm.HalfNormal('β_σ', 5.)
    #     # train type specific model parameters
    #     α_tmp = pm.Normal('α_tmp', mu=α_μ_tmp, sd=α_σ_tmp, shape=n_train_types)
    #     # Intercept for each train type, distributed around train type mean
    #     β = pm.Normal('β', mu=β_μ, sd=β_σ, shape=n_train_types)
    #     # Model error
    #     eps = pm.HalfCauchy('eps', 5.)
    #     # Data estimate
    #     fare_est = α_tmp[train_type_idx] + \
    #         β[train_type_idx]*data.fare_encode.values
    #     # Data likelihood
    #     fare_like = pm.Normal('fare_like', mu=fare_est,
    #                           sd=eps, observed=data.price)
    # with hierarchical_model:
    #     '''
    #     hierarchical_trace = pm.sample(
    #         2000, tune=2000, return_inferencedata=False, target_accept=.9)
    #     with open('data/02_hierarchical_trace.pickle', 'wb') as f:
    #         pickle.dump(hierarchical_trace, f,
    #                     protocol=pickle.HIGHEST_PROTOCOL)
    #     '''
    #     with open('data/02_hierarchical_trace.pickle', 'rb') as f:
    #         hierarchical_trace = pickle.load(f)
    # '''
    # # marginal posteriors in the left column are highly informative
    # '''
    # # pm.traceplot(hierarchical_trace, var_names=[
    # #              'α_μ_tmp', 'β_μ', 'α_σ_tmp', 'β_σ', 'eps'])
    # '''
    # # look at how 5 train types compare in terms of the ticket price
    # '''
    # # pm.traceplot(hierarchical_trace, var_names=[
    # #              'α_tmp'], coords={'α_tmp_dim_0': range(5)})
    # '''
    # # Bayesian credible interval for the price of different train types
    # '''
    # # az.plot_forest(hierarchical_trace, var_names=['α_tmp', 'β'],
    # #                combined=True)
    # # plt.show()
    # '''
    # # compute r squared
    # # '''
    # # ppc = pm.sample_posterior_predictive(
    # #     hierarchical_trace, samples=2000, model=hierarchical_model)
    # # print(az.r2_score(data.price.values, ppc['fare_like']))

    # ########### ############### Part 4 ############### ###############
    ''' # animal observations
    # https://towardsdatascience.com/estimating-probabilities-with-bayesian-modeling-in-python-7144be007815
    '''
    # ---------- Some concepts ----------
    '''
    -------- model of the problem ----------
    # 1. independent trial, initial (prior) belief
    # 2. discrete choices and multinomial distribution
    # 3. Probability Mass Function, Dirichlet Distribution
    # 4. Hyperparameters, "vector" of probabilities, parameter of the prior
    # 5. hyperprior, hyperyhyperparameters
    # 6. Dirichlet parameter vector: pseudocounts capture prior 'belief'
    example is alpha = [1, 1, 1] -> prevalence of each animal is the same
    # 7. posterior distribution / observations(data) / 'hyperparameters' and
    # multinomial

    -------- Inference process ----------
    # 1. estimating unknown parameters: 'prevalence of each species'
    # 2. data: 'single set of observations' -> probability distribution
    # 3. find posterior distribution: probability of seeing each species
    # 4. MCMC to approximate the posterior
    '''
    # # -------- Code ----------
    # species = ['lions', 'tigers', 'bears']
    # # Observations
    # c = np.array([3, 2, 1])
    # # Pseudocounts/ hyperparameters
    # '''
    # more confident in our belief -> increase hyperparameters
    # more weight data -> reduce pseudocounts/ hyperparameters
    # # [0.1, 0.1, 0.1] or [15, 15, 15]
    # '''
    # alphas = np.array([1, 1, 1])
    # # print(species, c, alphas)
    # expected = (alphas+c)/(c.sum()+alphas.sum())
    # # print(expected)

    # # ---------- ---------- PyMC3 ---------- ----------
    # alphas = np.array([1, 1, 1])
    # c = np.array([3, 2, 1])
    # # --------------- Create model
    # with pm.Model() as model:
    #     # Parameters of the Multinomial from Dirichlet
    #     params = pm.Dirichlet('params', a=alphas, shape=alphas.shape[0])
    #     # Observed data from Multinomial dist
    #     observed_data = pm.Multinomial(
    #         'observed_data', n=6, p=params, shape=c.shape[0], observed=c)
    # # --------------- sample from the posterior
    # with model:
    #     # sampling 1 -----------
    #     '''
    #     trace = pm.sample(draws=1000, chains=2, tune=500,
    #                       discard_tuned_samples=True)
    #     with open('data/02_dirichlet.pickle', 'wb') as f:
    #         pickle.dump(trace, f, protocol=pickle.HIGHEST_PROTOCOL)
    #     '''
    #     with open('data/02_dirichlet.pickle', 'rb') as f:
    #         trace = pickle.load(f)
    #     '''
    #     left of graph: contains KDE (PDF of the event probabilities)
    #     right of graph: samples drawn
    #     '''
    #     # pm.traceplot(trace)
    #     '''
    #     1. histograms: number of times each probability was
    #     sampled from the posterior
    #     2. point estimate???
    #     '''
    #     # az.plot_posterior(trace)
    #     # plt.show()
    #     '''
    #     # quantify the level of uncertainty
    #     '''
    #     # print(az.summary(trace))

    #     # sampling 2 -----------
    #     '''
    #     samples = pm.sample_posterior_predictive(trace, samples=1000)
    #     with open('data/02_dirichlet_ppc.pickle', 'wb') as f:
    #         pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    #     '''
    #     with open('data/02_dirichlet_ppc.pickle', 'rb') as f:
    #         samples = pickle.load(f)

#     # -------------------------------------------------------
'''
# https://statsthinking21.github.io/statsthinking21-python/10-BayesianStatistics.html

'''
#     # -------------------------------------------------------
'''
# https://learn.datacamp.com/courses/bayesian-data-analysis-in-python
'''
#     # -------------------------------------------------------
'''
# https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers
'''
#     # -------------------------------------------------------
'''
# mathematical modelling – Bayesian:
# https://www.youtube.com/watch?v=HZGCoVF3YvM

'''
