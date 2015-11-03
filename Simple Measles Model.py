
# coding: utf-8

# # Disease Outbreak Response Decision-making Under Uncertainty: A retrospective analysis of measles in Sao Paulo


get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import numpy.ma as ma
from datetime import datetime
import matplotlib.pyplot as plt
import pdb

from IPython.core.display import HTML
def css_styling():
    styles = open("styles/custom.css", "r").read()
    return HTML(styles)
css_styling()



data_dir = "data/"



get_ipython().system('rm -rf ~/.theano')


# Import outbreak data


measles_data = pd.read_csv(data_dir+"measles.csv", index_col=0)
measles_data.NOTIFICATION = pd.to_datetime(measles_data.NOTIFICATION)
measles_data.BIRTH = pd.to_datetime(measles_data.BIRTH)
measles_data.ONSET = pd.to_datetime(measles_data.ONSET)



measles_data = measles_data.replace({'DISTRICT': {'BRASILANDIA':'BRAZILANDIA'}})


# Sao Paulo population by district


sp_pop = pd.read_csv(data_dir+'sp_pop.csv', index_col=0)



_names = sp_pop.index.values
_names[_names=='BRASILANDIA'] = 'BRAZILANDIA'
sp_pop.set_index(_names, inplace = True)



sp_pop.head()


# Annual vaccination data


vaccination_data = pd.read_csv('data/BrazilVaxRecords.csv', index_col=0)
vaccination_data.head()



vaccination_data.VAX[:18]



vax_97 = np.r_[[0]*(1979-1921+1), vaccination_data.VAX[:17]]
n = len(vax_97)
FOI_mat = np.resize((1 - vax_97*0.9), (n,n)).T



# Mean age of infection for those born prior to vaccination coverage, assuming R0=16
A = 4.375



(1 - vax_97*0.9)[:-1]



np.tril(FOI_mat).sum(0)



natural_susc = np.exp((-1/A) * np.tril(FOI_mat).sum(0))[::-1]
vacc_susc = (1 - vax_97*0.9)[::-1]
vacc_susc[0] = 0.5
vacc_susc



sia_susc = np.ones(len(vax_97))
birth_year = np.arange(1922, 1998)[::-1]
by_mask = (birth_year > 1983) & (birth_year < 1992)
sia_susc[by_mask] *= 0.2



total_susc = sia_susc * vacc_susc * natural_susc
total_susc



pd.Series(total_susc).plot();


# Plot of cumulative cases by district


measles_onset_dist = measles_data.groupby(['DISTRICT','ONSET']).size().unstack(level=0).fillna(0)
measles_onset_dist.cumsum().plot(legend=False, grid=False)



total_district_cases = measles_onset_dist.sum()


# Top 5 districts by number of cases


totals = measles_onset_dist.sum()
totals.sort(ascending=False)
totals[:5]


# Age distribution of cases, by confirmation status


by_conclusion = measles_data.groupby(["YEAR_AGE", "CONCLUSION"])
counts_by_cause = by_conclusion.size().unstack().fillna(0)
ax = counts_by_cause.plot(kind='bar', stacked=True, xlim=(0,50), figsize=(15,5))


# ## Stochastic Disease Transmission Model
# 
# As a baseline for comparison, we can fit a model to all the clinically-confirmed cases, regardless of lab confirmation status. For this, we will use a simple SIR disease model, which will be fit using MCMC.
# 
# This model fits the series of 2-week infection totals in each district $i$ as a set of Poisson models:
# 
# \\[Pr(I(t)_{i} | \lambda(t)_i) = \text{Poisson}(\lambda(t)_i) \\]
# 
# Where the outbreak intensity is modeled as:
# 
# \\[\lambda(t)_i = \beta [I^{(w)}(t-1)_i]^{\alpha} S(t-1)_i\\]
# 
# \\[\alpha \sim \text{Exp}(1)\\]
# 
# We will assume here that the transmission rate is constant over time (and across districts):
# 
# \\[\beta \sim \text{Gamma}(1, 0.1)\\]
# 
# To account for the influence of infected individuals from neighboring districts on new infections, the outbreak intensity was modeled using a spatial-weighted average of infecteds across districts, where populations were weighted as an exponential function of the distance between district centroids:
# 
# \\[w_{d} = \text{exp}(-\theta d)\\]
# 
# \\[\theta \sim \text{Exp}(1)\\]
# 
# ### Confirmation Sub-model
# 
# Rather than assume all clinical cases are true cases, we can adjust the model to account for lab confirmation probability. This is done by including a sub-model that estimates age group-specific probabilities of confirmation, and using these probabilities to estimate the number of lab-confirmed cases. These estimates are then plugged into the model in place of the clinically-confirmed cases.
# 
# We specified a structured confirmation model to retrospectively determine the age group-specific probabilities of lab confirmation for measles, conditional on clinical diagnosis. Individual lab confirmation events $c_i$ were modeled as Bernoulli random variables, with the probability of confirmation being allowed to vary by age group:
# 
# $$c_i \sim \text{Bernoulli}(p_{a(i)})$$
# 
# where $a(i)$ denotes the appropriate age group for the individual indexed by i. There were 16 age groups, the first 15 of which were 5-year age intervals $[0,5), [5, 10), \ldots , [70, 75)$, with the 16th interval including all individuals 75 years and older.
# 
# Since the age interval choices were arbitrary, and the confirmation probabilities of adjacent groups likely correlated, we modeled the correlation structure directly, using a multivariate logit-normal model. Specifically, we allowed first-order autocorrelation among the age groups, whereby the variance-covariance matrix retained a tridiagonal structure. 
# 
# $$\begin{aligned}
# \Sigma = \left[{
# \begin{array}{c}
#   {\sigma^2} & {\sigma^2 \rho} & 0& \ldots & {0} & {0}  \\
#   {\sigma^2 \rho} & {\sigma^2} &  \sigma^2 \rho & \ldots & {0}  & {0} \\
#   {0} & \sigma^2 \rho & {\sigma^2} & \ldots & {0} & {0} \\
#   \vdots & \vdots & \vdots &  & \vdots & \vdots\\
#   {0} & {0} & 0 & \ldots &  {\sigma^2} & \sigma^2 \rho  \\
# {0} & {0} & 0 & \ldots & \sigma^2 \rho &  {\sigma^2} 
# \end{array}
# }\right]
# \end{aligned}$$
# 
# From this, the confirmation probabilities were specified as multivariate normal on the inverse-logit scale.
# 
# $$ \text{logit}(p_a) = \{a\} \sim N(\mu, \Sigma)$$
# 
# Priors for the confirmation sub-model were specified by:
# 
# $$\begin{aligned}
# \mu_i &\sim N(0, 100) \\
# \sigma &\sim \text{HalfCauchy}(25) \\
# \rho &\sim U(-1, 1)
# \end{aligned}$$

# Age classes are defined in 5-year intervals.


age_classes = [0,5,10,15,20,25,30,35,40,100]
measles_data.dropna(subset=['YEAR_AGE'], inplace=True)
measles_data['YEAR_AGE'] = measles_data.YEAR_AGE.astype(int)
measles_data['AGE_GROUP'] = pd.cut(measles_data.AGE, age_classes, right=False)


# Lab-checked observations are extracted for use in estimating lab confirmation probability.


CONFIRMED = measles_data.CONCLUSION == 'CONFIRMED'
CLINICAL = measles_data.CONCLUSION == 'CLINICAL'
DISCARDED = measles_data.CONCLUSION == 'DISCARDED'


# Extract confirmed and clinical subset, with no missing county information.


lab_subset = measles_data[(CONFIRMED | CLINICAL) & measles_data.COUNTY.notnull()].copy()



age = lab_subset.YEAR_AGE.values
ages = lab_subset.YEAR_AGE.unique()
counties = lab_subset.COUNTY.unique()
y = (lab_subset.CONCLUSION=='CONFIRMED').values



_lab_subset = lab_subset.replace({"CONCLUSION": {"CLINICAL": "UNCONFIRMED"}})
by_conclusion = _lab_subset.groupby(["YEAR_AGE", "CONCLUSION"])
counts_by_cause = by_conclusion.size().unstack().fillna(0)
ax = counts_by_cause.plot(kind='bar', stacked=True, xlim=(0,50), figsize=(15,5), grid=False)


# Proportion of lab-confirmed cases older than 20 years


(measles_data[CONFIRMED].YEAR_AGE>20).mean()



#Extract cases by age and time.
age_group = pd.cut(age, age_classes, right=False)
age_index = np.array([age_group.categories.tolist().index(i) for i in age_group])
age_groups = age_group.categories
age_groups



# Get index from full crosstabulation to use as index for each district
dates_index = measles_data.groupby(
        ['ONSET', 'AGE_GROUP']).size().unstack().index



unique_districts = measles_data.DISTRICT.dropna().unique()



excludes = ['BOM RETIRO']

N = sp_pop.drop(excludes).ix[unique_districts].sum().drop('Total')
N



N_age = N.iloc[:8]
N_age.index = age_groups[:-1]
N_age[age_groups[-1]] = N.iloc[8:].sum()
N_age


# Calcualte average susceptibility over age groups


age_slice_endpoints = [g[1:-1].split(',') for g in age_groups.values]
age_slices = [slice(int(i[0]), int(i[1])) for i in age_slice_endpoints]



p_susc_age = np.array([total_susc[s].mean() for s in age_slices])


# Compile bi-weekly confirmed and unconfirmed data by Sao Paulo district


sp_counts_2w = lab_subset.groupby(
    ['ONSET', 'AGE_GROUP']).size().unstack().reindex(dates_index).fillna(0).resample('2W', how='sum')

# All confirmed cases, by district
confirmed_data = lab_subset[lab_subset.CONCLUSION=='CONFIRMED']
confirmed_counts = confirmed_data.groupby(
    ['ONSET', 'AGE_GROUP']).size().unstack().reindex(dates_index).fillna(0).sum()

all_confirmed_cases = confirmed_counts.reindex_axis(measles_data['AGE_GROUP'].unique()).fillna(0)



# Ensure the age groups are ordered
I_obs = sp_counts_2w.reindex_axis(measles_data['AGE_GROUP'].unique(), 
                            axis=1).fillna(0).values.astype(int)


# Check shape of data frame
# 
# - 28 bi-monthly intervals, 9 age groups


assert I_obs.shape == (28, len(age_groups))


# Prior distribution on susceptible proportion:
# 
# $$p_s \sim \text{Beta}(5, 100)$$


from pymc import rbeta
plt.hist(rbeta(5, 100, 10000))



I_obs



# Extract observed data
obs_date = '1997-06-15'
obs_index = sp_counts_2w.index <= obs_date
I_obs_t = I_obs[obs_index]

# Specify confirmation model
confirmation = True



from theano.printing import pp



from pymc3 import *
import theano.tensor as tt
from theano import shared, scan
from theano.tensor.nlinalg import matrix_inverse as inv
invlogit = tt.nnet.sigmoid

with Model() as model:
    
    n_periods, n_groups = I_obs_t.shape
    
    if confirmation:
    
        mu = Normal('mu', mu=0, tau=0.0001, shape=n_groups)
        sig = HalfCauchy('sig', 25, shape=n_groups, testval=np.ones(n_groups))

        Tau = tt.diag(tt.pow(sig, -2)) 

        # Age-specific probabilities of confirmation as multivariate normal 
        # random variables
        beta_age = MvNormal('beta_age', mu=mu, tau=Tau, 
                            shape=n_groups)

        p_age = Deterministic('p_age', invlogit(beta_age))

        p_confirm = invlogit(beta_age[age_index])

        # Confirmation likelihood
        lab_confirmed = Bernoulli('lab_confirmed', p=p_confirm, observed=y)

        # Confirmed infecteds by age
        I_conf = Binomial('I_conf', I_obs_t, p_age, shape=I_obs_t.shape)

        I = I_conf
        
    else:
        
        I = I_obs_t
    
    # Transmission parameter
    beta = HalfCauchy('beta', 25, testval=9)   
    
    # Downsample annual series to observed age groups
    downsample = lambda x: np.array([x[s].mean() for s in age_slices])
    
    A = Deterministic('A', 75./(beta - 1))
    lt_sum = downsample(np.tril(FOI_mat).sum(0)[::-1])
    natural_susc = tt.exp((-1/A) * lt_sum)
    
    total_susc = downsample(sia_susc) * downsample(vacc_susc) * natural_susc
    
    # Estimated total initial susceptibles
    S_0 = Binomial('S_0', n=shared(N_age.values), p=total_susc, shape=N_age.shape)

    # Susceptibles at each time step (subtract cumulative cases from initial S)
    S = S_0 - I.cumsum(0)      

    # Force of infection
    lam = tt.transpose(beta * I.sum(1) * tt.transpose(S / N_age.values))
    
    # Likelihood of observed cases as function of FOI of prevous period
    like = Potential('like', 
            dist_math.logpow(lam[:-1], I[1:]) - dist_math.factln(I[1:]) - lam[:-1])



model.vars.pop(beta)



with model:
    step1 = Slice(vars=[beta])
    step2 = Metropolis(vars=[mu, sig, beta_age, I_conf, beta, S_0])
    trace = sample(20000, step=(step1, step2))
    



burn = 10000



forestplot(trace, vars=['S_0'])



traceplot(trace[burn:], vars=['beta', 'A'])



traceplot(trace[burn:], vars=['beta', 'A'])



summary(trace[burn:], vars=['A'])



forestplot(trace[burn:], vars=['p_age'])


