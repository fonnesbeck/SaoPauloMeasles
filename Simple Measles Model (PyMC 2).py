
# coding: utf-8

# # Disease Outbreak Response Decision-making Under Uncertainty: A retrospective analysis of measles in Sao Paulo

# In[1]:

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


# In[2]:

data_dir = "data/"


# Import outbreak data

# In[3]:

measles_data = pd.read_csv(data_dir+"measles.csv", index_col=0)
measles_data.NOTIFICATION = pd.to_datetime(measles_data.NOTIFICATION)
measles_data.BIRTH = pd.to_datetime(measles_data.BIRTH)
measles_data.ONSET = pd.to_datetime(measles_data.ONSET)


# In[4]:

measles_data = measles_data.replace({'DISTRICT': {'BRASILANDIA':'BRAZILANDIA'}})


# Sao Paulo population by district

# In[5]:

sp_pop = pd.read_csv(data_dir+'sp_pop.csv', index_col=0)


# In[6]:

_names = sp_pop.index.values
_names[_names=='BRASILANDIA'] = 'BRAZILANDIA'
sp_pop.set_index(_names, inplace = True)


# In[7]:

sp_pop.head()


# Plot of cumulative cases by district

# In[8]:

measles_onset_dist = measles_data.groupby(['DISTRICT','ONSET']).size().unstack(level=0).fillna(0)
measles_onset_dist.cumsum().plot(legend=False, grid=False)


# In[9]:

total_district_cases = measles_onset_dist.sum()


# Top 5 districts by number of cases

# In[10]:

totals = measles_onset_dist.sum()
totals.sort(ascending=False)
totals[:5]


# Age distribution of cases, by confirmation status

# In[11]:

by_conclusion = measles_data.groupby(["YEAR_AGE", "CONCLUSION"])
counts_by_cause = by_conclusion.size().unstack().fillna(0)
ax = counts_by_cause.plot(kind='bar', stacked=True, xlim=(0,50), figsize=(15,5))


# ## Vaccination data

# In[12]:

vaccination_data = pd.read_csv('data/BrazilVaxRecords.csv', index_col=0)
vaccination_data.head()


# In[13]:

vaccination_data.VAX[:18]


# In[14]:

vax_97 = np.r_[[0]*(1979-1921+1), vaccination_data.VAX[:17]]
n = len(vax_97)
FOI_mat = np.resize((1 - vax_97*0.9), (n,n)).T


# In[15]:

# Mean age of infection for those born prior to vaccination coverage, assuming R0=16
A = 4.37


# In[16]:

(1 - vax_97*0.9)[:-1]


# In[17]:

np.tril(FOI_mat).sum(0)


# In[18]:

natural_susc = np.exp((-1/A) * np.tril(FOI_mat).sum(0))[::-1]
vacc_susc = (1 - vax_97*0.9)[::-1]
vacc_susc[0] = 0.5
vacc_susc


# In[19]:


sia_susc = np.ones(len(vax_97))
birth_year = np.arange(1922, 1998)[::-1]
by_mask = (birth_year > 1983) & (birth_year < 1992)
sia_susc[by_mask] *= 0.2


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

# In[22]:

age_classes = [0,5,10,15,20,25,30,35,40,100]
measles_data.dropna(subset=['YEAR_AGE'], inplace=True)
measles_data['YEAR_AGE'] = measles_data.YEAR_AGE.astype(int)
measles_data['AGE_GROUP'] = pd.cut(measles_data.AGE, age_classes, right=False)


# Lab-checked observations are extracted for use in estimating lab confirmation probability.

# In[23]:

CONFIRMED = measles_data.CONCLUSION == 'CONFIRMED'
CLINICAL = measles_data.CONCLUSION == 'CLINICAL'
DISCARDED = measles_data.CONCLUSION == 'DISCARDED'


# Extract confirmed and clinical subset, with no missing county information.

# In[24]:

lab_subset = measles_data[(CONFIRMED | CLINICAL) & measles_data.COUNTY.notnull()].copy()


# In[25]:

age = lab_subset.YEAR_AGE.values
ages = lab_subset.YEAR_AGE.unique()
counties = lab_subset.COUNTY.unique()
y = (lab_subset.CONCLUSION=='CONFIRMED').values


# In[26]:

_lab_subset = lab_subset.replace({"CONCLUSION": {"CLINICAL": "UNCONFIRMED"}})
by_conclusion = _lab_subset.groupby(["YEAR_AGE", "CONCLUSION"])
counts_by_cause = by_conclusion.size().unstack().fillna(0)
ax = counts_by_cause.plot(kind='bar', stacked=True, xlim=(0,50), figsize=(15,5), grid=False)


# In[27]:

lab_subset.shape


# In[28]:

y.sum()


# Proportion of lab-confirmed cases older than 20 years

# In[29]:

(measles_data[CONFIRMED].YEAR_AGE>20).mean()


# In[30]:

age_classes


# In[31]:

#Extract cases by age and time.
age_group = pd.cut(age, age_classes, right=False)
age_index = np.array([age_group.categories.tolist().index(i) for i in age_group])
age_groups = age_group.categories
age_groups


# In[32]:

age_slice_endpoints = [g[1:-1].split(',') for g in age_groups]
age_slices = [slice(int(i[0]), int(i[1])) for i in age_slice_endpoints]


# In[33]:

# Get index from full crosstabulation to use as index for each district
dates_index = measles_data.groupby(
        ['ONSET', 'AGE_GROUP']).size().unstack().index


# In[34]:

unique_districts = measles_data.DISTRICT.dropna().unique()


# In[35]:

excludes = ['BOM RETIRO']


# In[36]:

N = sp_pop.drop(excludes).ix[unique_districts].sum().drop('Total')
N


# In[37]:

N_age = N.iloc[:8]
N_age.index = age_groups[:-1]
N_age[age_groups[-1]] = N.iloc[8:].sum()
N_age


# Compile bi-weekly confirmed and unconfirmed data by Sao Paulo district

# In[38]:

sp_counts_2w = lab_subset.groupby(
    ['ONSET', 'AGE_GROUP']).size().unstack().reindex(dates_index).fillna(0).resample('2W', how='sum')

# All confirmed cases, by district
confirmed_data = lab_subset[lab_subset.CONCLUSION=='CONFIRMED']
confirmed_counts = confirmed_data.groupby(
    ['ONSET', 'AGE_GROUP']).size().unstack().reindex(dates_index).fillna(0).sum()

all_confirmed_cases = confirmed_counts.reindex_axis(measles_data['AGE_GROUP'].unique()).fillna(0)


# In[39]:

# Ensure the age groups are ordered
I_obs = sp_counts_2w.reindex_axis(measles_data['AGE_GROUP'].unique(), 
                            axis=1).fillna(0).values.astype(int)


# Check shape of data frame
# 
# - 28 bi-monthly intervals, 9 age groups

# In[40]:

assert I_obs.shape == (28, len(age_groups))


# Prior distribution on susceptible proportion:
# 
# $$p_s \sim \text{Beta}(2, 100)$$

# In[41]:

from pymc import rbeta
plt.hist(rbeta(2, 100, 10000))


# In[42]:

I_obs


# In[43]:

obs_date = '1997-12-01' #'1997-06-15'
obs_index = sp_counts_2w.index <= obs_date
I_obs_t = I_obs[obs_index]


# In[44]:

np.sum(I_obs_t, (0)) / float(I_obs_t.sum())


# In[45]:

from pymc import rgamma


# In[46]:

plt.hist(rgamma(16,1,size=10000))


# In[47]:

75./age.mean() 


# In[48]:

from pymc import MCMC, Matplot, AdaptiveMetropolis, Slicer, MAP
from pymc import (Uniform, DiscreteUniform, Beta, Binomial, Normal, CompletedDirichlet,
                  Poisson, NegativeBinomial, negative_binomial_like, poisson_like,
                  Lognormal, Exponential, binomial_like,
                  TruncatedNormal, Binomial, Gamma, HalfCauchy, normal_like,
                  MvNormalCov, Bernoulli, Uninformative, 
                  Multinomial, rmultinomial, rbinomial,
                  Dirichlet, multinomial_like)
from pymc import (Lambda, observed, invlogit, deterministic, potential, stochastic,)

def measles_model(obs_date, confirmation=True, spatial_weighting=False, all_traces=True):
    
    n_periods, n_age_groups = I_obs.shape
    
    ### Confirmation sub-model
    
    if confirmation:

        # Specify priors on age-specific means
        age_classes = np.unique(age_index)

        mu = Normal("mu", mu=0, tau=0.0001, value=[0]*len(age_classes))
        sig = HalfCauchy('sig', 0, 25, value=1)
        var = sig**2
        cor = Uniform('cor', -1, 1, value=0)

        # Build variance-covariance matrix with first-order correlation 
        # among age classes
        @deterministic
        def Sigma(var=var, cor=cor):
            I = np.eye(len(age_classes))*var
            E = np.diag(np.ones(len(age_classes)-1), k=-1)*var*cor
            return I + E + E.T

        # Age-specific probabilities of confirmation as multivariate normal 
        # random variables
        beta_age = MvNormalCov("beta_age", mu=mu, C=Sigma, 
                        value=[1]*len(age_classes))
        p_age = Lambda('p_age', lambda t=beta_age: invlogit(t))

        @deterministic(trace=False)
        def p_confirm(beta=beta_age):
            return invlogit(beta[age_index])


        # Confirmation likelihood
        lab_confirmed = Bernoulli('lab_confirmed', p=p_confirm, value=y, 
                                observed=True)


    '''
    Truncate data at observation period
    '''
    obs_index = sp_counts_2w.index <= obs_date
    I_obs_t = I_obs[obs_index]  
                        

    # Index for observation date, used to index out values of interest 
    # from the model.
    t_obs = obs_index.sum() - 1
    
    if confirmation:
        
        @stochastic(trace=all_traces, dtype=int)
        def I(value=(I_obs_t).astype(int), n=I_obs_t, p=p_age):
            # Binomial confirmation process
            return np.sum([binomial_like(x, x.sum(), p) for x in value])

    else:
        
        I = I_obs_t
        
    assert I.shape == (t_obs +1, n_age_groups)
    
    # Transmission parameter
    beta = HalfCauchy('beta', 0, 25, value=[8]*n_age_groups) 

    # Downsample annual series to observed age groups
    downsample = lambda x: np.array([x[s].mean() for s in age_slices])
    
    A = Lambda('A', lambda beta=beta: 75./(beta - 1))
    lt_sum = downsample(np.tril(FOI_mat).sum(0)[::-1])
    natural_susc = Lambda('natural_susc', lambda A=A: np.exp((-1/A) * lt_sum))
    

#     natural_susc = Beta('natural_susc', 1, 1, value=[0.02]*n_age_groups)
    @deterministic
    def p_susceptible(natural_susc=natural_susc): 
        return downsample(sia_susc) * downsample(vacc_susc) * natural_susc
    
    # Estimated total initial susceptibles
    S_0 = Binomial('S_0', n=N_age.astype(int), p=p_susceptible)

    S = Lambda('S', lambda I=I, S_0=S_0: S_0 - I.cumsum(0))
    
    # Check shape
    assert S.value.shape == (t_obs+1., n_age_groups)

    S_t = Lambda('S_t', lambda S=S: S[-1])      
    
    
    @deterministic
    def R(beta=beta, S=S): 
        return (beta * S / N_age.values).T
    
    # Force of infection
    @deterministic
    def lam(beta=beta, I=I, S=S): 
        return (I.sum(1) * (beta * S / N_age.values).T).T
    
    # Check shape
    assert lam.value.shape == (t_obs+1, n_age_groups)
    
    # Poisson likelihood for observed cases
    @potential
    def new_cases(I=I, lam=lam):
        return poisson_like(I[1:], lam[:-1])
    

    return locals()


# In[49]:

iterations = 200000
burn = 190000


# In[57]:

M = MCMC(measles_model('1997-06-15', confirmation=True))
M.sample(iterations, burn)


# In[58]:

M.sample(iterations, burn)


# In[63]:

Matplot.summary_plot(M.p_susceptible, chain=-1)


# In[60]:

Matplot.summary_plot(M.natural_susc, chain=-1)


# In[62]:

Matplot.summary_plot(M.beta, chain=-1)


# In[ ]:



