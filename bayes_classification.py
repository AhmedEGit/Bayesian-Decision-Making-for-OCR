import numpy as np


q_discrete = np.array([0]*10 + [1] + [0]*10)
W = np.array([[0, 1], [1, 0]])
discreteA = {'Prior': 0.6153846153846154,
             'Prob': np.array([0.0125, 0., 0., 0.0125, 0.025, 0.0125, 0.025, 0.0375, 0.075, 0.1, 0.2125, 0.1375, 0.15, 0.1, 0.0875, 0.0125, 0., 0., 0., 0., 0.])}
discreteC = {'Prior': 0.38461538461538464,
             'Prob': np.array([0., 0., 0., 0.02, 0.02, 0.22, 0.46, 0.16, 0.1, 0.02, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])}

def bayes_risk_discrete(discreteA, discreteC, W, q_discrete):
    R_discrete=discreteA['Prior']*np.sum(discreteA['Prob'][:11])*0+discreteA['Prior']*discreteA['Prob'][10]*1+discreteA['Prior']*np.sum(discreteA['Prob'][12:])*0+discreteC['Prior']*np.sum(discreteC['Prob'][:11])*1+discreteC['Prior']*discreteC['Prob'][10]*0+discreteC['Prior']*np.sum(discreteC['Prob'][12:])*1
    return R_discrete
print(bayes_risk_discrete(discreteA, discreteC, W, q_discrete))
#%%
def find_strategy_discrete(discreteA, discreteC, W):
    a=discreteA['Prob']*discreteA['Prior']*W[0,0]+discreteC['Prob']*discreteC['Prior']*W[1,0]
    c=discreteA['Prob']*discreteA['Prior']*W[0,1]+discreteC['Prob']*discreteC['Prior']*W[1,1]
    f=np.array([a,c])
    q_discrete=np.argmin(f,axis=0)


    return q_discrete
print(find_strategy_discrete(discreteA, discreteC, W))

#%%
distribution1 = {}
distribution2 = {}
distribution1['Prior'] = 0.3
distribution2['Prior'] = 0.7
distribution1['Prob'] = np.array([0.2, 0.3, 0.4, 0.1])
distribution2['Prob'] = np.array([0.5, 0.4, 0.1, 0.0])
W = np.array([[0, 1], [1, 0]])
q = find_strategy_discrete(distribution1, distribution2, W)
print(q)
#%%
print(bayes_risk_discrete(discreteA, discreteC, W, find_strategy_discrete(discreteA, discreteC, W)))
#%%
import matplotlib.pyplot as plt
import matplotlib.patches as patches
def visualize_discrete(discrete_A, discrete_C, q):
    posterior_A = discrete_A['Prob'] * discrete_A['Prior']
    posterior_C = discrete_C['Prob'] * discrete_C['Prior']

    max_prob = np.max([posterior_A, posterior_C])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("Posterior probabilities and strategy q")
    plt.xlabel("feature")
    plt.ylabel("posterior probabilities")

    bins = np.array(range(posterior_A.size + 1)) - int(posterior_A.size / 2)

    width = 0.75
    bar_plot_A = plt.bar(bins[:-1], posterior_A, width=width, color='b', alpha=0.75)
    bar_plot_C = plt.bar(bins[:-1], posterior_C, width=width, color='r', alpha=0.75)

    plt.legend((bar_plot_A, bar_plot_C), (r'$p_{XK}(x,A)$', r'$p_{XK}(x,C)$'))

    sub_level = - max_prob / 8
    height = np.abs(sub_level)
    for idx in range(len(bins[:-1])):
        b = bins[idx]
        col = 'r' if q[idx] == 1 else 'b'
        patch = patches.Rectangle([b - 0.5, sub_level], 1, height, angle=0.0, color=col, alpha=0.75)
        ax.add_patch(patch)

    plt.ylim(bottom=sub_level)
    plt.text(bins[0], -max_prob / 16, 'strategy q')

W1 = np.array([[0, 1],
              [1, 0]])
q_discrete1 = find_strategy_discrete(discreteA, discreteC, W1)
print(visualize_discrete(discreteA, discreteC, q_discrete1))
plt.savefig("classif_W1.png")


#%%
W2 = np.array([[0, 5],
               [1, 0]])
q_discrete2 = find_strategy_discrete(discreteA, discreteC, W2)

print(visualize_discrete(discreteA, discreteC, q_discrete2))
plt.savefig("classif_W2.png")
#%%
def compute_measurement_lr_discrete(imgs):
    assert len(imgs.shape) in (3, 4)
    assert (imgs.shape[2] == 3 or len(imgs.shape) == 3)

    mu = -563.9
    sigma = 2001.6

    if len(imgs.shape) == 3:
        imgs = np.expand_dims(imgs, axis=2)

    imgs = imgs.astype(np.int32)
    height, width, channels, count = imgs.shape

    x_raw = np.sum(np.sum(np.sum(imgs[:, 0:int(width / 2), :, :], axis=0), axis=0), axis=0) - \
            np.sum(np.sum(np.sum(imgs[:, int(width / 2):, :, :], axis=0), axis=0), axis=0)
    x_raw = np.squeeze(x_raw)

    x = np.atleast_1d(np.round((x_raw - mu) / (2 * sigma) * 10))
    x[x > 10] = 10
    x[x < -10] = -10

    assert x.shape == (imgs.shape[-1], )
    return x



data = np.load("data_33rpz_bayes.npz", allow_pickle=True)
alphabet = data["alphabet"]
images_test = data["images_test"]
labels_test = data["labels_test"]
contA = {'Mean': 124.2625,'Sigma': 1434.45420083,'Prior': 0.61538462}
contC = {'Mean': -2010.98,'Sigma': 558.42857106,'Prior': 0.38461538}
discreteA = {'Prior': 0.6153846153846154,
             'Prob': np.array([0.0125, 0., 0., 0.0125, 0.025, 0.0125, 0.025, 0.0375, 0.075, 0.1, 0.2125, 0.1375, 0.15, 0.1, 0.0875, 0.0125, 0., 0., 0., 0., 0.])}
discreteC = {'Prior': 0.38461538461538464,
             'Prob': np.array([0., 0., 0., 0.02, 0.02, 0.22, 0.46, 0.16, 0.1, 0.02, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])}
measurements_discrete = compute_measurement_lr_discrete(images_test)
data = np.load("data_33rpz_bayes.npz", allow_pickle=True)
alphabet = data["alphabet"]
images_test = data["images_test"]
labels_test = data["labels_test"]
contA = {'Mean': 124.2625,'Sigma': 1434.45420083,'Prior': 0.61538462}
contC = {'Mean': -2010.98,'Sigma': 558.42857106,'Prior': 0.38461538}
discreteA = {'Prior': 0.6153846153846154,
             'Prob': np.array([0.0125, 0., 0., 0.0125, 0.025, 0.0125, 0.025, 0.0375, 0.075, 0.1, 0.2125, 0.1375, 0.15, 0.1, 0.0875, 0.0125, 0., 0., 0., 0., 0.])}
discreteC = {'Prior': 0.38461538461538464,
             'Prob': np.array([0., 0., 0., 0.02, 0.02, 0.22, 0.46, 0.16, 0.1, 0.02, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])}
measurements_discrete = compute_measurement_lr_discrete(images_test)
#print(measurements_discrete)
def classify_discrete(measurements, q):
    label = np.array([])
    for i in range(len(measurements)):
        v = q[int(measurements[i]) + 10]
        label = np.append(label, v)
    return label
print(classify_discrete(measurements_discrete, find_strategy_discrete(discreteA, discreteC, W)))
def classification_error(predictions, labels):
    compt=0
    for i in range(len(predictions)):
        if predictions[i]!=labels[i]:
           compt+=1
    error=compt/len(predictions)

    return error
labels_estimated_discrete = classify_discrete(measurements_discrete, q_discrete)
error_discrete = classification_error(labels_estimated_discrete, labels_test)
print(error_discrete)

def find_strategy_2normal(distribution_A, distribution_B):
    a=(1/(2*distribution_B['Sigma']**2))-(1/(2*distribution_A['Sigma']**2))
    b=(distribution_A['Mean']/(distribution_A['Sigma']**2))-(distribution_B['Mean']/(distribution_B['Sigma']**2))
    c=((distribution_B['Mean']**2/(2*distribution_B['Sigma']**2))-(distribution_A['Mean']**2/(2*distribution_A['Sigma']**2)))+(np.log(distribution_A['Prior']*distribution_B['Sigma']/(distribution_B['Prior']*distribution_A['Sigma'])))
    root=np.roots([a,b,c])
    s_A = distribution_A['Sigma']
    m_A = distribution_A['Mean']
    p_A = distribution_A['Prior']
    s_B = distribution_B['Sigma']
    m_B = distribution_B['Mean']
    p_B = distribution_B['Prior']
    eps = 1e-10
    if p_A < eps:
        q={'decision': 1}
    elif p_B < eps:
        q={'decision': 0}
    else:

        if a == 0:
            # same sigmas -> not quadratic
            if b == 0:
                # same sigmas and same means -> not even linear
                q={'decision': 0}
            else:
                # same sigmas, different means -> linear equation
                pol=np.poly1d([(m_A+m_B)/(s_A**2),((m_B**2-m_A**2)/(2*s_A**2))+np.log(p_A/p_B)])
                if pol<0:
                    dec2=1
                else:
                    dec1=0
                q={'t': -c/b,'decision': np.array([dec1,dec2])}
        else:
            # quadratic equation
            D=(b**2)-(4*a*c)
            if D > 0:
                if 0<a:
                   q={'t2':root[1],'t1':root[0] ,'decision': np.array([0,1,0])}
                elif a<0:
                   q={'t2':root[0],'t1':root[1],'decision': np.array([1,0,1])}
            elif D == 0:
                if 0<a:
                   q={'t':-b/(2*a),'decision': np.array([0,0])}
                elif a<0:
                   q={'t':-b/(2*a),'decision': np.array([1,1])}
            elif D < 0:
                if 0<a:
                   q={'decision': 0}
                elif a<0:
                   q={'decision': 1}

    return q
contA = {'Mean': 124.2625,'Sigma': 1434.45420083,'Prior': 0.61538462}
contC = {'Mean': -2010.98,'Sigma': 558.42857106,'Prior': 0.38461538}
print(find_strategy_2normal(contA, contC))





#%%
from scipy.stats import norm
def visualize_2norm(cont_A, cont_B, q):
    n_sigmas = 5
    n_points = 200

    A_range = (cont_A['Mean'] - n_sigmas * cont_A['Sigma'],
               cont_A['Mean'] + n_sigmas * cont_A['Sigma'])
    B_range = (cont_B['Mean'] - n_sigmas * cont_B['Sigma'],
               cont_B['Mean'] + n_sigmas * cont_B['Sigma'])
    start = min(A_range[0], B_range[0])
    stop = max(A_range[1], B_range[1])

    xs = np.linspace(start, stop, n_points)
    A_vals = cont_A['Prior'] * norm.pdf(xs, cont_A['Mean'], cont_A['Sigma'])
    B_vals = cont_B['Prior'] * norm.pdf(xs, cont_B['Mean'], cont_B['Sigma'])

    colors = ['r', 'b']
    plt.plot(xs, A_vals, c=colors[0], label='A')
    plt.plot(xs, B_vals, c=colors[1], label='B')

    plt.axvline(x=q['t1'], c='k', lw=0.5, ls=':')
    plt.axvline(x=q['t2'], c='k', lw=0.5, ls=':')

    offset = 0.000007
    sub_level = -0.000025
    left = xs[0]
    right = xs[-1]
    plt.savefig("thresholds.png")

print(visualize_2norm(contA, contC, find_strategy_2normal(contA, contC)))
#%%
def bayes_risk_2normal(distribution_A, distribution_B, q):
    a=find_strategy_2normal(distribution_A, distribution_B)['decision'][0]
    b=find_strategy_2normal(distribution_A, distribution_B)['decision'][1]
    c=find_strategy_2normal(distribution_A, distribution_B)['decision'][2]
    myr1=find_strategy_2normal(distribution_A, distribution_B)['t2']
    myr2=find_strategy_2normal(distribution_A, distribution_B)['t1']
    if a==0:
       d=distribution_A['Prior']
       area1=norm(distribution_A['Mean'],distribution_A['Sigma']).cdf(myr2)
    else:
       d=distribution_B['Prior']
       area1=norm(distribution_B['Mean'],distribution_B['Sigma']).cdf(myr2)
    if b==0:
       e=distribution_A['Prior']
       area2=norm(distribution_A['Mean'],distribution_A['Sigma']).cdf(myr1)-norm(distribution_A['Mean'],distribution_A['Sigma']).cdf(myr2)
    else:
       e=distribution_B['Prior']
       area2=norm(distribution_B['Mean'],distribution_B['Sigma']).cdf(myr1)-norm(distribution_B['Mean'],distribution_B['Sigma']).cdf(myr2)
    if c==0:
       f=distribution_A['Prior']
       area3=1-norm(distribution_A['Mean'],distribution_A['Sigma']).cdf(myr1)
    else:
       f=distribution_B['Prior']
       area3=1-norm(distribution_B['Mean'],distribution_B['Sigma']).cdf(myr1)
    R=1-(d*area1+e*area2+f*area3)
    return R
print(bayes_risk_2normal(contA,contC,find_strategy_2normal(contA,contC)))

#%%
def compute_measurement_lr_cont(imgs):


    assert len(imgs.shape) == 3

    width = imgs.shape[1]
    sum_rows = np.sum(imgs, dtype=np.float64, axis=0)

    x = np.sum(sum_rows[0:int(width / 2), :], axis=0) - np.sum(sum_rows[int(width / 2):, :], axis=0)

    assert x.shape == (imgs.shape[2], )
    return x
print(compute_measurement_lr_cont(images_test))
