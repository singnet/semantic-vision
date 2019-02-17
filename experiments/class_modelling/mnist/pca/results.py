import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

rc, lk, f1r, f1lk = [], [], [], []
logit_score = []
for i in range(10, 120, 10): # through metrics_z{i}.txt
    m = np.loadtxt(f'metrics_z{i}.txt')
    rc.append(m[:, 0].reshape(-1, 1))
    lk.append(m[:, 1].reshape(-1, 1))
    f1r.append(m[:, 2].reshape(-1, 1))
    f1lk.append(m[:, 3].reshape(-1, 1))
    m = np.loadtxt(f'logit_z{i}.txt')
    logit_score.append(m.reshape(-1, 1))

#print([r.shape for r in rc]) 
rc = np.hstack(rc)
rc_mean = np.mean(rc, 0)
rc = np.vstack((rc, rc_mean))

lk = np.hstack(lk)
lk_mean = np.mean(lk, 0)
lk = np.vstack((lk, lk_mean))

f1r = np.hstack(f1r)
f1r_mean = np.mean(f1r, 0)
f1r = np.vstack((f1r, f1r_mean))

f1lk = np.hstack(f1lk)
f1lk_mean = np.mean(f1lk, 0)
f1lk = np.vstack((f1lk, f1lk_mean))

logit_score = np.hstack(logit_score)
logit_mean = np.mean(logit_score, 0)
logit_score = np.vstack((logit_score, logit_mean))

# reconstruction
drc = pd.DataFrame(data=rc,
                   index=[str(i) for i in range(10)] + ['среднее'],
                   columns=[str(i) for i in range(10, 120, 10)])

#print(drc)
#pdrc = drc.pivot(index=)
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(drc, annot=True, linewidths=.5, cbar=False, cmap='coolwarm', square=True, ax=ax)
plt.title('Fails on reconstruction error')
plt.xlabel('components')
plt.ylabel('digit')
plt.savefig('rec_fails.png')
plt.cla()
plt.clf()
plt.close()

# likelihood
dlk = pd.DataFrame(data=lk,
                   index=[str(i) for i in range(10)] + ['среднее'],
                   columns=[str(i) for i in range(10, 120, 10)])

#print(dlk)
#pdrc = drc.pivot(index=)
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(dlk, annot=True, linewidths=.5, cbar=False, cmap='coolwarm', square=True, ax=ax)
plt.title('Fails on likelihood')
plt.xlabel('components')
plt.ylabel('digit')
plt.savefig('lk_fails.png')
plt.cla()
plt.clf()
plt.close()

# f1 reconstruction


drc = pd.DataFrame(data=f1r,
                   index=[str(i) for i in range(10)] + ['среднее'],
                   columns=[str(i) for i in range(10, 120, 10)])

#print(drc)
#pdrc = drc.pivot(index=)
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(drc, annot=True, linewidths=.5, cbar=False, cmap='coolwarm', square=True, ax=ax)
plt.title('F1 reconstruction error')
plt.xlabel('components')
plt.ylabel('digit')
plt.savefig('f1_rec.png')
plt.cla()
plt.clf()
plt.close()

# f1 reconstruction

drc = pd.DataFrame(data=f1lk,
                   index=[str(i) for i in range(10)] + ['среднее'],
                   columns=[str(i) for i in range(10, 120, 10)])

#print(drc)
#pdrc = drc.pivot(index=)
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(drc, annot=True, linewidths=.5, cbar=False, cmap='coolwarm', square=True, ax=ax)
plt.title('F1 likelihood error')
plt.xlabel('components')
plt.ylabel('digit')
plt.savefig('f1_lk.png')
plt.cla()
plt.clf()
plt.close()

# logit score

drc = pd.DataFrame(data=logit_score,
                   index=[str(i) for i in range(10)] + ['среднее'],
                   columns=[str(i) for i in range(10, 120, 10)])

#print(drc)
#pdrc = drc.pivot(index=)
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(drc, annot=True, linewidths=.5, cbar=False, cmap='coolwarm', square=True, ax=ax)
plt.title('Merged error')
plt.xlabel('components')
plt.ylabel('digit')
plt.savefig('logit_score.png')
plt.cla()
plt.clf()
plt.close()


