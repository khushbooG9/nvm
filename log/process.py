import os
from math import isnan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D

tonic = 0.0                                                                     
Ra = 1.0                                                                        
Rv = 20.0 

class Sample:
    def __init__(self, filename, s, av, va, mean_latency, stdev_latency,
            bold_amy, bold_vmpfc, activ_amy, activ_vmpfc, max_activ_amy,
            max_activ_vmpfc, num_responses):
        self.filename = filename
        self.s = s
        self.av = av
        self.va = va
        self.mean_latency = mean_latency
        self.stdev_latency = stdev_latency

        # Add recurrent activity to BOLD?
        self.bold_amy = bold_amy + ((1.0 - 0.025) * activ_amy)
        self.bold_vmpfc = bold_vmpfc + ((1.0 - 0.00125) * activ_vmpfc)
        #self.bold_amy = bold_amy
        #self.bold_vmpfc = bold_vmpfc

        self.activ_amy = activ_amy
        self.activ_vmpfc = activ_vmpfc
        self.max_activ_amy = max_activ_amy
        self.max_activ_vmpfc = max_activ_vmpfc
        self.num_responses = num_responses
        self.steady = Ra * (s - va * tonic) / (1 + (Ra * Rv * av * va)) 

    def valid(self, num_faces):
        return num_faces == self.num_responses

    def clipped(self, threshold):
        return self.mean_latency <= threshold

    def to_3d(self):
        return (self.s, self.av, self.va)

    def cls(self, num_faces, threshold):
        v = self.valid(num_faces)
        c = self.clipped(threshold)
        return (v and not c, not v, c).index(True)

    def activ_bounded(self, bound):
        return self.max_activ_amy < bound and self.max_activ_vmpfc < bound

    def __str__(self):
        return (" %20s: " + ("%8.4f " * 5)) % (
            self.filename, self.s, self.av, self.va, self.mean_latency, self.steady)

    def __eq__(self, other): 
        return self.s == other.s \
            and self.av == other.av \
            and self.va == other.va \
            and self.mean_latency == other.mean_latency \
            and self.stdev_latency == other.stdev_latency \
            and self.bold_amy == other.bold_amy \
            and self.bold_vmpfc == other.bold_vmpfc \
            and self.activ_amy == other.activ_amy \
            and self.activ_vmpfc == other.activ_vmpfc \
            and self.max_activ_amy == other.max_activ_amy \
            and self.max_activ_vmpfc == other.max_activ_vmpfc \
            and self.num_responses == other.num_responses

samples = []
max_samples = 500

directories = ["explore"] + ["explore_trial_%d" % i for i in [1, 2, 3, 4, 5]]

for d in directories:
    count = 1
    for f in sorted(os.listdir(d)):
        if f.endswith(".txt"):

            s = 0.0
            av = 0.0
            va = 0.0
            mean_latency = 0.0
            stdev_latency = 0.0
            bold_amy = 0.0
            bold_vmpfc = 0.0
            activ_amy = 0.0
            activ_vmpfc = 0.0
            max_activ_amy = 0.0
            max_activ_vmpfc = 0.0
            num_responses = len(tuple(l for l in open("%s/%s" % (d, f)) if "correct" in l))

            for line in open("%s/%s" % (d, f)):
                if line.startswith("Using amygdala sensitivity:"):
                    s = (float(line.split(":")[1].strip()))
                elif line.startswith("Response latency mean:"):
                    mean_latency = (float(line.split(":")[1].strip()))
                elif line.startswith("Response latency std dev:"):
                    stdev_latency = (float(line.split(":")[1].strip()))
                elif line.startswith("Using amygdala->vmPFC:"):
                    av = (float(line.split(":")[1].strip()))
                elif line.startswith("Using vmPFC->amygdala:"):
                    va = (float(line.split(":")[1].strip()))
                elif line.startswith("BOLD Amygdala:"):
                    bold_amy = (float(line.split(":")[1].strip()))
                elif line.startswith("BOLD vmPFC:"):
                    bold_vmpfc = (float(line.split(":")[1].strip()))
                elif line.startswith("Activation Amygdala:"):
                    activ_amy = (float(line.split(":")[1].strip()))
                elif line.startswith("Activation vmPFC:"):
                    activ_vmpfc = (float(line.split(":")[1].strip()))
                elif line.startswith("Max Activation Amygdala:"):
                    max_activ_amy = (float(line.split(":")[1].strip()))
                elif line.startswith("Max Activation vmPFC:"):
                    max_activ_vmpfc = (float(line.split(":")[1].strip()))

            if mean_latency != 0.0:
                count += 1
                samples.append(Sample(f, s, av, va, mean_latency, stdev_latency,
                    bold_amy, bold_vmpfc, activ_amy, activ_vmpfc, max_activ_amy,
                    max_activ_vmpfc, num_responses))
                if count == max_samples:
                    break


valid = tuple(s for s in samples if s.valid(5))
invalid = tuple(s for s in samples if not s.valid(5))
clipped = tuple(s for s in samples if s.clipped(67.0))
good = tuple(s for s in samples if s.valid(5) and not s.clipped(67.0))

print("Valid: %d" % len(valid))
print("Invalid: %d" % len(invalid))
print("Clipped: %d" % len(clipped))
print("Good: %d" % len(good))

'''
print("\nClipped:")
for s in clipped: print(s)

print("\nGood:")
for s in sorted(good, key=lambda x: x.mean_latency): print(s)

print("\nInvalid:")
for s in invalid: print(s)
'''

mean_s = np.mean(tuple(sample.s for sample in good))
mean_mean_latency = np.mean(tuple(sample.mean_latency for sample in good))
mean_stdev_latency = np.mean(tuple(sample.stdev_latency for sample in good))
mean_av = np.mean(tuple(sample.av for sample in good))
mean_va = np.mean(tuple(sample.va for sample in good))
mean_bold_amy = np.mean(tuple(sample.bold_amy for sample in good))
mean_bold_vmpfc = np.mean(tuple(sample.bold_vmpfc for sample in good))
mean_activ_amy = np.mean(tuple(sample.activ_amy for sample in good))
mean_activ_vmpfc = np.mean(tuple(sample.activ_vmpfc for sample in good))

stdev_amy = np.std(tuple(sample.s for sample in good))
stdev_mean = np.std(tuple(sample.mean_latency for sample in good))
stdev_stdev = np.std(tuple(sample.stdev_latency for sample in good))

# ms per timestep
scale_factor = (440.0 - 120.0) / mean_mean_latency

print("Mean Amygdala sensitivity:")
print(mean_s)
print("")
print("Mean amygdala -> vmPFC strength:")
print(mean_av)
print("")
print("Mean vmPFC -> amygdala strength:")
print(mean_va)
print("")
print("Mean latency:")
print(mean_mean_latency, mean_mean_latency * scale_factor,
    mean_mean_latency * scale_factor + 120)
print("")
print("Mean std dev latency:")
print(stdev_mean, stdev_mean * scale_factor)
print("")
print("Mean BOLD amygdala:")
print(mean_bold_amy)
print("")
print("Mean BOLD vmPFC:")
print(mean_bold_vmpfc)
print("")
print("Mean activation amygdala:")
print(mean_activ_amy)
print("")
print("Mean activation vmPFC:")
print(mean_activ_vmpfc)
print("\n\n")

fig = plt.figure()

mean_latencies = tuple(sample.mean_latency for sample in good)
bold_vmpfcs = tuple(sample.bold_vmpfc for sample in good)
bold_amys = tuple(sample.bold_amy for sample in good)
max_activ_amys = tuple(sample.max_activ_amy for sample in good)
max_activ_vmpfcs = tuple(sample.max_activ_vmpfc for sample in good)

# Plot latencies and BOLD
plt.subplot(221)
plt.title("Mean latencies")
plt.hist(mean_latencies)
plt.subplot(222)
plt.title("BOLD vmPFCs")
plt.hist(bold_vmpfcs)
plt.subplot(223)
plt.title("Max Amygdala Activations")
plt.hist(max_activ_amys)
plt.subplot(224)
plt.title("Max vmPFC Activations")
plt.hist(max_activ_vmpfcs)

try:
    plt.show()
except: pass

# Plot steady state estimates for valid and invalid samples
plt.subplot(131)

data = []
for i in range(3):
    data.append(tuple(s.steady for s in samples
        if s.activ_bounded(0.5) and s.cls(5, 67.0) == i))
plt.title("Amygdala steady state estimate (low max activity)")
plt.hist(data, 100, (0.0, 0.2), normed=1, histtype='bar', stacked=True)
plt.legend(["good", "invalid", "clipped"])


plt.subplot(132)

for i,color in enumerate(["green", "red", "yellow"]):
    data = tuple((s.steady, s.max_activ_amy, s.mean_latency)
        for s in samples
            if s.cls(5, 67.0) == i and s.steady < 1.0 and s.max_activ_amy < 0.9)

    x = [d[0] for d in data]
    y = [d[1] for d in data]
    plt.scatter(x, y)
plt.legend(["good", "invalid", "clipped"])
plt.title("Amygdala steady state estimate")

plt.subplot(133)
data = tuple((s.steady, s.max_activ_amy, s.mean_latency)
    for s in samples
        if s.steady < 1.0 and s.max_activ_amy < 0.9)

x = [d[0] for d in data]
y = [d[1] for d in data]
colors = [d[2] for d in data]
plt.scatter(x, y, c=colors, cmap='gist_heat')
plt.colorbar()
plt.title("Amygdala steady state estimate")

try:
    plt.show()
except: pass


fig = plt.figure()

# Perform linear regression to predict latencies in good samples
data = [s.to_3d() for s in good]
labels = mean_latencies

model = LinearRegression()
model.fit(data, labels)
print("%30s : %f %s %f %s" % ("Latency model",
    model.score(data, labels), model.coef_, model.intercept_,
    model.predict([(0.4, 1.0, 1.0), (0.6, 0.25, 2.5)])))

# Plot data points by class
ax = fig.add_subplot(131, projection='3d')
ax.set_title("Full dataset")

x, y, z = zip(*[s.to_3d() for s in good])
ax.scatter(x, y, z, c='yellow')

x, y, z = zip(*[s.to_3d() for s in invalid])
ax.scatter(x, y, z, c='red')

x, y, z = zip(*[s.to_3d() for s in clipped])
ax.scatter(x, y, z, c='green')

# Add spheres for healthy/PTSD
ax.scatter([0.4, 0.6], [1.0, 0.25], [1.0, 2.5], s=1000, c='cyan')


# Perform linear regression to predict vmPFC BOLD in good samples
data = [s.to_3d() for s in good]
labels = bold_vmpfcs

model = LinearRegression()
model.fit(data, labels)
print("%30s : %f %s %f" % ("vmPFC BOLD",
    model.score(data, labels), model.coef_, model.intercept_))

# Perform linear regression to predict amygdala BOLD in good samples
data = [s.to_3d() for s in good]
labels = bold_amys

model = LinearRegression()
model.fit(data, labels)
print("%30s : %f %s %f" % ("Amygdala BOLD",
    model.score(data, labels), model.coef_, model.intercept_))

# Perform linear regression to predict latency from steady state
data = []
labels = []
for s,lat in zip(good, mean_latencies):
    if s.activ_bounded(0.5):
        data.append((s.steady,))
        labels.append(lat)

model = LinearRegression()
model.fit(data, labels)
print("%30s : %f %s %f" % (
    "Latency < steady state < 0.5",
    model.score(data, labels), model.coef_, model.intercept_))

data = [(s.steady,) for s in good]
labels = mean_latencies

model = LinearRegression()
model.fit(data, labels)
print("%30s : %f %s %f" % (
    "Latency < steady state < 1.0",
    model.score(data, labels), model.coef_, model.intercept_))

# Perform logistic regression to separate valid, invalid, and clipped
data = tuple(s.to_3d() for s in samples)
labels = tuple(s.cls(5, 67.0) for s in samples)

x_train, x_test, y_train, y_test = \
    train_test_split(data, labels, test_size=0.25, random_state=0)

model = LogisticRegression()
model.fit(x_train, y_train)
print("%30s : %f\n%s\n%s" % (
    "Logistic model",
    model.score(data, labels), model.coef_, model.intercept_))

print("Confusion matrix")
print(confusion_matrix(
    np.array(y_test),
    np.array(model.predict(x_test))))

for i in (1,2):
    xx, yy, zz = np.mgrid[0:5:.05, 0:5:.05, 0:5:.05]
    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    probs = model.predict_proba(grid)[:, i]
    grid = [grid[i] for i in xrange(len(probs)) if probs[i] > 0.499 and probs[i] < 0.501]
    probs = [p for p in probs if p > 0.499 and p < 0.501]

    xx, yy, zz = zip(*grid)
    #ax.scatter(xx, yy, zz)
    ax.plot_trisurf(xx, yy, zz)



# Plot latency of good data points using heatmap
ax = fig.add_subplot(132, projection='3d')
ax.set_title("Latency Heatmap")

data = np.array([s.to_3d() for s in good])
labels = np.array([s.mean_latency for s in good])
x, y, z = zip(*data)
p = ax.scatter(x, y, z, c=labels, cmap='gist_heat')
fig.colorbar(p)

# Add spheres for healthy/PTSD
ax.scatter([0.4, 0.6], [1.0, 0.25], [1.0, 2.5], s=1000, c='cyan')


# Plot vmPFC bold of good data points using heatmap
ax = fig.add_subplot(133, projection='3d')
ax.set_title("vmPFC BOLD Heatmap")

data = np.array([s.to_3d() for s in good])
labels = np.array([s.bold_vmpfc for s in good])
x, y, z = zip(*data)
p = ax.scatter(x, y, z, c=labels, cmap='gist_heat')
fig.colorbar(p)

# Add spheres for healthy/PTSD
ax.scatter([0.4, 0.6], [1.0, 0.25], [1.0, 2.5], s=1000, c='cyan')

plt.tight_layout()

try:
    plt.show()
except: pass



# SEARCH PARAMS
restricted = True
if restricted:
    # Matches BOLD data
    latency_min = 67.0
    vmpfc_activ_bound = 0.75
    bins = 10

    latency_deviation_cutoff = 0.05
    vmpfc_bold_ratio_min = 0.0
    vmpfc_bold_ratio_max = 0.95
    amy_bold_ratio_min = 1.05
    amy_bold_ratio_max = 999.0
    wm_min = 0.0
    wm_max = 0.95

    # One candidate found for True, 0.5, 2.0
    # Support:   2   4
    #      0.3905      4.3801      0.4338     86.6000   5356.6968   5481.0843 
    #      0.5278      2.2265      0.6790    138.3000   4214.3106   7544.8573 
    #      --------------------------------------------------------------------------------
    #      1.3514      0.5083      1.5653      1.5970      0.7867      1.3765 
    #
    # One candidate found for vmPFC < 0.9, True, 0.5, 2.0
    # Support:   1   1
    #      0.9263      4.4987      2.7756     68.6000   1911.0284  13632.7362 
    #      1.3148      2.7078      4.1572    105.0000   1809.1418  19314.4489 
    #      --------------------------------------------------------------------------------
    #      1.4195      0.6019      1.4978      1.5306      0.9467      1.4168 
    #
    # One candidate found for vmPFC < 0.75, True, 0.5, 2.0
    # Support:   1   1
    #      0.3832      4.3156      0.5117     73.0000   4089.6016   5214.8041 
    #      0.5453      2.8845      0.8856    117.0000   3373.3101   7677.4128 
    #      --------------------------------------------------------------------------------
    #      1.4230      0.6684      1.7306      1.6027      0.8249      1.4722 
    #
    #
    # Two candidates found for vmPFC < 0.5, True, 0.5, 2.0
    # ================================================================================
    # Support:   1   1
    #      0.9263      4.4987      2.7756     68.6000   1911.0284  13632.7362 
    #      1.2552      2.2556      4.1704    107.0000   1710.7959  18454.6608 
    #      --------------------------------------------------------------------------------
    #      1.3552      0.5014      1.5025      1.5598      0.8952      1.3537 
    #
    # ================================================================================
    # Support:   1   1
    #      0.9263      4.4987      2.7756     68.6000   1911.0284  13632.7362 
    #      1.3148      2.7078      4.1572    105.0000   1809.1418  19314.4489 
    #      --------------------------------------------------------------------------------
    #      1.4195      0.6019      1.4978      1.5306      0.9467      1.4168 
    param_ratio_filter = False
    param_ratio_min = 0.5
    param_ratio_max = 2.0
else:
    # Full range
    latency_min = 67.0
    vmpfc_activ_bound = 1.0
    bins = 10

    latency_deviation_cutoff = 0.05
    vmpfc_bold_ratio_min = 0.0
    vmpfc_bold_ratio_max = 999.0
    amy_bold_ratio_min = 0.0
    amy_bold_ratio_max = 999.0
    wm_min = 0.0
    wm_max = 999.9

    param_ratio_filter = False
    param_ratio_min = 0.5
    param_ratio_max = 2.0



# Bin the data and search for increased latency and decreased vmPFC bold
# Use only data where max vmPFC activation stays below upper bound
bounded = [s for s in good
    if s.activ_bounded(vmpfc_activ_bound) and s.mean_latency > latency_min]
print("\nConsidering %d parameter sets..." % len(bounded))

data = np.array([[sample.s, sample.av, sample.va]
    for sample in bounded]).reshape(len(bounded), 3)
hist, binedges = np.histogramdd(data, normed=False, bins=bins)
bins_s, bins_av, bins_va = binedges

# Place each sample into a bin
indices = set()
binned = dict((index, []) for index in indices)
for sample in bounded:
    i_s = 0
    i_av = 0
    i_va = 0

    for i,b in enumerate(bins_s):
        if sample.s > b:
            i_s = i

    for i,b in enumerate(bins_av):
        if sample.av > b:
            i_av = i

    for i,b in enumerate(bins_va):
        if sample.va > b:
            i_va = i

    index = (i_s, i_av, i_va)
    indices.add(index)
    if index in binned:
        binned[index].append(sample)
    else:
        binned[index] = [sample]

# Compute mean of latencies, BOLD, count, and center for each bin
latencies = dict()
vmpfc_bolds = dict()
amy_bolds = dict()
counts = dict()
centers = dict()
for key,val in binned.iteritems():
    if len(val) > 0:
        latencies[key] = sum(x.mean_latency for x in val) / len(val)
        vmpfc_bolds[key] = sum(x.bold_vmpfc for x in val) / len(val)
        amy_bolds[key] = sum(x.bold_amy for x in val) / len(val)
        counts[key] = len(val)
        centers[key] = (
            np.mean([x.s for x in val]),
            np.mean([x.av for x in val]),
            np.mean([x.va for x in val]))

# Functions for processing parameters
def get_params(i):
    return centers[i]
def get_param_ratios(l, r):
    l_s, l_av, l_va = get_params(l)
    r_s, r_av, r_va = get_params(r)
    return (r_s / l_s, r_av / l_av, r_va / l_va)
def polarize_ratios(ratios):
    a,b,c = ratios
    a_adj = a - 1.0 if a > 1.0 else -(1.0 / a - 1.0)
    b_adj = b - 1.0 if b > 1.0 else -(1.0 / b - 1.0)
    c_adj = c - 1.0 if c > 1.0 else -(1.0 / c - 1.0)
    return a_adj, b_adj, c_adj
def get_wm_change(l, r):
    return (get_params(r)[1] + get_params(r)[2]) / (get_params(l)[1] + get_params(l)[2])


# Find parameter set pairs where:
# 1. Amygdala sensitivity increases
# 2. Saccade latency increases
# 3. vmPFC BOLD decreases
candidates = []
for l in indices:
    for r in indices:
        sensitivity_ratio = get_params(r)[0] / get_params(l)[0]
        latency_ratio = latencies[r] / latencies[l]
        vmpfc_bold_ratio = vmpfc_bolds[r] / vmpfc_bolds[l]
        amy_bold_ratio = amy_bolds[r] / amy_bolds[l]

        if latency_ratio > 1.0:
            candidates.append((l,r, latency_ratio, vmpfc_bold_ratio, amy_bold_ratio))

print("\nFound %d / %d candidate parameter pairs" % \
    (len(candidates), len(indices) ** 2 / 2 - len(indices)))


# Filter out candidates by deviation from ideal latency
target = 1.603125
candidates = [c for c in candidates if abs(target - c[2]) / target < latency_deviation_cutoff]
print("    ... %d within %f of target latency ratio" % (len(candidates), latency_deviation_cutoff))

# Filter out candidates with white matter increases
candidates = [c for c in candidates
    if get_wm_change(c[0], c[1]) < wm_max
        and get_wm_change(c[0], c[1]) > wm_min]
print("    ... %d with net white matter change %f < r < %f" % (
    len(candidates), wm_min, wm_max))

# Filter out candidates by BOLD ratio cutoff
candidates = [c for c in candidates
    if c[3] > vmpfc_bold_ratio_min and c[3] < vmpfc_bold_ratio_max
        and c[4] > amy_bold_ratio_min and c[4] < amy_bold_ratio_max]
print("    ... %d within BOLD ratio ranges" % len(candidates))

# Filter out any candidates < 0.5 or > 2.0 parameter ratios
if param_ratio_filter:
    candidates = [c for c in candidates
        if all(r > param_ratio_min and r < param_ratio_max
                for r in get_param_ratios(c[0], c[1]))]
    print("    ... %d within %f < r < %f for all parameter ratios" % (
        len(candidates), param_ratio_min, param_ratio_max))

raw_input("Continue...")

# Sort by BOLD ratio
candidates = sorted(candidates, key = lambda x: x[3])

print("\nParameter set candidates (%d/%d):" % (min(25, len(candidates)), len(candidates)))
for l,r,lat_rat,vmpfc_bold_rat,amy_bold_rat in candidates[:25]:
    print("=" * 80)
    print("Support: %3d %3d" % (counts[l], counts[r]))
    l_s, l_av, l_va = get_params(l)
    print(("%11.4f " * 6) % (l_s, l_av, l_va, latencies[l], vmpfc_bolds[l], amy_bolds[l]))

    r_s, r_av, r_va = get_params(r)
    print(("%11.4f " * 6) % (r_s, r_av, r_va, latencies[r], vmpfc_bolds[r], amy_bolds[r]))

    print("-" * 80)
    print(("%11.4f " * 6) % (r_s / l_s, r_av / l_av, r_va / l_va, lat_rat, vmpfc_bold_rat, amy_bold_rat))
    print("")

print("\nTotal candidates: %d" % len(candidates))

# Plot parameter ratios colored by BOLD ratios
param_ratios = []
vmpfc_bold_ratios = []
amy_bold_ratios = []
for l,r,lat_rat,vmpfc_bold_rat,amy_bold_rat in candidates:
    param_ratios.append(get_param_ratios(l, r))
    vmpfc_bold_ratios.append(vmpfc_bold_rat)
    amy_bold_ratios.append(amy_bold_rat)

fig = plt.figure()

ax = fig.add_subplot(121, projection='3d')
ax.set_title("Candidate Parameter Ratios")

x, y, z = zip(*param_ratios)
p = ax.scatter(x, y, z, c=vmpfc_bold_ratios, cmap='gist_heat')
fig.colorbar(p)

# Add sphere for mean
x = np.mean([r[0] for r in param_ratios])
y = np.mean([r[1] for r in param_ratios])
z = np.mean([r[2] for r in param_ratios])
print("Ratio mean: %9.4f %9.4f %9.4f" % (x,y,z))
ax.scatter([x], [y], [z], s=1000, c='cyan')

# Plot parameter ratios colored by polarized BOLD ratios
param_ratios = []
vmpfc_bold_ratios = []
amy_bold_ratios = []
for l,r,lat_rat,vmpfc_bold_rat,amy_bold_rat in candidates:
    param_ratios.append(polarize_ratios(get_param_ratios(l, r)))
    vmpfc_bold_ratios.append(vmpfc_bold_rat)
    amy_bold_ratios.append(amy_bold_rat)

ax = fig.add_subplot(122, projection='3d')
ax.set_title("Polarized Ratios")

x, y, z = zip(*param_ratios)
p = ax.scatter(x, y, z, c=vmpfc_bold_ratios, cmap='gist_heat')
fig.colorbar(p)

# Add sphere for mean
x = np.mean([r[0] for r in param_ratios])
y = np.mean([r[1] for r in param_ratios])
z = np.mean([r[2] for r in param_ratios])
print("Ratio mean (polarized): %9.4f %9.4f %9.4f" % (x,y,z))
ax.scatter([x], [y], [z], s=1000, c='cyan')


try:
    plt.show()
except: pass


# Divide ratios by greater than or less than one (8 bins)
quadrants = dict()
for ((a,b,c),vmpfc_br),amy_br in zip(zip(param_ratios, vmpfc_bold_ratios), amy_bold_ratios):
    key = []
    for k in a,b,c:
        if k < -0.05:
            key.append(0)
        elif k > 0.05:
            key.append(2)
        else:
            key.append(1)

    try:
        quadrants[tuple(key)].append((a,b,c,vmpfc_br,amy_br))
    except KeyError:
        quadrants[tuple(key)] = [(a,b,c,vmpfc_br,amy_br)]

print("\ns/av/va <=>  Count    sens        av        va     vmPFC       amy      mean")
for k,v in sorted(quadrants.iteritems(), key = lambda x: len(x[1])):
    a, b, c, vmpfc_br, amy_br = zip(*v)
    ma, mb, mc, mvbr, mabr = np.mean(a), np.mean(b), np.mean(c), np.mean(vmpfc_br), np.mean(amy_br)
    print(("%s %6d " + "%9.4f " * 6) % (k, len(v), ma, mb, mc, mvbr, mabr, np.linalg.norm((ma, mb, mc))))


# Plot vectors for candidate pairs
vectors = []
for l,r,lat_rat,vmpfc_bold_rat,amy_bold_rat in candidates:
    vectors.append(get_params(l) + \
        tuple(x - y for x,y in zip(get_params(r), get_params(l))))

fig = plt.figure()

ax = fig.add_subplot(121, projection='3d')
ax.set_title("Candidate Parameter Vectors")

x, y, z, u, v, w = zip(*vectors)
ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.05)

# Plot vectors for candidate pairs
vectors = []
for l,r,lat_rat,vmpfc_bold_rat,amy_bold_rat in candidates:
    vectors.append(get_params(l) + \
        tuple(x - y for x,y in zip(get_params(r), get_params(l))))

ax = fig.add_subplot(122, projection='3d')
ax.set_title("Candidate Parameter Vectors")

x, y, z, u, v, w = zip(*vectors)
ax.quiver([0.0 for _ in x], [0.0 for _ in y], [0.0 for _ in z], u, v, w, arrow_length_ratio=0.05)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_zlim(-2.5, 2.5)

try:
    plt.show()
except: pass
