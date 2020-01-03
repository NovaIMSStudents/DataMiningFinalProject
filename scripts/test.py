import numpy as np
from numpy.linalg import svd
from scipy.spatial import distance_matrix
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

test1 = np.array([[34, 37, 44, 27, 19, 39, 74, 44, 27, 61, 12, 65, 69, 22, 14, 21],
                  [18, 33, 47, 24, 14, 38, 66, 41, 36, 72, 15, 62, 63, 31, 12, 18],
                  [32, 43, 36, 12, 21, 51, 75, 33, 23, 60, 24, 68, 85, 18, 13, 14],
                  [13, 31, 55, 29, 15, 62, 74, 43, 28, 73, 8, 59, 54, 32, 19, 20],
                  [8, 28, 34, 24, 17, 68, 75, 34, 25, 70, 16, 56, 72, 31, 14, 11],
                  [9, 34, 43, 25, 18, 68, 84, 25, 32, 76, 14, 69, 64, 27, 11, 18],
                  [15, 20, 28, 18, 19, 65, 82, 34, 29, 89, 11, 47, 74, 18, 22, 17],
                  [18, 14, 40, 25, 21, 60, 70, 15, 37, 80, 15, 65, 68, 21, 25, 9],
                  [19, 18, 41, 26, 19, 58, 64, 18, 38, 78, 15, 65, 72, 20, 20, 11],
                  [13, 29, 49, 31, 16, 61, 73, 36, 29, 69, 13, 63, 58, 18, 20, 25],
                  [17, 34, 43, 29, 14, 62, 64, 26, 26, 71, 26, 78, 64, 21, 18, 12],
                  [13, 22, 43, 16, 11, 70, 68, 46, 35, 57, 30, 71, 57, 19, 22, 20],
                  [16, 18, 56, 13, 27, 67, 61, 43, 20, 63, 14, 43, 67, 34, 41, 23],
                  [15, 21, 66, 21, 19, 50, 62, 50, 24, 68, 14, 40, 58, 31, 36, 26],
                  [19, 17, 70, 12, 28, 53, 72, 39, 22, 71, 11, 40, 67, 25, 41, 17]])

test2 = np.array([[12, 2, 3, 2],
                 [10, 3, 7, 4],
                 [25, 10, 12, 4],
                 [18, 24, 33, 13],
                 [10, 6, 7, 2]])

input_matrix = np.array([[4256., 0., 0., 343., 1380., 2193., 340., 1237., 3019., 3157., 1099.],
                         [0., 4344., 0., 377., 1440., 2211., 316., 1435., 2909., 3327., 1017.],
                         [0., 0., 1647., 536., 678., 395., 38., 325., 1322., 1539., 108.],
                         [343., 377., 536., 1256., 0., 0., 0., 372., 884., 1167., 89.],
                         [1380., 1440., 678., 0., 3498., 0., 0., 1020., 2478., 2993., 505.],
                         [2193., 2211., 395., 0., 0., 4799., 0., 1398., 3401., 3455., 1344.],
                         [340., 316., 38., 0., 0., 0., 694., 207., 487., 408., 286.],
                         [1237., 1435., 325., 372., 1020., 1398., 207., 2997., 0., 2572., 425.],
                         [3019., 2909., 1322., 884., 2478., 3401., 487., 0., 7250., 5451., 1799.],
                         [3157., 3327., 1539., 1167., 2993., 3455., 408., 2572., 5451., 8023., 0.],
                         [1099., 1017., 108., 89., 505., 1344., 286., 425., 1799., 0., 2224.]])

# FORM INPUT MATRIX
cat_labels = cat_df.columns
    if mca:
        Z = cat_df.values
        input_matrix = Z.T@Z
    else:
        input_matrix = mca_df.loc[:,row_cats].values.T @ mca_df.loc[:,[i for i in mca_df.columns if i not in row_cats]].values

# FROM CONTIGENCY TABLE TO CHI-SQ VALUES
n = np.sum(input_matrix)  # grand total
P = input_matrix / n  # correspondence matrix/ frequency table
rt = np.expand_dims(np.sum(P, axis=1), axis=1)  # row totals
ct = np.expand_dims(np.sum(P, axis=0), axis=1)  # column totals
RP = np.diag(rt[:, 0] ** -1) @ P  # row profile table
CP = np.diag(ct[:, 0] ** -1) @ P.T  # columns profile table
E = rt @ ct.T  # independence model/ expected proportions
R = P - E  # residuals/ difference between observed and expected proportions
CS = (R ** 2) / E  # chi-sq values

# CHI-SQ TEST OF INDEPENDENCE OF COLUMNS AND ROWS
Csqs = n * np.sum(CS)  # Chi-sq statistic
inertia = Csqs / n  # total inertia of Fm
df = (input_matrix.shape[0] - 1) * (input_matrix.shape[1] - 1)  # df for test of independence
p_value = 1 - chi2.cdf(Csqs, df)  # p-value for independence test of employees and smoking habits

# COMPUTING THE CHI-SQ DISTANCES (WILL BE REPRESENTED IN THE PECEPTUAL MAP)
# ROW-WISE
norm_row = RP @ np.diag(ct[:, 0] ** -0.5)  # row profiles weighted on column masses for euclid dist
CS_rdist = distance_matrix(norm_row, norm_row)  # chisq distances between rows
# COLUMN-WISE
norm_col = CP @ np.diag(rt[:, 0] ** -0.5)  # column profiles weighted on row masses for euclid dist
CS_cdist = distance_matrix(norm_col, norm_col)  # chisq distances between columns

# DECOMPOSING THE STANDARDIZED RESIDUALS MATRIX
I = R / E  # indexed residuals - tells us the association between 2 categories comparing to what would be expected
Z = I * (E ** 0.5)  # standardized residuals (sqrt of CS)
U, s, v = svd(Z, full_matrices=False)  # singular values decomposition of Z
V = v.T
S = np.diag(s)

# DETERMINING NUMBER OF DIMENSIONS TO RETAIN
max_dim = np.min(input_matrix.shape) - 1  # maximum number of dimensions
inrt_dim = np.diag(S)[:max_dim] ** 2  # inertia retained by each dim
perc_inrt = inrt_dim / inertia  # proportion inertia retained by each dim
cumul_inrt = np.cumsum(perc_inrt)  # cumulative inertia retained by each dim
expect_inrt = inertia / max_dim  # expected inertia for each dim
# Figure
sns.set(style="dark")
fig, ax1 = plt.subplots(figsize=(12, 7))
# Traces
x = np.arange(1, max_dim+1)
ax2 = ax1.twinx()
ax2.plot(x, cumul_inrt, color='r', marker='o', linewidth=2.5, label='Cumulative Proportion of Inertia')
ax1.bar(x, height=inrt_dim, label='Inertia by Dimension')
ax1.axhline(expect_inrt, color='k', linestyle="--", label="Expect Inertia by Dimension")
# Layout
ax1.set_ylabel('Inertia')
ax2.set_ylabel('Cumulative Proportion of Inertia')
ax1.set_xlabel('Dimensions')
ax1.set_title('Inertia Over Dimensions', fontsize=18, pad=20)
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
plt.legend(handles1 + handles2, labels1 + labels2, loc='upper left')  # build legend
plt.xticks(x, list(map(lambda i: str(i), list(x))))
# Show figure
plt.show()

ret_dim = 2  # number of dimensions to retain

# COMPUTING ROW SCORES
row_scores = np.diag(rt[:, 0] ** -0.5) @ U  # standard coordinates
row_scores = (row_scores @ S)[:, :max_dim]  # applying principal normalization
ret_rscr = row_scores[:, 0:ret_dim]

# COMPUTING COLUMN SCORES
col_scores = np.diag(ct[:, 0] ** -0.5) @ V  # standard coordinates
col_scores = col_scores[:, :max_dim]  # no normalization (row principal normalization)
# col_scores = (col_scores @ S)[:, :max_dim]  # applying principal normalization
ret_cscr = col_scores[:, 0:ret_dim]

# VERIFYING THAT CHI-SQ DISTANCES ARE REPRODUCED BY SCORES
# ROW-WISE
recons_rdist = distance_matrix(row_scores, row_scores)  # reconstructed distances using row scores
np.round(CS_rdist - recons_rdist, 2)  # do the row score distances match the row profile weighted distances?
# yes -> principal coordinates | no -> standard coordinates
# COLUMN-WISE
recons_cdist = distance_matrix(col_scores, col_scores)  # reconstructed distances using column scores
np.round(CS_cdist - recons_cdist, 2)  # do the column score distances match the column profile weighted distances?
# yes -> principal coordinates | no -> standard coordinates

# RECONSTRUCTING THE INDEXED RESIDUALS MATRIX
recons_I = row_scores @ col_scores.T  # rebuilt associations is given by inner product between column and row vectors
# inner product can be written as function of the cosine of the angle between 2 vectors, therefore the visual
# analysis of the angle
np.round(recons_I - I, 2)  # are the indexed residuals(associations between col and rows) rebuilt perfectly?
# yes -> able to interpret relationship between rows and columns | no -> not able to interpret relationship between
# rows and columns

# PM REPRESENTATION QUALITY ASSESSMENT
row_inert = np.sum(CS, axis=1)  # inertia by rows
col_inert = np.sum(CS, axis=0)  # inertia by columns
row_mass = rt  # mass of rows
col_mass = ct  # mass of columns
eigval = (S @ S.T)[:max_dim, :max_dim]  # Getting eigenvalues (inertia accounted by each dimension)
ret_var = np.sum(np.diag(eigval)[:ret_dim]) / np.sum(np.diag(eigval))  # % inertia retained by 2 dimensions - 99.8%
base = np.append(row_scores, col_scores, axis=0) ** 2
sq_cosines = np.diag(1 / np.sum(base, axis=1)) @ base
# squared cosines: proportion of categories variation accounted by each dim (analogous to communalities in FA)
abs_ctrb = np.diag(np.append(row_mass, col_mass, axis=0)[:, 0]) @ base @ np.diag(np.diag(eigval) ** -1)
# absolute contributions: proportion of dimension's inertia attributable to each category
quality = np.sum(sq_cosines[:, :ret_dim], axis=1)  # representation of each category overall

summary_df = pd.DataFrame(np.concatenate((np.expand_dims(np.append(row_inert/inertia, col_inert/inertia), axis=1),
                                          np.expand_dims(np.append(row_mass, col_mass), axis=1),
                                          np.expand_dims(quality, axis=1),
                                          np.append(row_scores[:, :ret_dim], col_scores[:, :ret_dim], axis=0),
                                          sq_cosines[:, :ret_dim],
                                          abs_ctrb[:, :ret_dim]), axis=1),
                          columns=["Prop inertia","Mass","Quality","D1","D2","SqC1","SqC2","Ctr1","Ctr2"])
                          # index=[])

#PLOTTING THE PERCEPTUAL MAP (PM)
# Data
plot_cord = pd.DataFrame(np.append(ret_rscr, ret_cscr, axis=0), columns=["Dim1", "Dim2"])

# Figure
sns.set()
fig = plt.figure(figsize=(12, 7))

sns.scatterplot(x="Dim1", y="Dim2", data=plot_cord)

# Layout
plt.axhline(y=0, color='k', linewidth=1)
plt.axvline(x=0, color='k', linewidth=1)
plt.title("Cluster Profiles", fontsize=18)
plt.xlabel("Dim1 - {0:.2f}%".format(perc_inrt[0]*100))
plt.ylabel("Dim2 - {0:.2f}%".format(perc_inrt[1]*100))

plt.show()
