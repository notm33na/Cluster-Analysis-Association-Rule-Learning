import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
import os, warnings
warnings.filterwarnings('ignore')


OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
# 0. Loading Data
df = pd.read_csv("Country-data.csv")
print(f"Shape: {df.shape}")
print(df.head(3))

FEATURES = ['child_mort','exports','health','imports',
            'income','inflation','life_expec','total_fer','gdpp']

X_raw = df[FEATURES].copy()
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# 2.1  DISTANCE MATRIX HEATMAP

dist_matrix = pairwise_distances(X, metric="euclidean")
dist_df = pd.DataFrame(dist_matrix, index=df['country'], columns=df['country'])

fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(
    dist_df, cmap="YlOrRd", square=True,
    xticklabels=False, yticklabels=False,
    cbar_kws={"label": "Euclidean Distance", "shrink": 0.7},
    ax=ax
)
ax.set_title(
    "Pairwise Euclidean Distance Matrix — Country Economic Indicators (167 countries)\n"
    "Darker = more similar economic profile   |   Lighter = more different",
    fontsize=13, fontweight='bold', pad=15
)
ax.set_xlabel("Countries", fontsize=11)
ax.set_ylabel("Countries", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/01_distance_matrix_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ 2.1 Distance matrix heatmap saved.")

# Print 6×6 snippet
print("\nDistance Matrix Snippet (first 6 × 6 countries):")
print(dist_df.iloc[:6, :6].round(3).to_string())

k_range = range(2, 11)
inertias, silhouettes = [], []

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow
axes[0].plot(list(k_range), inertias, 'o-', color='#2196F3',
             linewidth=2.5, markersize=9, markerfacecolor='white',
             markeredgewidth=2.5)
axes[0].axvline(x=3, color='#E53935', linestyle='--', linewidth=2, label='Optimal k = 3')
axes[0].fill_between(list(k_range), inertias, alpha=0.08, color='#2196F3')
axes[0].set_title("Elbow Method\nWithin-Cluster Sum of Squares (WCSS)",
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel("Number of Clusters (k)", fontsize=11)
axes[0].set_ylabel("Inertia (WCSS)", fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(list(k_range))

# Silhouette
axes[1].plot(list(k_range), silhouettes, 's-', color='#43A047',
             linewidth=2.5, markersize=9, markerfacecolor='white',
             markeredgewidth=2.5)
axes[1].axvline(x=3, color='#E53935', linestyle='--', linewidth=2, label='Optimal k = 3')
axes[1].fill_between(list(k_range), silhouettes, alpha=0.08, color='#43A047')
axes[1].set_title("Silhouette Score\nHigher = better-separated clusters",
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel("Number of Clusters (k)", fontsize=11)
axes[1].set_ylabel("Silhouette Score", fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(list(k_range))

plt.suptitle("Determining Optimal Number of Clusters — Country Economic Indicators",
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/02_elbow_silhouette.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ 2.3 Elbow + Silhouette plot saved.")

for k, inn, sil in zip(k_range, inertias, silhouettes):
    print(f"  k={k}  inertia={inn:.1f}  silhouette={sil:.4f}")
# 2.4  CLUSTER VISUALISATION  — PCA 2-D  for k = 2, 3, 4, 5
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
var1, var2 = pca.explained_variance_ratio_ * 100

PALETTE = ['#E63946', '#2196F3', '#43A047', '#FF9800', '#9C27B0']
LABEL_MAP = {        # for k=3 only (used in final model)
    0: "Developing Nations",
    1: "Developed Nations",
    2: "Fragile / Least Developed"
}

k_values = [2, 3, 4, 5]
fig, axes = plt.subplots(2, 2, figsize=(15, 13))
axes = axes.flatten()

for ax, k in zip(axes, k_values):
    km = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42)
    labels = km.fit_predict(X)
    for c in range(k):
        mask = labels == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=PALETTE[c], s=40, alpha=0.75,
                   edgecolors='white', linewidths=0.4,
                   label=f"Cluster {c+1}")
    cents_pca = pca.transform(km.cluster_centers_)
    ax.scatter(cents_pca[:, 0], cents_pca[:, 1],
               c='black', marker='X', s=160, zorder=6, label='Centroids')
    sil = silhouette_score(X, labels)
    ax.set_title(f"K-Means  (k = {k})   |   Silhouette = {sil:.3f}",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=9)
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=9)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.8)
    ax.grid(True, alpha=0.2)

plt.suptitle("K-Means Cluster Maps — Country Economic Indicators\n(PCA 2-D Projection)",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/03_cluster_maps_all_k.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ 2.4 Cluster maps (k=2,3,4,5) saved.")

# FINAL MODEL: k = 3

OPTIMAL_K = 3
km_final = KMeans(n_clusters=OPTIMAL_K, init='k-means++', n_init=20, random_state=42)
raw_labels = km_final.fit_predict(X)

# Re-order cluster labels by mean GDP (ascending) so naming is consistent
cluster_gdpp = {}
for c in range(OPTIMAL_K):
    cluster_gdpp[c] = df.loc[raw_labels == c, 'gdpp'].mean()
# sort: 0=lowest GDP, 1=middle, 2=highest
order = sorted(cluster_gdpp, key=cluster_gdpp.get)
remap = {old: new for new, old in enumerate(order)}
df['Cluster'] = [remap[l] + 1 for l in raw_labels]   # 1-indexed

CLUSTER_NAMES = {1: "Fragile / Least Developed",
                 2: "Emerging / Developing",
                 3: "Developed / High-Income"}

print("\nCluster Sizes:")
print(df['Cluster'].value_counts().sort_index())
print("\nCountries per Cluster (sample):")
for c in [1, 2, 3]:
    countries = df[df['Cluster'] == c]['country'].tolist()
    print(f"  Cluster {c} ({CLUSTER_NAMES[c]}): {', '.join(countries[:8])}...")

# 2.5  CSV OUTPUT
df.to_csv(f"{OUT}/country_clustered.csv", index=False)
print("\n✓ 2.5 Clustered CSV saved.")
print("\nCSV Snippet (first 15 rows):")
print(df[['country','child_mort','income','gdpp','life_expec','Cluster']].head(15).to_string(index=False))

# 2.6  PER-CLUSTER VISUALISATION — Box plots of key features
KEY_FEATS = ['child_mort', 'income', 'gdpp', 'life_expec',
             'inflation', 'health']
FEAT_LABELS = {
    'child_mort':  'Child Mortality\n(per 1000 births)',
    'income':      'Income per Person\n(USD)',
    'gdpp':        'GDP per Capita\n(USD)',
    'life_expec':  'Life Expectancy\n(years)',
    'inflation':   'Inflation Rate\n(%)',
    'health':      'Health Expenditure\n(% of GDP)'
}
COLORS = ['#E63946', '#FF9800', '#2196F3']   # red=fragile, orange=emerging, blue=developed

fig, axes = plt.subplots(2, 3, figsize=(17, 11))
axes = axes.flatten()

for ax, feat in zip(axes, KEY_FEATS):
    data_to_plot = [df[df['Cluster'] == c][feat].values for c in [1, 2, 3]]
    bp = ax.boxplot(
        data_to_plot, patch_artist=True,
        medianprops=dict(color='black', linewidth=2.2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markersize=4, alpha=0.5),
        widths=0.55
    )
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax.set_title(FEAT_LABELS[feat], fontsize=11, fontweight='bold')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Cluster 1\n(Fragile)', 'Cluster 2\n(Emerging)', 'Cluster 3\n(Developed)'],
                       fontsize=8.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

patches = [mpatches.Patch(color=c, alpha=0.72, label=f"Cluster {i} — {CLUSTER_NAMES[i]}")
           for i, c in zip([1,2,3], COLORS)]
fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.04), frameon=True)
plt.suptitle("Per-Cluster Feature Distributions — K-Means (k = 3)\nCountry Economic Indicators Dataset",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/04_per_cluster_boxplots.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ 2.6 Per-cluster box plots saved.")

# ── Cluster means table
print("\nCluster Means (rounded):")
means = df.groupby('Cluster')[FEATURES].mean().round(2)
means.index = [f"C{i} — {CLUSTER_NAMES[i]}" for i in means.index]
print(means.T.to_string())

# Boonuss — Annotated PCA scatter for k=3 with country labels for notable ones

ANNOTATE = ['United States','Luxembourg','Norway','Switzerland',
            'Somalia','Niger','Mali','Afghanistan',
            'China','India','Brazil','Nigeria']

fig, ax = plt.subplots(figsize=(13, 9))
for c in [1, 2, 3]:
    mask = df['Cluster'].values == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=COLORS[c-1], s=55, alpha=0.78,
               edgecolors='white', linewidths=0.5,
               label=f"Cluster {c} — {CLUSTER_NAMES[c]}", zorder=3)

cents_pca = pca.transform(km_final.cluster_centers_)
ax.scatter(cents_pca[:, 0], cents_pca[:, 1],
           c='black', marker='X', s=220, zorder=7, label='Centroids')

for _, row in df.iterrows():
    if row['country'] in ANNOTATE:
        idx = df.index.get_loc(_)
        ax.annotate(row['country'], (X_pca[idx, 0], X_pca[idx, 1]),
                    fontsize=7.5, ha='left', va='bottom',
                    xytext=(5, 4), textcoords='offset points',
                    arrowprops=dict(arrowstyle='-', lw=0.7, color='grey'))

ax.set_title("K-Means Clustering (k = 3) — Country Economic Indicators\n"
             "PCA 2-D Projection with Cluster Interpretation",
             fontsize=13, fontweight='bold')
ax.set_xlabel(f"PC1 ({var1:.1f}% variance explained)", fontsize=11)
ax.set_ylabel(f"PC2 ({var2:.1f}% variance explained)", fontsize=11)
ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/05_final_k3_annotated.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Annotated final cluster map saved.")

print(f"\n All outputs saved to: {OUT}")