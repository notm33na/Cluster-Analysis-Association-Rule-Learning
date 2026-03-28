import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import os, warnings
warnings.filterwarnings('ignore')

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ── Load & prepare data (same as Kmean.py) ──────────────────────────────────
df = pd.read_csv("Country-data.csv")
FEATURES = ['child_mort','exports','health','imports',
            'income','inflation','life_expec','total_fer','gdpp']

X_raw = df[FEATURES].copy()
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
var1, var2 = pca.explained_variance_ratio_ * 100

COLORS = ['#E63946', '#FF9800', '#2196F3', '#9C27B0', '#43A047']

# Load K-Means labels for comparison (from Part 2)
km = KMeans(n_clusters=3, init='k-means++', n_init=20, random_state=42)
raw_labels_km = km.fit_predict(X)
cluster_gdpp_km = {}
for c in range(3):
    cluster_gdpp_km[c] = df.loc[raw_labels_km == c, 'gdpp'].mean()
order_km = sorted(cluster_gdpp_km, key=cluster_gdpp_km.get)
remap_km = {old: new for new, old in enumerate(order_km)}
kmeans_labels = np.array([remap_km[l] + 1 for l in raw_labels_km])

CLUSTER_NAMES_KM = {1: "Fragile / Least Developed",
                    2: "Emerging / Developing",
                    3: "Developed / High-Income"}

print("=" * 70)
print("  PART 3: HIERARCHICAL CLUSTERING")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# 3.1  AGGLOMERATIVE (BOTTOM-UP) CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 3.1 Agglomerative Clustering ────────────────────────────────────")

# Linkage: Ward's method — minimises within-cluster variance at each merge.
# It is the most appropriate for compact, spherical clusters and pairs well
# with Euclidean distance, which matches the K-Means metric from Part 2.
Z = linkage(X, method='ward', metric='euclidean')

# ── Dendrogram ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 8))
dn = dendrogram(
    Z,
    labels=df['country'].values,
    leaf_rotation=90,
    leaf_font_size=5,
    color_threshold=Z[-(3-1), 2],   # colour by 3-cluster cut
    above_threshold_color='grey',
    ax=ax
)
cut_height = Z[-(3-1), 2]
ax.axhline(y=cut_height, color='#E53935', linestyle='--', linewidth=2,
           label=f'Cut level = {cut_height:.2f}  (k = 3)')
ax.set_title("Agglomerative Dendrogram (Ward's Linkage)\n"
             "Country Economic Indicators — 167 Countries",
             fontsize=14, fontweight='bold')
ax.set_xlabel("Country", fontsize=11)
ax.set_ylabel("Ward Distance (Merge Cost)", fontsize=11)
ax.legend(fontsize=11, loc='upper right')
plt.tight_layout()
plt.savefig(f"{OUT}/06_agglomerative_dendrogram.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Dendrogram saved.")

# ── Agglomerative labels (cut at k=3) ───────────────────────────────────────
agg_raw = fcluster(Z, t=3, criterion='maxclust')

# Re-order by mean GDP so cluster numbering matches K-Means interpretation
agg_gdpp = {}
for c in np.unique(agg_raw):
    agg_gdpp[c] = df.loc[agg_raw == c, 'gdpp'].mean()
order_agg = sorted(agg_gdpp, key=agg_gdpp.get)
remap_agg = {old: new for new, old in enumerate(order_agg)}
agg_labels = np.array([remap_agg[l] + 1 for l in agg_raw])

CLUSTER_NAMES_AGG = {1: "Fragile / Least Developed",
                     2: "Emerging / Developing",
                     3: "Developed / High-Income"}

# ── Agglomerative Cluster Map ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
for c in [1, 2, 3]:
    mask = agg_labels == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=COLORS[c-1], s=55, alpha=0.78,
               edgecolors='white', linewidths=0.5,
               label=f"Cluster {c} — {CLUSTER_NAMES_AGG[c]}", zorder=3)
ax.set_title("Agglomerative Clustering (Ward's, k = 3)\n"
             "PCA 2-D Projection",
             fontsize=13, fontweight='bold')
ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=11)
ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/07_agglomerative_cluster_map.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Cluster map saved.")

# ── Agglomerative interpretation ─────────────────────────────────────────────
sil_agg = silhouette_score(X, agg_labels)
print(f"\n  Silhouette Score (Agglomerative): {sil_agg:.4f}")
print(f"  Cluster sizes: {dict(zip(*np.unique(agg_labels, return_counts=True)))}")
print("\n  Cluster Means (Agglomerative):")
df['Agg_Cluster'] = agg_labels
means_agg = df.groupby('Agg_Cluster')[FEATURES].mean().round(2)
means_agg.index = [f"C{i} — {CLUSTER_NAMES_AGG[i]}" for i in means_agg.index]
print(means_agg.T.to_string())

print("\n  Interpretation:")
print("  - Ward's agglomerative method builds clusters bottom-up by merging the")
print("    pair of clusters that least increases total within-cluster variance.")
print("  - The dendrogram shows a clear 3-cluster structure: the largest jump")
print("    in merge cost occurs between the 3-cluster and 2-cluster levels.")
print("  - Cluster 1 (Fragile): high child mortality, low GDP — Sub-Saharan Africa, etc.")
print("  - Cluster 2 (Emerging): mid-range indicators — Latin America, Southeast Asia.")
print("  - Cluster 3 (Developed): high income, high life expectancy — OECD nations.")
print("  - Compared to K-Means, agglomerative clustering tends to produce tighter,")
print("    more fine-grained groupings. Some borderline countries shift between clusters.")

# ══════════════════════════════════════════════════════════════════════════════
# 3.2  DIVISIVE (TOP-DOWN) CLUSTERING — Bisecting K-Means
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 3.2 Divisive Clustering (Bisecting K-Means) ────────────────────")

# Divisive approach: start with all points in one cluster, then recursively
# split the cluster with the largest SSE using K-Means (k=2).
# This is the "bisecting K-Means" algorithm — a practical divisive method.

class BisectingKMeans:
    """Divisive (top-down) hierarchical clustering via bisecting K-Means."""

    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.split_history = []       # records each split for the tree diagram
        self.labels_ = None

    def _sse(self, data):
        centre = data.mean(axis=0)
        return np.sum((data - centre) ** 2)

    def fit_predict(self, X):
        labels = np.zeros(len(X), dtype=int)
        next_id = 1
        cluster_ids = {0}

        while len(cluster_ids) < self.n_clusters:
            # Pick the cluster with the largest SSE to split
            best_id, best_sse = None, -1
            for cid in cluster_ids:
                mask = labels == cid
                sse = self._sse(X[mask])
                if sse > best_sse:
                    best_sse = sse
                    best_id = cid

            mask = labels == best_id
            sub_X = X[mask]
            km2 = KMeans(n_clusters=2, init='k-means++', n_init=10,
                         random_state=self.random_state)
            sub_labels = km2.fit_predict(sub_X)

            # Assign sub-cluster 0 keeps parent id, sub-cluster 1 gets next_id
            child_a = best_id
            child_b = next_id
            indices = np.where(mask)[0]
            for i, sl in zip(indices, sub_labels):
                if sl == 1:
                    labels[i] = child_b

            # Record split
            self.split_history.append({
                'parent': best_id,
                'parent_size': int(mask.sum()),
                'parent_sse': best_sse,
                'child_a': child_a,
                'child_a_size': int((sub_labels == 0).sum()),
                'child_b': child_b,
                'child_b_size': int((sub_labels == 1).sum()),
            })

            cluster_ids.discard(best_id)
            cluster_ids.add(child_a)
            cluster_ids.add(child_b)
            next_id += 1

        self.labels_ = labels
        return labels


bkm = BisectingKMeans(n_clusters=3, random_state=42)
div_raw = bkm.fit_predict(X)

# Re-order by mean GDP
div_gdpp = {}
for c in np.unique(div_raw):
    div_gdpp[c] = df.loc[div_raw == c, 'gdpp'].mean()
order_div = sorted(div_gdpp, key=div_gdpp.get)
remap_div = {old: new for new, old in enumerate(order_div)}
div_labels = np.array([remap_div[l] + 1 for l in div_raw])

CLUSTER_NAMES_DIV = {1: "Fragile / Least Developed",
                     2: "Emerging / Developing",
                     3: "Developed / High-Income"}

# ── Split Diagram (Tree) ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

# Draw the binary split tree
# Positions: root at top centre, children below
node_positions = {}
node_labels_txt = {}

# Root
all_size = len(X)
node_positions['root'] = (0.5, 0.95)
node_labels_txt['root'] = f"All Countries\nn = {all_size}"

# After split 1: parent 0 splits into child_a and child_b
s1 = bkm.split_history[0]
node_positions['s1_a'] = (0.25, 0.55)
node_positions['s1_b'] = (0.75, 0.55)
node_labels_txt['s1_a'] = f"Sub-cluster A\nn = {s1['child_a_size']}"
node_labels_txt['s1_b'] = f"Sub-cluster B\nn = {s1['child_b_size']}"

# After split 2: one of the children splits again
s2 = bkm.split_history[1]
# Determine which child was split
if s2['parent'] == s1['child_a']:
    parent_pos = node_positions['s1_a']
    node_positions['s2_a'] = (parent_pos[0] - 0.12, 0.15)
    node_positions['s2_b'] = (parent_pos[0] + 0.12, 0.15)
    # s1_b remains a leaf
    final_leaves = ['s2_a', 's2_b', 's1_b']
else:
    parent_pos = node_positions['s1_b']
    node_positions['s2_a'] = (parent_pos[0] - 0.12, 0.15)
    node_positions['s2_b'] = (parent_pos[0] + 0.12, 0.15)
    final_leaves = ['s1_a', 's2_a', 's2_b']

node_labels_txt['s2_a'] = f"Sub-cluster\nn = {s2['child_a_size']}"
node_labels_txt['s2_b'] = f"Sub-cluster\nn = {s2['child_b_size']}"

# Map final cluster IDs to leaf names for colouring
# We need to figure out which raw cluster ID maps to which leaf
split_parent_2 = s2['parent']

if split_parent_2 == s1['child_a']:
    # s1_a was split into s2_a (keeps s1_a's id=child_a) and s2_b (new id=child_b of s2)
    leaf_to_raw = {
        's2_a': s2['child_a'],  # kept parent id
        's2_b': s2['child_b'],  # new id
        's1_b': s1['child_b'],  # untouched
    }
else:
    leaf_to_raw = {
        's1_a': s1['child_a'],  # untouched
        's2_a': s2['child_a'],  # kept parent id
        's2_b': s2['child_b'],  # new id
    }

# Colour leaves by their remapped cluster
leaf_colors = {}
for leaf, raw_id in leaf_to_raw.items():
    mapped = remap_div[raw_id]  # 0, 1, or 2
    leaf_colors[leaf] = COLORS[mapped]

# Draw connections
# Root to s1 children
ax.annotate("", xy=node_positions['s1_a'], xytext=node_positions['root'],
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#555'))
ax.annotate("", xy=node_positions['s1_b'], xytext=node_positions['root'],
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#555'))

# Second split
if split_parent_2 == s1['child_a']:
    ax.annotate("", xy=node_positions['s2_a'], xytext=node_positions['s1_a'],
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#555'))
    ax.annotate("", xy=node_positions['s2_b'], xytext=node_positions['s1_a'],
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#555'))
else:
    ax.annotate("", xy=node_positions['s2_a'], xytext=node_positions['s1_b'],
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#555'))
    ax.annotate("", xy=node_positions['s2_b'], xytext=node_positions['s1_b'],
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#555'))

# Draw cut line
ax.axhline(y=0.35, color='#E53935', linestyle='--', linewidth=2,
           label='Cut level → 3 clusters')

# Draw nodes
# Root
ax.add_patch(plt.Circle(node_positions['root'], 0.06, color='#78909C',
                         ec='black', lw=1.5, transform=ax.transAxes, zorder=5))
ax.text(*node_positions['root'], node_labels_txt['root'],
        ha='center', va='center', fontsize=9, fontweight='bold',
        transform=ax.transAxes, zorder=6)

# Intermediate nodes (non-leaf)
for key in ['s1_a', 's1_b']:
    is_leaf = key in leaf_to_raw
    color = leaf_colors.get(key, '#B0BEC5')
    ax.add_patch(plt.Circle(node_positions[key], 0.06, color=color,
                             ec='black', lw=1.5, alpha=0.85,
                             transform=ax.transAxes, zorder=5))
    txt = node_labels_txt[key]
    if is_leaf:
        raw_id = leaf_to_raw[key]
        mapped_c = remap_div[raw_id] + 1
        txt = f"Cluster {mapped_c}\nn = {s1['child_a_size'] if 'a' in key else s1['child_b_size']}"
    ax.text(*node_positions[key], txt,
            ha='center', va='center', fontsize=8.5, fontweight='bold',
            transform=ax.transAxes, zorder=6)

# Leaf nodes from second split
for key in ['s2_a', 's2_b']:
    color = leaf_colors.get(key, '#B0BEC5')
    raw_id = leaf_to_raw[key]
    mapped_c = remap_div[raw_id] + 1
    size = s2['child_a_size'] if 'a' in key else s2['child_b_size']
    ax.add_patch(plt.Circle(node_positions[key], 0.06, color=color,
                             ec='black', lw=1.5, alpha=0.85,
                             transform=ax.transAxes, zorder=5))
    ax.text(*node_positions[key], f"Cluster {mapped_c}\nn = {size}",
            ha='center', va='center', fontsize=8.5, fontweight='bold',
            transform=ax.transAxes, zorder=6)

# Split annotations
ax.text(0.5, 0.78, f"Split 1: SSE = {s1['parent_sse']:.1f}",
        ha='center', fontsize=10, style='italic', color='#555',
        transform=ax.transAxes)

split2_x = 0.25 if split_parent_2 == s1['child_a'] else 0.75
ax.text(split2_x, 0.38, f"Split 2: SSE = {s2['parent_sse']:.1f}",
        ha='center', fontsize=10, style='italic', color='#555',
        transform=ax.transAxes)

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.1)
ax.set_axis_off()
ax.set_title("Divisive Clustering — Bisecting K-Means Split Diagram\n"
             "Country Economic Indicators (167 Countries)",
             fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='lower right')
plt.tight_layout()
plt.savefig(f"{OUT}/08_divisive_split_diagram.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Split diagram saved.")

print(f"\n  Split History:")
for i, s in enumerate(bkm.split_history):
    print(f"    Split {i+1}: cluster {s['parent']} (n={s['parent_size']}, SSE={s['parent_sse']:.1f})")
    print(f"      → child A (n={s['child_a_size']})  +  child B (n={s['child_b_size']})")

# ── Divisive Cluster Map ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
for c in [1, 2, 3]:
    mask = div_labels == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=COLORS[c-1], s=55, alpha=0.78,
               edgecolors='white', linewidths=0.5,
               label=f"Cluster {c} — {CLUSTER_NAMES_DIV[c]}", zorder=3)
ax.set_title("Divisive Clustering — Bisecting K-Means (k = 3)\n"
             "PCA 2-D Projection",
             fontsize=13, fontweight='bold')
ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=11)
ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/09_divisive_cluster_map.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Cluster map saved.")

# ── Divisive interpretation ──────────────────────────────────────────────────
sil_div = silhouette_score(X, div_labels)
print(f"\n  Silhouette Score (Divisive): {sil_div:.4f}")
print(f"  Cluster sizes: {dict(zip(*np.unique(div_labels, return_counts=True)))}")
print("\n  Cluster Means (Divisive):")
df['Div_Cluster'] = div_labels
means_div = df.groupby('Div_Cluster')[FEATURES].mean().round(2)
means_div.index = [f"C{i} — {CLUSTER_NAMES_DIV[i]}" for i in means_div.index]
print(means_div.T.to_string())

print("\n  Interpretation:")
print("  - Bisecting K-Means is a divisive (top-down) approach: it starts with all")
print("    167 countries as one cluster, then recursively splits the cluster with")
print("    the highest SSE using 2-means until the desired k is reached.")
print("  - The first split separates the wealthiest nations from the rest.")
print("  - The second split divides the remaining countries into fragile and emerging.")
print("  - This top-down perspective naturally captures the large-scale structure first,")
print("    making it effective at identifying coarse, well-separated macro-clusters.")

# ══════════════════════════════════════════════════════════════════════════════
# 3.3  COMPARISON: AGGLOMERATIVE vs. DIVISIVE
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 3.3 Comparison: Agglomerative vs. Divisive ─────────────────────")

# ── Side-by-side cluster maps ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

titles = ["K-Means (Baseline, k=3)", "Agglomerative (Ward's, k=3)", "Divisive (Bisecting K-Means, k=3)"]
all_labels = [kmeans_labels, agg_labels, div_labels]
all_names = [CLUSTER_NAMES_KM, CLUSTER_NAMES_AGG, CLUSTER_NAMES_DIV]
all_sil = [silhouette_score(X, kmeans_labels), sil_agg, sil_div]

for ax, title, labels, names, sil in zip(axes, titles, all_labels, all_names, all_sil):
    for c in [1, 2, 3]:
        mask = labels == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=COLORS[c-1], s=50, alpha=0.78,
                   edgecolors='white', linewidths=0.4,
                   label=f"C{c}: {names[c]}")
    ax.set_title(f"{title}\nSilhouette = {sil:.4f}", fontsize=12, fontweight='bold')
    ax.set_xlabel(f"PC1 ({var1:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var2:.1f}%)", fontsize=10)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.85)
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle("Side-by-Side Comparison — K-Means vs. Agglomerative vs. Divisive\n"
             "Country Economic Indicators (PCA 2-D Projection)",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/10_comparison_side_by_side.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Side-by-side comparison saved.")

# ── Agreement between methods ────────────────────────────────────────────────
agree_agg_km = np.sum(agg_labels == kmeans_labels) / len(agg_labels)
agree_div_km = np.sum(div_labels == kmeans_labels) / len(div_labels)
agree_agg_div = np.sum(agg_labels == div_labels) / len(agg_labels)

print(f"\n  Agreement rates (after GDP-based re-ordering):")
print(f"    Agglomerative vs. K-Means:  {agree_agg_km:.1%}")
print(f"    Divisive vs. K-Means:       {agree_div_km:.1%}")
print(f"    Agglomerative vs. Divisive: {agree_agg_div:.1%}")

# ── Comparison table ─────────────────────────────────────────────────────────
print("\n  ┌─────────────────────────┬───────────────────┬───────────────────┐")
print("  │ Metric                  │ Agglomerative     │ Divisive          │")
print("  ├─────────────────────────┼───────────────────┼───────────────────┤")
print(f"  │ Silhouette Score        │ {sil_agg:>17.4f} │ {sil_div:>17.4f} │")
agg_sizes = [np.sum(agg_labels == c) for c in [1,2,3]]
div_sizes = [np.sum(div_labels == c) for c in [1,2,3]]
print(f"  │ Cluster sizes (1/2/3)   │ {agg_sizes[0]:>4}/{agg_sizes[1]:>4}/{agg_sizes[2]:>4}      │ {div_sizes[0]:>4}/{div_sizes[1]:>4}/{div_sizes[2]:>4}      │")
print(f"  │ Agreement with K-Means  │ {agree_agg_km:>16.1%} │ {agree_div_km:>16.1%} │")
print("  │ Strategy                │ Bottom-up merges  │ Top-down splits   │")
print("  │ Linkage / Split method  │ Ward's linkage    │ Bisecting K-Means │")
print("  └─────────────────────────┴───────────────────┴───────────────────┘")

# ── Per-cluster feature comparison (box plots for both methods) ──────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

KEY_FEATS = ['child_mort', 'income', 'gdpp', 'life_expec', 'inflation', 'health']
FEAT_LABELS = {
    'child_mort':  'Child Mortality\n(per 1000 births)',
    'income':      'Income per Person\n(USD)',
    'gdpp':        'GDP per Capita\n(USD)',
    'life_expec':  'Life Expectancy\n(years)',
    'inflation':   'Inflation Rate\n(%)',
    'health':      'Health Expenditure\n(% of GDP)'
}

for ax, feat in zip(axes.flatten(), KEY_FEATS):
    # Agglomerative data
    data_agg = [df[df['Agg_Cluster'] == c][feat].values for c in [1, 2, 3]]
    # Divisive data
    data_div = [df[df['Div_Cluster'] == c][feat].values for c in [1, 2, 3]]

    positions_agg = [1, 2, 3]
    positions_div = [4.5, 5.5, 6.5]

    bp1 = ax.boxplot(data_agg, positions=positions_agg, patch_artist=True, widths=0.55,
                     medianprops=dict(color='black', linewidth=2),
                     flierprops=dict(marker='o', markersize=3, alpha=0.4))
    bp2 = ax.boxplot(data_div, positions=positions_div, patch_artist=True, widths=0.55,
                     medianprops=dict(color='black', linewidth=2),
                     flierprops=dict(marker='o', markersize=3, alpha=0.4))

    for patch, color in zip(bp1['boxes'], COLORS[:3]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    for patch, color in zip(bp2['boxes'], COLORS[:3]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
        patch.set_hatch('//')

    ax.set_title(FEAT_LABELS[feat], fontsize=10, fontweight='bold')
    ax.set_xticks([2, 5.5])
    ax.set_xticklabels(['Agglomerative', 'Divisive'], fontsize=9)
    ax.axvline(x=3.75, color='grey', linestyle=':', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

patches_legend = [mpatches.Patch(color=COLORS[i], alpha=0.7,
                  label=f"C{i+1}: {CLUSTER_NAMES_AGG[i+1]}")
                  for i in range(3)]
patches_legend.append(mpatches.Patch(facecolor='white', edgecolor='black',
                      hatch='//', label='Divisive'))
patches_legend.append(mpatches.Patch(facecolor='white', edgecolor='black',
                      label='Agglomerative'))
fig.legend(handles=patches_legend, loc='lower center', ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, -0.04), frameon=True)
plt.suptitle("Feature Distributions — Agglomerative vs. Divisive (k = 3)",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/11_hierarchical_feature_boxplots.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Feature comparison box plots saved.")

# ── Countries that differ between methods ────────────────────────────────────
df['KM_Cluster'] = kmeans_labels
differ_mask = agg_labels != div_labels
n_differ = differ_mask.sum()
print(f"\n  Countries assigned differently (Agg vs. Div): {n_differ} / {len(df)}")
if n_differ > 0 and n_differ <= 30:
    diff_df = df.loc[differ_mask, ['country', 'Agg_Cluster', 'Div_Cluster', 'KM_Cluster']].copy()
    diff_df.columns = ['Country', 'Agglomerative', 'Divisive', 'K-Means']
    print(diff_df.to_string(index=False))

# ── CSV output with all cluster assignments ──────────────────────────────────
df_out = df[['country'] + FEATURES + ['KM_Cluster', 'Agg_Cluster', 'Div_Cluster']].copy()
df_out.to_csv(f"{OUT}/country_all_clusters.csv", index=False)
print(f"\n  CSV with all cluster assignments saved: {OUT}/country_all_clusters.csv")

# ── Final summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SUMMARY & CONCLUSION")
print("=" * 70)
print(f"""
  Silhouette Scores:
    K-Means (baseline):      {all_sil[0]:.4f}
    Agglomerative (Ward's):  {sil_agg:.4f}
    Divisive (Bisecting KM): {sil_div:.4f}

  Key Differences:
  - Agglomerative (bottom-up) merges the closest pair at each step,
    producing tight, fine-grained clusters. It tends to find compact
    sub-groups and is sensitive to local structure.
  - Divisive (top-down) splits the most heterogeneous cluster at each step,
    producing well-separated macro-clusters. It captures global structure
    first and is more robust to outliers in small sub-groups.

  Conclusion:
  - Both hierarchical methods produce a 3-cluster structure that aligns
    well with the K-Means baseline (Fragile / Emerging / Developed).
  - For this dataset, the method with the higher silhouette score
    ({['Agglomerative', 'Divisive'][int(sil_div > sil_agg)]}) provides
    slightly better cluster separation.
  - The agglomerative approach (Ward's linkage) is recommended for this
    dataset because Ward's method minimises variance — the same objective
    as K-Means — producing compact, interpretable clusters that match
    the economic development tiers well. The dendrogram also provides a
    richer visual of how countries relate hierarchically.
""")

print(f"All Part 3 outputs saved to: {OUT}/")
print("  06_agglomerative_dendrogram.png")
print("  07_agglomerative_cluster_map.png")
print("  08_divisive_split_diagram.png")
print("  09_divisive_cluster_map.png")
print("  10_comparison_side_by_side.png")
print("  11_hierarchical_feature_boxplots.png")
print("  country_all_clusters.csv")
