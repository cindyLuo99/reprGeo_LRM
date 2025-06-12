import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import pearsonr # Added import for Pearson correlation
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA

DEFAULT_FONTS = dict(
  title=16, axis=14, ticks=12, legend=10, annotation=12
)
DEFAULT_COLORS = {'base':'#5B0A0B','mod':'#E41A1C','shift':'#f04a00'}

def plot_distance_change_histogram(
    distances_baseline, distances_modulated, category_labels, dist_metric_name="cosine",
    save_path='reprGeo_default/figures/'
):
    """
    Plots a histogram of the change in exemplar-prototype distances
    between modulated and baseline passes, including statistical analysis.

    Args:
        distances_baseline (np.array or torch.Tensor): Distances in baseline pass.
        distances_modulated (np.array or torch.Tensor): Distances in modulated pass.
        category_labels (np.array or torch.Tensor): Labels indicating category/group for mixed model.
        dist_metric_name (str): Name of the distance metric used (for labeling).
    """
    # Ensure numpy arrays for calculations
    if hasattr(distances_baseline, 'cpu'): # Check if it's a tensor that needs moving/converting
        distances_baseline = distances_baseline.cpu().numpy()
    if hasattr(distances_modulated, 'cpu'):
        distances_modulated = distances_modulated.cpu().numpy()
    if hasattr(category_labels, 'cpu'):
        category_labels = category_labels.cpu().numpy()

    # Difference: Modulated - Baseline
    distance_diff = distances_modulated - distances_baseline

    # --- Statistical Analysis (Linear Mixed Model) ---
    df = pd.DataFrame({
        'distance_diff': distance_diff,
        'group': category_labels # Assuming 'group' is based on category or image ID
    })

    # Fit a mixed-effects model (Intercept-only, grouping by label)
    try:
        model = smf.mixedlm("distance_diff ~ 1", df, groups=df["group"])
        result = model.fit()
        mean_diff = result.params["Intercept"]
        p_value = result.pvalues["Intercept"]
        # Assuming one-tailed test (hypothesis is decrease, p < 0)
        if mean_diff < 0:
            p_value_one_tailed = p_value / 2
        else:
            p_value_one_tailed = 1 - (p_value / 2)

        # Format p-value for display
        if p_value_one_tailed < 0.0001:
            p_text = "p < 0.0001"
        else:
            p_text = f"p = {p_value_one_tailed:.4f}"
        stats_text = f"Mean Diff. = {mean_diff:.3f}\n{p_text}"
        print(result.summary())
        print(f"One-tailed p-value: {p_value_one_tailed}")

    except Exception as e:
        print(f"Statsmodels error: {e}. Skipping stats display on plot.")
        mean_diff = np.mean(distance_diff)
        stats_text = f"Mean Diff. = {mean_diff:.3f}\n(Stats model error)"

    # --- Plotting ---

    plt.figure(figsize=(4.5, 4)) # Adjust size as needed

    # Histogram
    counts, bin_edges, patches = plt.hist(
        distance_diff, bins=30, alpha=0.7, color='lightgray', edgecolor='black', linewidth=0.6 # Changed color
        )

    # Mean line
    plt.axvline(mean_diff, color='red', linestyle='--', linewidth=1.5, label=f'Mean Diff')
    plt.axvline(0, color='black', linestyle=':', linewidth=1.3, label='No Change') # Reference line at zero

    # Improve readability with larger fonts and updated labels
    plt.title('Change in Cluster Size \n(Modulated - Baseline)', fontsize=DEFAULT_FONTS['title']) # Kept user's title
    plt.xlabel(f'Distance Difference ({dist_metric_name})', fontsize=DEFAULT_FONTS['axis'])
    plt.ylabel('Frequency', fontsize=DEFAULT_FONTS['axis']) # Kept user's y-label
    plt.xticks(fontsize=DEFAULT_FONTS['ticks'])
    plt.yticks(fontsize=DEFAULT_FONTS['ticks'])

    # Remove top and right spines for cleaner look
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add stats text back to the plot with larger font
    # text_x_pos = plt.xlim()[0] + (plt.xlim()[1] - plt.xlim()[0]) * 0.05 # Position text near left edge
    # text_y_pos = plt.ylim()[1] * 0.85 # Position text near top
    # plt.text(text_x_pos, text_y_pos, stats_text, fontsize=annotation_fontsize, # Uncommented this block
    #          verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))

    plt.legend(fontsize=DEFAULT_FONTS['legend'])
    plt.tight_layout()
    plt.savefig(f'{save_path}/Histogram of Differences-Local.pdf', bbox_inches='tight') # Kept user's savefig line
    plt.show()

    return {
  'mean_diff': mean_diff,
  'p_one_tailed': p_value_one_tailed
}

def plot_category_preservation_curve(ks, category_pres_1, category_pres_3, color_base='#5B0A0B', color_mod='#E41A1C', save_path='reprGeo_default/figures/'):
    # --- Plotting ---
    plt.figure(figsize=(4.5, 4.2)) # Adjust size as needed

    # Plot category preservation lines with distinct styles and chosen colors
    plt.plot(ks, category_pres_3,
            label="Modulated", color=color_mod, linestyle='-', linewidth=3) # Solid line for modulated
    plt.plot(ks, category_pres_1,
            label="Baseline", color=color_base, linestyle='--', linewidth=3) # Dashed line for baseline

    # Add labels, title, and legend with larger fonts
    plt.xlabel("Number of Neighbors (k)", fontsize=DEFAULT_FONTS['axis'])
    plt.ylabel("Proportion Same Category", fontsize=DEFAULT_FONTS['axis']) # Assuming y-axis is a proportion
    plt.title("Local Category Preservation via k-NN", fontsize=DEFAULT_FONTS['title'])
    plt.xticks(fontsize=DEFAULT_FONTS['ticks'])
    plt.yticks(fontsize=DEFAULT_FONTS['ticks'])
    plt.legend(fontsize=DEFAULT_FONTS['legend'])

    # Add grid and remove spines for cleaner look
    plt.grid(axis='y', linestyle=':', alpha=0.7) # Light grid on y-axis
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust layout and display
    plt.tight_layout()
    # Add savefig command if needed
    if save_path:
        plt.savefig(f'{save_path}Category Preservation-Local.pdf', bbox_inches='tight')
    plt.show()



def plot_category_distance_change_heatmap(
    matrix, unique_labels,number2label,
    tick_spacing=4,
    cmap='RdBu_r',
    vmin=-0.02,vmax=0.02,
    save_path='reprGeo_default/figures/'
):
    """
    Plot a half-masked heatmap of average distance changes between category pairs.
    """
    # Convert to numpy matrix
    if hasattr(matrix, 'cpu'):
        matrix = matrix.cpu().numpy()

    # Determine categories
    cats = np.array(unique_labels)
    C = len(cats)

    # Build sparse tick labels
    tick_idxs = np.arange(0, C, tick_spacing)
    if number2label:
        tick_labels = [number2label[c] if i in tick_idxs else '' for i, c in enumerate(cats)]
    else:
        tick_labels = ['' if i not in tick_idxs else str(c) for i, c in enumerate(cats)]

    # Mask upper triangle
    mask = np.triu(np.ones((C, C), dtype=bool), k=1)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        mask=mask,
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar=True,
        vmin=vmin,
        vmax=vmax
    )

    ax = plt.gca()
    kept_labels = [tick_labels[i] for i in tick_idxs]
    ax.set_xticks(tick_idxs + 0.5)
    ax.set_yticks(tick_idxs + 0.5)
    ax.set_xticklabels(kept_labels, fontsize=10, rotation=45, ha='right')
    ax.set_yticklabels(kept_labels, fontsize=10, rotation=0)

    # Colorbar styling
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=DEFAULT_FONTS['ticks'])

    # Title
    plt.title("Between-Category Distance Change", fontsize=DEFAULT_FONTS['title'], pad=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}Heatmap Category Distances-Global.pdf', bbox_inches='tight')
    plt.show()

def plot_prototype_shift_pca(
    prot_base, prot_mod, act_base, act_mod,
    pca_on='baseline',
    colors=DEFAULT_COLORS,
    save_path='reprGeo_default/figures/'
):
    """
    2D PCA of prototypes, drawing gray lines and colored points.
    """
    # 1. fit PCA
    data_for_pca = act_base if pca_on=='baseline' else torch.cat([act_base,act_mod],0)
    pca = PCA(n_components=2)
    pca.fit(data_for_pca.cpu().numpy())

    pb2 = pca.transform(prot_base.cpu().numpy())
    pm2 = pca.transform(prot_mod.cpu().numpy())

    # 2. plot
    fig, ax = plt.subplots(figsize=(4.5,4))
    for i in range(pb2.shape[0]):
        ax.plot([pb2[i,0],pm2[i,0]],[pb2[i,1],pm2[i,1]],
                color='gray',linestyle='--',linewidth=0.75,alpha=0.6,zorder=1)
    ax.scatter(pm2[:,0],pm2[:,1],c=colors['mod'],s=50, label='Modulated',zorder=3)
    ax.scatter(pb2[:,0],pb2[:,1],c=colors['base'],s=50, label='Baseline',alpha=0.8,zorder=2)
    
    # style
    ax.set_title("Prototype Shifts in PCA Space", fontsize=DEFAULT_FONTS['title'], pad=10)
    for spine in ('top','right','left','bottom'):
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    ax.legend(fontsize=DEFAULT_FONTS['legend'], loc='upper right')
    # arrows
    xmin,xmax = ax.get_xlim(); ymin,ymax = ax.get_ylim()
    ax.arrow(xmin,0,xmax-xmin,0,head_width=(ymax-ymin)*0.02, length_includes_head=True, color='k')
    ax.arrow(0,ymin,0,ymax-ymin,head_width=(xmax-xmin)*0.02, length_includes_head=True, color='k')
    ax.text(xmax,0.02*(ymax-ymin),'PC1',ha='right',va='bottom',fontsize=DEFAULT_FONTS['axis'])
    ax.text(0.02*(xmax-xmin),ymax,'PC2',ha='left',va='top',fontsize=DEFAULT_FONTS['axis'])
    plt.savefig(f'{save_path}Prototype Shifts PCA-Global.pdf', bbox_inches='tight')
    plt.tight_layout()
    plt.show()


def plot_shift_vs_separation_kde(
    shift_dist, baseline_sep,
    colors=DEFAULT_COLORS,
    save_path='reprGeo_default/figures/'
):
    """
    Overlaid KDEs of shift_dist and baseline_sep, with mean lines.
    """
    fig, ax = plt.subplots(figsize=(6,4))
    sns.kdeplot(shift_dist,   color=colors['shift'], fill=True, ax=ax, label='Prototype Shifts',   alpha=0.3)
    ax.axvline(shift_dist.mean(),   color=colors['mod'], linestyle='--', label=f"Mean Shift ({shift_dist.mean():.3f})")
    sns.kdeplot(baseline_sep, color=colors['base'],  fill=True, ax=ax, label='Baseline Separation', alpha=0.3)
    ax.axvline(baseline_sep.mean(), color=colors['base'],linestyle=':', label=f"Mean Sep ({baseline_sep.mean():.3f})")
    ax.set_title('Prototype Shifts vs. Baseline Separations', fontsize=DEFAULT_FONTS['title'])
    ax.set_xlabel("Cosine Distance", fontsize=DEFAULT_FONTS['axis'])
    ax.set_ylabel("Density", fontsize=DEFAULT_FONTS['axis'])
    ax.tick_params(labelsize=DEFAULT_FONTS['ticks'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=DEFAULT_FONTS['legend'], loc='upper center', bbox_to_anchor=(0.4, 1.0)) # 
    plt.tight_layout()
    plt.savefig(f'{save_path}Overlaid_Shift_Separation_KDE-Global.pdf', bbox_inches='tight')
    plt.show()


def plot_local_global_correlation(global_avg, local_avg, global_all, local_all,
    error_type='sem', annotate=False,
    save_path='reprGeo_default/figures/'
):
    """
    Scatter global_avg vs. local_avg with error bars from “all” dicts.
    """
    cats      = list(global_avg.keys())
    x         = np.array([global_avg[c] for c in cats])
    y         = np.array([local_avg[c]  for c in cats])
    # errors
    def get_err(d):
        return np.array([
            (np.std(d[c]) / np.sqrt(len(d[c]))) if error_type=='sem'
            else np.std(d[c])
            for c in cats
        ])
    xerr = get_err(global_all)
    yerr = get_err(local_all)

    # correlation
    r, p = pearsonr(x, y)
    p_text = "p < 0.001" if p<0.001 else f"p = {p:.3f}"
    corr_text = f"r = {r:.2f}\n{p_text}"

    plt.figure(figsize=(4.5,4.2))
    plt.errorbar(
        x, y, xerr=xerr, yerr=yerr,
        fmt='o', ecolor='gray', elinewidth=1, capsize=3,
        markerfacecolor='darkgray', markeredgecolor='darkgray', alpha=0.8
    )
    plt.xlabel("Global Prototype Shift (cosine)", fontsize=DEFAULT_FONTS['axis'])
    plt.ylabel("Local Cluster Change (cosine)", fontsize=DEFAULT_FONTS['axis'])
    plt.title("Global Shift vs. Local Compaction", fontsize=DEFAULT_FONTS['title'])
    plt.xticks(fontsize=DEFAULT_FONTS['ticks'])
    plt.yticks(fontsize=DEFAULT_FONTS['ticks'])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # annotate correlation
    if annotate:
        plt.text(0.05, 0.05, f"r = {r:.2f}\np = {p:.2g}",
            transform=plt.gca().transAxes, fontsize=DEFAULT_FONTS['legend'],
            verticalalignment='bottom',
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    if save_path:
        plt.savefig(f'{save_path}Correlation Plot-Global_Local.pdf', bbox_inches='tight')
    plt.tight_layout()
    plt.show()