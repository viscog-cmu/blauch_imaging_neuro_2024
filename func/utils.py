import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from scipy.special import logit, expit
import itertools
import pandas as pd
import statsmodels.formula.api as smf
import random

def hide_current_axis(*args, **kwds):
    """
    Hide the current matplotlib axis.
    
    Parameters
    ----------
    *args
        Variable length argument list
    **kwds
        Arbitrary keyword arguments
    """
    plt.gca().set_visible(False)
    

def reject_outliers(data, m=3):
    """
    Remove outliers from data based on z-score threshold.
    
    Parameters
    ----------
    data : array-like
        Input data to remove outliers from
    m : float, optional
        Z-score threshold for outlier detection, default 3
        
    Returns
    -------
    tuple
        (cleaned data array, number of outliers removed)
    """
    z_dat = (data - data.mean())/data.std(ddof=0)
    is_outlier = np.abs(z_dat)> m
    n_outliers = np.sum(is_outlier)
    data[is_outlier] = np.nan
    return data, n_outliers


def reject_df_outliers(df, m=3, exclude_cols=['subnum', 'gender'], 
                       bounded_cols=[], 
                       verbose=False, standardize=False,
                       ):
    """
    Remove outliers from all numeric columns in a DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to clean
    m : float, optional
        Z-score threshold for outlier detection, default 3
    exclude_cols : list, optional
        Columns to exclude from outlier detection
    bounded_cols : list, optional
        Columns to apply logit transform before outlier detection
    verbose : bool, optional
        Whether to print outlier detection results
    standardize : bool, optional
        Whether to standardize data after outlier removal
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with outliers removed
    """
    for col in df.columns:
        if col in exclude_cols:
            continue
        done = False
        tot_outliers = 0
        while not done:
            if col in bounded_cols:
                transformed, n_outliers = reject_outliers(logit(df[col]), m)
                df[col] = expit(transformed)
            else:
                df[col], n_outliers = reject_outliers(df[col], m=m)
            tot_outliers += n_outliers
            if n_outliers == 0:
                done = True
                if verbose:
                    print(f'{col}: {tot_outliers} outliers')
                if standardize:
                    df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)
                    if verbose:
                        print(f'z max: {np.nanmax(df[col])}')
    return df


def partialled_regression(data, y, x, nuis, x_norm=[], y_norm=[], standardize=False, plot=True, 
                          x_name=None, y_name=None,
                          **kwargs,
                          ):
    """
    Perform partial regression controlling for nuisance variables.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data containing variables
    y : str
        Dependent variable name
    x : str
        Independent variable name
    nuis : list
        Nuisance variable names to control for
    x_norm : list, optional
        Variables to normalize x by
    y_norm : list, optional
        Variables to normalize y by
    standardize : bool, optional
        Whether to standardize variables
    plot : bool, optional
        Whether to plot regression results
    x_name : str, optional
        Display name for x variable
    y_name : str, optional
        Display name for y variable
    **kwargs
        Additional arguments passed to plotting
        
    Returns
    -------
    tuple
        (regression results, residuals, figure, axis) if plot=True
        (regression results, residuals) if plot=False
    """
    data = drop_specific_nans(data, [y]+[x]+nuis+x_norm+y_norm)
    if standardize:
        for var in [y]+[x]+nuis+x_norm+y_norm:
            data[var] = (data[var] - data[var].mean())/data[var].std(ddof=0)
    if x_name is None:
        x_name = x
    if y_name is None:
        y_name = y
    if len(x_norm) > 0:
        assert len(x_norm) == 1
        if x_name is not None:
            num = x.replace('_', '\_')
            denom = x_norm[0].replace('_', '\_')
            x_name = rf'$\dfrac{{\rm{{{num}}}}}{{{denom}}}$'
        data[x+'_norm'] = (data[x] / data[x_norm[0]])
        x = x+'_norm'
    if len(y_norm) > 0:
        assert len(y_norm) == 1
        if y_name is not None:
            y_name = f'{x}/\n{x_norm[0]}'
        data[y+'_norm'] = (data[y] / data[y_norm[0]])
        y = y+'_norm'
    if len(nuis) > 0:
        y_residuals = multiple_regression(data, y, nuis).resid
        x_residuals = multiple_regression(data, x, nuis).resid
    else:
        y_residuals = data[y]
        x_residuals = data[x]
    new_data = {y: y_residuals, x:x_residuals}
    reg_res = smf.ols(f'{y} ~ {x}', data=new_data).fit()
    if plot:
        fig, ax = _plot_pairwise_residuals(new_data, x, y, nuisance=nuis, standardized=standardize, 
                                           x_name = x_name, y_name=y_name, **kwargs)
        return reg_res, y_residuals, fig, ax
    else:
        return reg_res, y_residuals


def multiple_regression(data, y, x_vars, interactions=[], categorical_xvars=[]):
    """
    Perform multiple regression with optional interactions.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data containing variables
    y : str
        Dependent variable name
    x_vars : list
        Independent variable names
    interactions : list, optional
        Interaction terms to include
    categorical_xvars : list, optional
        Variables to treat as categorical
        
    Returns
    -------
    statsmodels.regression.linear_model.RegressionResultsWrapper
        Regression results
    """
    data = drop_specific_nans(data, [y]+x_vars)
    x_vars_eq = []
    for var in x_vars:
        if categorical_xvars is not None and var in categorical_xvars:
            x_vars_eq.append(f'C({var})')
        else:
            x_vars_eq.append(var)
    eq= f'{y} ~ {x_vars_eq[0]}'
    for var in x_vars_eq[1:] + interactions:
        if var == y:
            continue
        else:
            eq+=f'+ {var}'
    model = smf.ols(eq, data=data).fit()
    return model


def drop_specific_nans(data, variables):
    """
    Drop rows with NaN values in specified variables.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame
    variables : list
        Variables to check for NaN values
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with NaN rows removed
    """
    new_data = pd.DataFrame({var: data[var] for var in variables})
    new_data = new_data.dropna()
    return new_data


def _plot_pairwise_residuals(data, x, y, nuisance, show=True, standardized=False, ax=None, nuis_in_name=True, 
                             x_name=None, y_name=None, fontsize=None,
                             **kwargs): #, alpha=0.05):
    """
    Plot residuals from pairwise regression.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data containing variables
    x : str
        Independent variable name
    y : str
        Dependent variable name
    nuisance : list
        Nuisance variables controlled for
    show : bool, optional
        Whether to display plot
    standardized : bool, optional
        Whether data is standardized
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    nuis_in_name : bool, optional
        Whether to include nuisance variables in plot title
    x_name : str, optional
        Display name for x variable
    y_name : str, optional
        Display name for y variable
    fontsize : float, optional
        Font size for labels
    **kwargs
        Additional arguments passed to seaborn.regplot
        
    Returns
    -------
    tuple
        (figure, axis)
    """
    r,p = pearsonr(data[x], data[y])
    rho, p_rho = spearmanr(data[x], data[y])
    if ax is None:
        tight=False
        fig, ax = plt.subplots(figsize=(2.75,2.5), )
    else:
        tight=False
        fig = ax.get_figure()
    if 'laterality' in y or 'lat' in y:
        ax.axhline(color='r',linestyle='--')
        ax.set_ylim(-1.05, 1.05)
    if 'laterality' in x or 'lat' in x:
        ax.axvline(color='r',linestyle='--')
        ax.set_xlim(-1.05, 1.05)
    g = sns.regplot(data=data, x=x, y=y, ax=ax, fit_reg=True, truncate=False, **kwargs)
    if len(nuisance)>0 and nuis_in_name:
        nuis_title = 'nuisance: '
        for ii, nuis in enumerate(nuisance):
            nuis_title += nuis
            if ii+1 < len(nuisance):
                nuis_title += ', '
    else:
        nuis_title=''
    if p_rho < 0.00001:
        title = f'rho={rho:.03f}, p<0.00001, n={len(data[x])}'
    else:
        title = f'rho={rho:.03f}, p={p_rho:.05f}, n={len(data[x])}'
    g.set_title(title) #, fontweight='bold' if p<alpha else None)
    if standardized:
        g.set_xlabel(f'z( {x_name if x_name is not None else x} )', fontsize=fontsize)
        g.set_ylabel(f'{nuis_title}\nz( {y_name if y_name is not None else y} )', fontsize=fontsize)
    else:
        g.set_xlabel(x_name if x_name is not None else x, fontsize=fontsize)
        g.set_ylabel(f'{nuis_title}\n{y_name if y_name is not None else y}', fontsize=fontsize)
    
    if tight:
        plt.tight_layout()
    
    if show:
        plt.show()
    return fig, ax


def grouped_bar_with_connections(data, x, y, hue, groupby, order=None, ci=95, main_alpha=0.5, dot_alpha=0.75, 
                                 connect_actual_x=True, 
                                 main_fig_type='bar', dot_fig_type='swarm',
                                 palette=None,
                                 ax=None,
                                 ):
    """
    Create grouped bar plot with connected points.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot
    x : str
        Variable for x-axis
    y : str
        Variable for y-axis
    hue : str
        Variable for color grouping
    groupby : str
        Variable for connecting points
    order : list, optional
        Order of x-axis categories
    ci : float, optional
        Confidence interval for error bars
    main_alpha : float, optional
        Alpha for main plot elements
    dot_alpha : float, optional
        Alpha for dot plot elements
    connect_actual_x : bool, optional
        Whether to connect points at actual x positions
    main_fig_type : {'bar', 'point'}, optional
        Type of main plot
    dot_fig_type : {'swarm', 'strip'}, optional
        Type of dot plot
    palette : list, optional
        Color palette
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axes
    """
    if dot_fig_type == 'swarm':
        ax = sns.swarmplot(data=data, x=x, y=y, hue=hue, order=order, dodge=True, alpha=dot_alpha, palette=palette, edgecolor='k', linewidth=1, ax=ax)
    elif dot_fig_type == 'strip':
        ax = sns.stripplot(data=data, x=x, y=y, hue=hue, order=order, dodge=True, alpha=dot_alpha, jitter=False, palette=palette, edgecolor='k', linewidth=1, ax=ax)
    # ax = plt.gca()
    if connect_actual_x:
        offsets = [p.get_offsets() for p in ax.collections]
        actual_x = [o[:, 0] for o in offsets]
        actual_y = [o[:, 1] for o in offsets]
        for i, j in itertools.combinations(range(len(actual_x)), 2):
            if not len(actual_x[i]) or not len(actual_x[j]) or np.abs(np.nanmean(actual_x[j]) - np.nanmean(actual_x[i])) > 0.41:
                # print(np.abs(np.nanmean(actual_x[j]) - np.nanmean(actual_x[i])))
                continue
            for ind in range(len(actual_x[i])):
                ax.plot([actual_x[i][ind], actual_x[j][ind]], [actual_y[i][ind], actual_y[j][ind]], marker='', linestyle='-', color='grey', alpha=0.5)
    else:
        data[x] = data[x].astype('category')
        data[hue] = data[hue].astype('category')
        if len(np.unique(data[hue])) > 2:
            raise NotImplementedError('Only works for 2 hues')
        for group_hue, group in data.groupby(groupby):
            xs = group[x].cat.codes
            ys = group[y]
            hues = group[hue].cat.codes
            # Plot a line connecting the points in the group
            for i, j in itertools.combinations(range(len(xs)), 2):

                # Check if the x variable is the same
                if xs.iloc[i] == xs.iloc[j]:
                    # Plot a line connecting the points
                    ax.plot([xs.iloc[i]-0.2, xs.iloc[j]+0.2], [ys.iloc[i], ys.iloc[j]], marker='', linestyle='-', color='k', alpha=0.8)
    if main_fig_type == 'point':
        sns.pointplot(data=data, x=x, y=y, hue=hue, order=order, ci=ci, linestyles='', dodge=0.25, palette=palette, alpha=main_alpha, ax=ax)
    elif main_fig_type == 'bar':
        sns.barplot(data=data, x=x, y=y, hue=hue, order=order, ci=ci, palette=palette, alpha=main_alpha, ax=ax)
    
    return plt.gca()


def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size between two groups.
    
    Parameters
    ----------
    group1 : array-like
        First group's data
    group2 : array-like
        Second group's data
        
    Returns
    -------
    float
        Cohen's d effect size
    """
    mean1 = np.nanmean(group1, 0)
    mean2 = np.nanmean(group2, 0)
    sd1 = np.nanstd(group1,0)
    sd2 = np.nanstd(group2, 0)
    n1=group1.shape[0]
    n2=group2.shape[0]
    return ((mean1 - mean2)/(np.sqrt(((n1 - 1)*sd1 * sd1 + (n2-1)*sd2 * sd2) / (n1 + n2-2)))).squeeze()


def plot_scatters(layout, shape, df, fn=None, pwidth=4, pheight=3.3, fontsize=None, show_full=True,
                    sharex=False, sharey=False, constrain_axes=True,
                    **kws,
                    ):
    """
    Create grid of scatter plots.
    
    Parameters
    ----------
    layout : list
        List of plot specifications
    shape : tuple
        Grid shape (rows, cols)
    df : pandas.DataFrame
        Data to plot
    fn : str, optional
        Output filename
    pwidth : float, optional
        Width per plot
    pheight : float, optional
        Height per plot
    fontsize : float, optional
        Font size
    show_full : bool, optional
        Whether to display full plot
    sharex : bool, optional
        Whether to share x axes
    sharey : bool, optional
        Whether to share y axes
    constrain_axes : bool, optional
        Whether to constrain axis limits
    **kws
        Additional arguments passed to plotting functions
        
    Returns
    -------
    tuple
        (figure, axes array)
    """
    fig, axs = plt.subplots(nrows=shape[0], ncols=shape[1], figsize=(shape[1]*pwidth, shape[0]*pheight), sharex=sharex, sharey=sharey)
    axs = axs.reshape(-1)
    for ii, ax in enumerate(axs[:len(layout)]):
        comp = layout[ii]
        if len(comp) == 5:
            x, y, nuis, x_norm, title = comp
            more_kws = {}
        else:
            x, y, nuis, x_norm, title, more_kws = comp
        more_kws.update(kws)
        model, residuals, fig, ax = partialled_regression(df, x=x, y=y, ax = ax, 
                                                      nuis=nuis, 
                                                      x_norm=x_norm,
                                                      fontsize=fontsize,
                                                      **more_kws)
        if ('laterality' in y or 'lat' in y) and 'laterality_diff' not in y:
            ax.axhline(color='r',linestyle='--')
            if constrain_axes:
                ax.set_ylim(-1.05, 1.05)
        if ('laterality' in x or 'lat' in x) and 'laterality_diff' not in x:
            ax.axvline(color='r',linestyle='--')
            if constrain_axes:
                ax.set_xlim(-1.05, 1.05)
        if ('lh' in y or 'rh' in y) and df[y].max() > 50:
            ax.set_ylim(-50, ax.get_ylim()[1])
        if title is not None:
            ax.set_title(f'{title}\n{ax.get_title()}', fontsize=fontsize)
    plt.tight_layout()
    if fn is not None:
        plt.savefig(fn, dpi=300, bbox_inches='tight')
    if show_full:
        plt.show()
    return fig, axs


def random_combination(iterable, r, n):
    """
    Generate n random combinations of r items from iterable.
    
    Parameters
    ----------
    iterable : iterable
        Collection to sample from
    r : int
        Size of each combination
    n : int
        Number of combinations to generate
        
    Returns
    -------
    list
        List of random combinations
    """
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    indices = sorted([random.sample(range(len(pool)), 2) for ii in range(n)])
    rand_combs = [tuple(pool[i] for i in ind) for ind in indices]
    return rand_combs


def cleanup_legend(ax, bbox_to_anchor=(1.05, 0.5), loc='lower left', **kws):
    """
    Clean up duplicate legend entries.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes containing legend
    bbox_to_anchor : tuple, optional
        Legend anchor point
    loc : str, optional
        Legend location
    **kws
        Additional arguments passed to legend
    """
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    plt.sca(ax)
    plt.legend(unique_handles, unique_labels, bbox_to_anchor=bbox_to_anchor, loc=loc, borderaxespad=0., **kws)