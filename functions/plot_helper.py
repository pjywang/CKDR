import itertools, ternary
import matplotlib.pyplot as plt

import numpy as np
import torch

from .Kquantiles import KQuantiles

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def symmetric_radial_positions(center, radius, n):
    if n == 1:
        return [center]
    # projection matrix from barycentric to Cartesian (x=b + c/2, y=c*sqrt(3)/2)
    M = np.array([[0, 1, 0.5],
                  [0, 0, np.sqrt(3)/2]])
    # augmented system adding sum constraint to invert projection
    X = np.vstack([M, np.ones(3)])
    # compute center in 2D Cartesian
    center2D = M.dot(center)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    positions = []
    for angle in angles:
        off2D = center2D + radius * np.array([np.cos(angle), np.sin(angle)])
        b = np.concatenate([off2D, [1]])
        # solve for barycentric coords summing to 1
        q = np.linalg.solve(X, b)
        positions.append(q)
    return positions


def tern_scatter(Z, Y, ax=None, groups=None, show=False, save=False, 
                 marker_list=itertools.repeat(None), color_list=itertools.repeat(None), 
                 names=None, offset=[0.08, 0.2, 0.08],
                 grid=False, legend=True, **params):
    """
    Wrapper function for ternary.scatterplot for convenience

    Parameters
    ----------
    :param np.ndarray Z: projected data
    :param np.ndarray Y: class labels
    :param ax: to use a predefined axis defined by fig.add_subplot, etc.

    :param list groups: text-valued names for the class labels. Default is set to class labels in Y
    :param bool show: conduct plt.show() if True (default)
    :param str save: if not False, put the filename here; e.g., save="images/tern"
    :param list marker_list: list of markers for each group
    :param list color_list: list of colors for each group
    :param dict params: additional parameters for plt.scatter
    """

    # Ternary axis
    if isinstance(ax, ternary.TernaryAxesSubplot):
        figure, tax = ax.get_figure(), ax
    else:
        figure, tax = ternary.figure(scale=1, ax=ax)    # ax=None makes a new axis

    # Boundary and grid lines
    tax.boundary(linewidth=0.8, alpha=0.8, zorder=1)
    if grid:
        tax.gridlines(multiple=0.2, color="black", alpha=0.35)

    if groups is None:
        groups = sorted(set(Y.ravel()))
        Y_new = Y.copy()
    else:
        # dtype matching with Y and groups; dtype=object is necessary for string comparison
        Y_new = np.array([groups[0] for i in range(Y.shape[0])], dtype=object)
        Y_new[Y == -1.], Y_new[Y == 1.] = groups[0], groups[1]

    if next(iter(color_list)) is not None:
        color_list = itertools.chain(iter(color_list), itertools.repeat(None)) 

    if next(iter(marker_list)) is not None:
        marker_list= itertools.chain(iter(marker_list), itertools.repeat(None))

    params.setdefault('s', 12)
    # Scatter plot
    for group, m, c in zip(groups, marker_list, color_list):
        if 'facecolors' in params and params['facecolors'] == 'none' and (m == 'x' or m == '+'):
            del params['facecolors']
            tax.scatter(Z[Y_new == group], label=group, marker=m, color=c, **params)
            params['facecolors'] = 'none'
        else:
            tax.scatter(Z[Y_new == group], label=group, marker=m, color=c, **params)

    if legend:
        tax.legend()

    # Corner labels + Removing matplotlib grids
    corner_labeling(tax, names=names, offset=offset)

    if save:
        tax.savefig(save, dpi=400)

    if show:
        tax.show()
    
    return tax


def tern_scatter_decbdry(P, model, sigma, epsilon, ax=None, show=False,
                         intercept=0, linewidth=0.7, linecolor='black', linestyle='dashed', linealpha=0.8,
                         **params):
    """
    Pytorch migrated version of tern_scatter_decisionbdry
    Plots binary decision boundary with scatters
        :param np.ndarray P: CDR matrix result
        :param model: CKDR class instance
            Having processed information on X and Y's original values
        :param ax: external plot axis if not None
        :param intercept: if decision boundary looks severely biased, try to adjust this value
        :param dict params: additional parameters for the function tern_scatter()
    """
    # Reconstruction for Y
    Y = model.Y.numpy().ravel()
    assert len(Y) == model.n, "Sample sizes do not match; probably the response variable is not binary"
    assert len(model.values) == 2, "Response variable is not binary"

    # Make Ternary Grid
    x_range = np.arange(0, 1.01, 0.0025) 
    coordinate_list = np.asarray(list(itertools.product(x_range, repeat=2))) 
    coordinate_list = np.append(coordinate_list, (1 - coordinate_list[:, 0] - coordinate_list[:, 1]).reshape(-1, 1), axis=1)

    # === reshape coordinates and data for use with pyplot contour function
    x = coordinate_list[:, 0].reshape(x_range.shape[0], -1)
    y = coordinate_list[:, 1].reshape(x_range.shape[0], -1)

    # Take nonnegative part
    nonneg_list = coordinate_list[(coordinate_list >= 0).all(axis=1)]
    nonneg_list = torch.tensor(nonneg_list, dtype=model.dtype, device=model.device)

    # Make prediction on the grid points--proj_result parameter takes place here
    pred = model.predict(P, sigma, epsilon, nonneg_list, proj_result=True) + intercept

    # Store space with NaN for negative coordinates
    z = np.empty(coordinate_list.shape[0])
    z.fill(np.nan)

    # Scores
    z[(coordinate_list >= 0).all(axis=1)] = pred
    z = z.reshape(x.shape[0], -1)

    # Contour plot
    level = [0.]  # indicates the decision value between -1 and 1
    
    # Extracting contour information with plt.contour function (with no plot)
    f, a = plt.subplots()
    contours = a.contour(x, y, z, level)
    f.clf()  # makes sure that contours are not plotted in cartesian plot

    # New ternary axis for plotting
    if isinstance(ax, ternary.TernaryAxesSubplot):
        figure, tax = ax.get_figure(), ax
    else:
        figure, tax = ternary.figure(scale=1, ax=ax)    # ax=None makes a new axis

    # === plot contour lines - need to set colors?
    if contours.allsegs[0][0].shape[0] != 0: 
        for ii, contour in enumerate(contours.allsegs):
            for jj, seg in enumerate(contour):
                tax.plot(seg[:, 0:2] * 1, alpha=linealpha, color=linecolor, linestyle=linestyle, linewidth=linewidth)    ## scale = 1

    # Reduced data
    Z = torch.matmul(model.X, P.T).numpy()

    # check if the params has groups attribute
    if 'groups' not in params:
        # If not, use the model values as groups
        params['groups'] = model.values

    # print(model.values)
    tern_scatter(Z, Y, ax=tax,
                 show=show, **params)

    yhat = model.predict(P, sigma, epsilon, Z, proj_result=True).ravel()
    print("Train accuracy:", np.mean(model.Y.numpy().ravel() == yhat))

    return tax


def tern_scatter_cmap(Z, Y, ax=None, show=False, save=False, cmap=plt.cm.jet,
                      colorbar=True, cbar_label=None, 
                      cax_pos=[0.9, 0.45, 0.04, 0.45], label_position=[0, 0], 
                      grid=False, **params):
    """
    ax should be the original plt.axis class; ternary-axis has no method .inset_axes
    """

    if ax is None:
        _, ax = plt.subplots()  ## make a room for colorbar arrangement
    cax = ax.inset_axes(cax_pos)  ## Axis space for the colorbar
    figure, tax = ternary.figure(ax=ax)
    # figure.set_size_inches(5.5, 5, forward=True)

    # Boundary and grid lines
    tax.boundary(linewidth=0.8, alpha=0.8, zorder=1)
    if grid:
        tax.gridlines(multiple=0.2, color="black", alpha=0.35)

    vmax, vmin = Y.max() * 1., Y.min() * 1.

    # For customized control of colorbar with the python-ternary package
    # see the source codes of https://github.com/marcharper/python-ternary/tree/master
    params.setdefault('s', 12)
    ax2 = tax.scatter(Z, vmax=vmax, vmin=vmin,
            colormap=cmap, colorbar=False, c=Y, cmap=cmap, **params)
    
    if colorbar:
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), 
                           cmap=cmap)
        cb = plt.colorbar(sm, cax=cax, ax=ax2, shrink=0.6)
        label_pos = np.array([-1.5, 47]) + np.array(label_position)
        cb.ax.text(*label_pos, cbar_label)

    # Corner labels + Removing matplotlib grids
    corner_labeling(tax)

    if save:
        tax.savefig(save, dpi=400)
    if show:
        tax.show()

    return tax


def corner_labeling(tax, fontsize=12, names=None, offset=[0.08, 0.2, 0.08]):
    """
    names: a list of strings; default = [r"$Z_1$", r"$Z_2$", r"$Z_3$"]
    offset: distances from the corners
            a list of floats; default = [0.08, 0.08, 0.08]
    """
    # Delete matplotlib background and ticks
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()
    tax.set_background_color(color="white")

    if names is None:
        names = [r"$Z_1$", r"$Z_2$", r"$Z_3$"]

    # Corner labels
    tax.right_corner_label(names[0], fontsize=fontsize, offset=offset[0])
    tax.top_corner_label(names[1], fontsize=fontsize, offset=offset[1])
    tax.left_corner_label(names[2], fontsize=fontsize, offset=offset[2])

    return tax