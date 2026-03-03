# Nathaniel Linden (UCSD MAE) - 2024
# Edited by Annette Thompson (CU Boulder) - 2026
import jax
import numpy as np

import pytensor.tensor as pt
from pytensor.graph import Apply, Op

import preliz as pz
import pymc as pm

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
import pandas as pd
import arviz as az
import seaborn as sns
from matplotlib.patches import Patch

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(seed=1234)

###############################################################################
#### PyMC/Jax ODE conversion ####
###############################################################################
class SolOp(Op):
    """ Pytensor Op for the solution of an ODE system 
    using Diffrax with an associated pytensor gradient Op. See VJPSOLOp below. """
    def __init__(self, sol_op_jax_jitted, vjp_sol_op):
        self.sol_op_jax_jitted = sol_op_jax_jitted
        self.vjp_sol_op = vjp_sol_op

    def make_node(self, *inputs):
        # Convert our inputs to symbolic variables
        inputs = [pt.as_tensor_variable(inp) for inp in inputs]
        # Assume the output to always be a float64 matrix
        outputs = [pt.matrix()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        result = self.sol_op_jax_jitted(*inputs)
        outputs[0][0] = np.asarray(result, dtype="float64")
        
    def grad(self, inputs, output_grads):
        (gz,) = output_grads
        return self.vjp_sol_op(inputs, gz)
    
class VJPSolOp(Op):
    """ Pytensor Op for the gradient of the solution of an ODE system 
    using Diffrax """
    def __init__(self, vjp_sol_op_jax_jitted):
        self.vjp_sol_op_jax_jitted = vjp_sol_op_jax_jitted

    def make_node(self, inputs, gz):
        inputs = [pt.as_tensor_variable(inp) for inp in inputs] 
        inputs += [pt.as_tensor_variable(gz)]
        outputs = [inputs[i].type() for i in range(len(inputs)-1)]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        *params, gz = inputs
        result = self.vjp_sol_op_jax_jitted(gz, *params)
        for i, res in enumerate(result):
            outputs[i][0] = np.asarray(res, dtype="float64")

###############################################################################
#### Plotting Utils ####
###############################################################################
def export_legend(legend, filename="legend.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox, transparent=True)

def get_sized_fig_ax(width, height, hspace=1.0, vspace=0.5, fig_height=6, fig_width=6):
    fig = plt.figure(figsize=(fig_width, fig_height))
    # The first items are for padding and the second items are for the axes.
    # sizes are in inch.
    h = [Size.Fixed(hspace), Size.Fixed(width)]
    v = [Size.Fixed(vspace), Size.Fixed(height)]

    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    # The width and height of the rectangle are ignored.
    ax = fig.add_axes(divider.get_position(),
                    axes_locator=divider.new_locator(nx=1, ny=1))
    
    return fig, ax

def plot_predictive(inf_data, data, times, plot_prior=True, plot_post=True,
                    add_t_0=True, n_traces=200, figsize=(6, 4), prior_color='blue',
                    post_color='black', data_color='red', data_marker_size=10, 
                    cred_int=95, fig_ax = (None, None), linestyle='-',llike_name='llike'):
    """"plots prior and posterior predictive checks for the given model 
    along with the data supplied for inference"""

    # first create data frames for plotting of prior and posterior predictive checks
    if plot_prior: # if plotting prior, then convert prior predictive into a dataframe
        prior_sims = inf_data.prior_predictive[llike_name].values
        # Convert the nchains x ndraws x ntime Prior predictive into a dataframe
        nchains, ndraws, _, ntime = prior_sims.shape
        prior_sims_df = pd.DataFrame({
            'chain': np.repeat(np.arange(nchains), ndraws * ntime),
            'draw': np.tile(np.repeat(np.arange(ndraws), ntime), nchains),
            'time': np.tile(times, nchains * ndraws),
            'y': prior_sims.flatten()
        })
        # Add rows with time 0 and y 0 for each draw
        zero_time_rows = prior_sims_df.groupby(['chain', 'draw']).apply(lambda x: pd.DataFrame({
            'chain': [x['chain'].iloc[0]],
            'draw': [x['draw'].iloc[0]],
            'time': [0],
            'y': [0]
        })).reset_index(drop=True)
        prior_sims_df = pd.concat([prior_sims_df, zero_time_rows], 
                                    ignore_index=True).sort_values(by=['chain', 
                                    'draw', 'time']).reset_index(drop=True)
    
    if plot_post:
        # convert posterior predictive into a dataframe
        if type(inf_data) == az.InferenceData:
            post_sims = inf_data.posterior_predictive[llike_name].values
            nchains, ndraws, _, ntime = post_sims.shape
        elif type(inf_data) == np.ndarray:
            post_sims = inf_data
            nchains = 1
            ndraws, ntime = post_sims.shape
    
        post_sims_df = pd.DataFrame({
            'chain': np.repeat(np.arange(nchains), ndraws * ntime),
            'draw': np.tile(np.repeat(np.arange(ndraws), ntime), nchains),
            'time': np.tile(times, nchains * ndraws),
            'y': post_sims.flatten()
        })
        # Add rows with time 0 and y 0 for each draw
        zero_time_rows = post_sims_df.groupby(['chain', 'draw']).apply(lambda x: pd.DataFrame({
            'chain': [x.name[0]],
            'draw': [x.name[1]],
            'time': [0],
            'y': [0]
        })).reset_index(drop=True)
        post_sims_df = pd.concat([post_sims_df, zero_time_rows], 
                                    ignore_index=True).sort_values(by=['chain', 
                                'draw', 'time']).reset_index(drop=True)

    # plot predictive checks
    if fig_ax[0] is not None and fig_ax[1] is not None:
        fig, ax = fig_ax
    else:
        fig, ax = get_sized_fig_ax(figsize[0], figsize[1])
    if n_traces > 0:
        for i in range(n_traces):
            if plot_prior:
                if i == 0:
                    label = 'Prior'
                else:
                    label = None
                ax.plot(np.hstack((np.array([0]), times)), 
                        np.hstack((np.array([0]), np.squeeze(prior_sims[0, i, 0, :]))), 
                        color=prior_color, alpha=0.05, linewidth=0.5, label=label)
            elif plot_post:
                if i == 0:
                    label = 'Posterior'
                else:
                    label = None
                ax.plot(np.hstack((np.array([0]), times)), 
                        np.hstack((np.array([0]), np.squeeze(post_sims[0, i, 0, :]))), 
                        color=post_color, alpha=0.05, linewidth=0.5, label=label)
            
    # plot predictive densities
    if plot_prior:
        sns.lineplot(data=prior_sims_df, x='time', y='y', 
                    errorbar=("pi", cred_int), ax=ax, color=prior_color, 
                    label='Prior', linewidth=1.0)
    elif plot_post:
        sns.lineplot(data=post_sims_df, x='time', y='y',
                    errorbar=("pi", cred_int), ax=ax, color=post_color, 
                    label='Posterior predictive', linewidth=2.0, linestyle=linestyle)
    
    # plot data
    ax.scatter(times, data, color=data_color, s=data_marker_size, 
               label='Data', marker='x', linewidth=1.0)

    # label formatting
    ax.set_xlabel('')
    ax.set_ylabel('')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(8)

    # set min lims to 0 for x and y axes
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(0, ax.get_ylim()[1])
    
    leg = ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add shaded region to the legend item for the line on a seaborn lineplot
    handles, labels = ax.get_legend_handles_labels()
    print(handles)
    for i, label in enumerate(labels):
        if label == 'Posterior predictive':
            # handles[i] = Patch(facecolor=post_color, edgecolor='none', alpha=0.3)
            handles[i] = (handles[i], Patch(facecolor=post_color, edgecolor='none', alpha=0.3))
        elif label == 'Prior':
            handles[i] = Patch(facecolor=prior_color, edgecolor='none', alpha=0.3)
        elif label == 'Data':
            handles[i] = handles[i]
    leg  = ax.legend(handles=handles, labels=labels, fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')

    return fig, ax, leg