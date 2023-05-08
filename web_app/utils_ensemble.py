"""Plotting and downloading utilities."""
import ast
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import COLOR_MAP, scatter_plot, rescale


def plot(df, scaling="linear", metric_x="var", metric_y="bias", add_linear_fit=False):
  """Generate an interactive scatter plot."""

  test_sets = df.dataset.unique()
  assert len(test_sets) == 1
  test_set = test_sets[0]

  title = f"{metric_x} vs. {metric_y} {test_set}"
  fig = make_subplots(
    rows=1, cols=1, subplot_titles=((f"{title} ({scaling} scaling)"),),
  )
  traces = []
  for label, color in COLOR_MAP.items():
    traces.append(
      go.Scatter(x=[None], y=[None], mode='markers',
                 marker=dict(size=8, color=color),
                 showlegend=True, name=label),
      )

  def get_name(row):
    ensemble_hyper = ast.literal_eval(row.ensemble_hyperparameters)
    model_name = "Ensemble:<br>"
    model_name += "<br>".join([f"{model}" for model in ensemble_hyper['model_names']])
    model_name += "<br>"
    return model_name

  # Generate the main scatter plot
  traces.extend(
    scatter_plot(
      xs=df[f"{metric_x}"],
      ys=df[f"{metric_y}"],
      model_names=list(df.apply(get_name, axis=1)),
      scaling=scaling,
      add_linear_fit=add_linear_fit,
      #colors=df.model_family.apply(lambda x: COLOR_MAP[x]),
    )
  )

  metric_min_x, metric_max_x = df[f"{metric_x}"].min() - 0.1, df[f"{metric_x}"].max() + 0.1  # Avoid numerical issues
  metric_min_y, metric_max_y = df[f"{metric_y}"].min() - 0.1, df[f"{metric_y}"].max() + 0.1  # Avoid numerical issues
  if add_linear_fit:
    traces.append(
      go.Scatter(
        mode="lines",
        #x=rescale(np.arange(metric_min_x, metric_max_x + 0.01, 0.01), scaling),
        #y=rescale(np.arange(metric_min, metric_max + 0.01, 0.01), scaling),
        name="y=x",
        line=dict(color="black", dash="dashdot")
      )
    )

  for trace in traces:
    fig.add_trace(trace, row=1, col=1)

  ax_range = [rescale(metric_min_x, scaling), rescale(metric_max_x, scaling)]
  ay_range = [rescale(metric_min_y, scaling), rescale(metric_max_y, scaling)]
  fig.update_xaxes(title_text=f"{metric_x}", range=ax_range, row=1, col=1)
  fig.update_yaxes(title_text=f"{metric_y}", range=ay_range, row=1, col=1)
  #tickmarks = np.array([0.1, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, metric_max])
  ticks = dict(
    tickmode="array",
    #tickvals=rescale(tickmarks, scaling),
    #ticktext=[f"{mark:.2f}" for mark in tickmarks],
  )
  fig.update_layout(width=1000, height=700, xaxis=ticks, yaxis=ticks)
  return fig
