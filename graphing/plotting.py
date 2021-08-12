import os
import csv
import random

# import numpy as np

# import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# COLORS = {
#     "train": "rgb(0,119,187)",
#     "validation": "rgb(255,66,94)",
# }
COLORS = {
    "abiu": {"train": "#4184f3", "validation": "#db4437"},
    "caju-amarelo": {"train": "#0f9d58", "validation": "#c1175a"},
    "caju-vermelho": {"train": "#FF693B", "validation": "#00abc0"},
    "gabiroba": {"train": "#9746BB", "validation": "#6FBB36"},
    "pequi": {"train": "#5A41FF", "validation": "#EF55B8"},
    "siriguela": {"train": "#00786a", "validation": "#f4b400"},
}

# ["#4184f3", "#db4437", "#f4b400", "#0f9d58", "#aa46bb", "#00abc0", "#ff6f42", "#9d9c23", "#5b6abf", "#ef6191", "#00786a", "#c1175a"]

Y_TICKS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


def smooth(scalars, weight):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed


data_paths = []
for dirpath, dirnames, filenames in os.walk(os.path.join(os.path.curdir, "data")):
    if len(filenames) > 0:
        data_paths.append(dirpath)


for i, data_path in enumerate(data_paths):
    if "regular" in data_path:
        continue

    fruit = data_path.split("/")[2]

    fig = go.Figure()

    for mode in ("train", "validation"):

        accuracy_x = []
        accuracy_y = []

        with open(os.path.join(data_path, f"run-{mode}-tag-epoch_accuracy.csv")) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                accuracy_x.append(int(row["Step"]) + 1)
                accuracy_y.append(float(row["Value"]))

        fig.add_trace(
            go.Scatter(
                x=accuracy_x,
                y=accuracy_y,
                mode="lines",
                name=f"{mode} - raw",
                showlegend=False,
                opacity=0.2,
                line={"dash": "dot", "color": COLORS[fruit][mode]}
                if mode == "validation"
                else {"color": COLORS[fruit][mode]},
            )
        )

        fig.add_trace(
            go.Scatter(
                x=smooth(accuracy_x, 0.6),
                y=smooth(accuracy_y, 0.6),
                mode="lines",
                name=f"{mode}",
                line={"dash": "dot", "color": COLORS[fruit][mode]}
                if mode == "validation"
                else {"color": COLORS[fruit][mode]},
            )
        )

    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        yaxis=dict(tickmode="array", tickvals=Y_TICKS),
        yaxis_tickformat="%",
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5, font=dict(size=24)),
        legend_itemsizing="constant",
        font=dict(size=24),
        autosize=False,
        height=500,
        width=600,
        template="none",
        margin=dict(
            l=5,
            r=20,
            t=5,
            b=5,
        ),
    )
    fig.update_xaxes(linecolor="rgb(180,180,180)", gridcolor="rgb(210,210,210)", automargin=True)
    fig.update_yaxes(linecolor="rgb(180,180,180)", gridcolor="rgb(210,210,210)", automargin=True)

    fig_path = os.path.join(os.path.curdir, "figures", *data_path.split("/")[2:-1])
    os.makedirs(fig_path, exist_ok=True)

    fig.write_image(os.path.join(fig_path, f'{data_path.split("/")[-1]}.eps'))


# filtered_paths = list(filter(lambda data_path: "gabor" in data_path and "glcm" in data_path, data_paths))

# for mode in ("train", "validation"):

#     fig = make_subplots(specs=[[{"secondary_y": True}]])

#     for data_path in filtered_paths:
#         fruit = data_path.split("/")[2]

#         accuracy_x = []
#         accuracy_y = []

#         with open(os.path.join(data_path, f"run-{mode}-tag-epoch_accuracy.csv")) as csv_file:
#             csv_reader = csv.DictReader(csv_file)
#             for row in csv_reader:
#                 accuracy_x.append(int(row["Step"]) + 1)
#                 accuracy_y.append(float(row["Value"]))

#         fig.add_trace(
#             go.Scatter(
#                 x=accuracy_x,
#                 y=accuracy_y,
#                 mode="lines",
#                 name=f"{fruit} - raw",
#                 showlegend=False,
#                 line_color=COLORS[fruit]["train"],
#                 opacity=0.2,
#                 legendgroup="accuracy",
#             ),
#             secondary_y=False,
#         )

#         fig.add_trace(
#             go.Scatter(
#                 x=smooth(accuracy_x, 0.6),
#                 y=smooth(accuracy_y, 0.6),
#                 mode="lines",
#                 name=f"accuracy - {fruit}",
#                 line_color=COLORS[fruit]["train"],
#                 legendgroup="accuracy",
#             ),
#             secondary_y=False,
#         )

#         loss_x = []
#         loss_y = []

#         with open(os.path.join(data_path, f"run-{mode}-tag-epoch_loss.csv")) as csv_file:
#             csv_reader = csv.DictReader(csv_file)
#             for row in csv_reader:
#                 loss_x.append(int(row["Step"]) + 1)
#                 loss_y.append(float(row["Value"]))

#         fig.add_trace(
#             go.Scatter(
#                 x=loss_x,
#                 y=loss_y,
#                 mode="lines",
#                 name=f"{fruit} - raw",
#                 showlegend=False,
#                 line_color=COLORS[fruit]["validation"],
#                 opacity=0.2,
#                 legendgroup="loss",
#             ),
#             secondary_y=True,
#         )

#         fig.add_trace(
#             go.Scatter(
#                 x=smooth(loss_x, 0.6),
#                 y=smooth(loss_y, 0.6),
#                 mode="lines",
#                 name=f"loss - {fruit}",
#                 line_color=COLORS[fruit]["validation"],
#                 legendgroup="loss",
#             ),
#             secondary_y=True,
#         )

#     fig.update_layout(
#         xaxis_title="Epoch",
#         yaxis_title="Accuracy",
#         yaxis2_title="Loss",
#         yaxis=dict(tickmode="array", tickvals=Y_TICKS),
#         yaxis2=dict(tickmode="array", tickvals=Y_TICKS),
#         yaxis_tickformat="%",
#         yaxis2_range=[0, 1],
#         yaxis_range=[0, 1],
#         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.475, font=dict(size=20)),
#         legend_itemsizing="constant",
#         font=dict(size=20),
#         autosize=False,
#         height=500,
#         width=700,
#         template="none",
#         margin=dict(
#             l=10,
#             r=10,
#             t=10,
#             b=10,
#         ),
#     )
#     fig.update_xaxes(linecolor="rgb(180,180,180)", gridcolor="rgb(210,210,210)", automargin=True)
#     fig.update_yaxes(linecolor="rgb(180,180,180)", gridcolor="rgb(210,210,210)", automargin=True)

#     fig_path = os.path.join(os.path.curdir, "figures", "comparison")

#     os.makedirs(fig_path, exist_ok=True)
#     fig.write_image(os.path.join(fig_path, f"{mode}.png"))
