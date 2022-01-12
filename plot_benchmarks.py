import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
c = ["darkred", "red", "lightcoral", "white", "palegreen", "green", "darkgreen"]
v = [0, .15, .4, .5, 0.6, .9, 1.]
l = list(zip(v, c))
cmap = LinearSegmentedColormap.from_list('rg', l, N=256)

df = pd.read_csv("yolo_cv_ph.csv", index_col="id")

y = ["slow", "normal", "fast"]
x = ["low", "medium", "high", "higher"]

result = {k: [] for k in y}
result["index"] = x

for speed in y:
    for cam in x:
        res = df[(df["camera"] == cam) & (df["speed"] == speed) & (df["detected"] == 1)]
        result[speed].append(res["solved"].mean())

temp_df = pd.DataFrame.from_dict(result).set_index("index")

ax, fig = plt.subplots()
sns.heatmap(temp_df, cmap=cmap, vmin=0.0, vmax=1.0, annot=True, square=True, fmt=".2f")
# fig.suptitle('test title', fontsize=12)
plt.ylabel("Camera angle")
plt.xlabel("Shuffle speed")
plt.savefig("yolo_cv_ph.pdf")


# Divide by number of shuffles
# for n in [3, 6, 10]:
#     result = {k: [] for k in y}
#     result["index"] = x
#
#     for speed in y:
#         for cam in x:
#             res = df[(df["number_of_shuffles"] == n) & (
#                 df["camera"] == cam) & (df["speed"] == speed) & (df["detected"] == 1)]
#             result[speed].append(res["solved"].mean())
#             print(res)
#             print(res["solved"].mean())
#
#     temp_df = pd.DataFrame.from_dict(result).set_index("index")
#
#     ax, fig = plt.subplots()
#     sns.heatmap(temp_df, cmap=cmap, vmin=0.0, vmax=1.0, annot=True, square=True)
#     # fig.suptitle('test title', fontsize=12)
#     plt.ylabel("Camera angle")
#     plt.xlabel("Shuffle speed")
#     plt.show()
#     break
