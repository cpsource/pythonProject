# This is a sample Python script.
import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import make_checkerboard
from sklearn.cluster import SpectralBiclustering
from sklearn.metrics import consensus_score

v1 = [1, 2]
v2 = [-3, 4]
print(np.cross(v1, v2))

binned_latitude =  [0, 0, 1]
binned_longitude = [0, 1, 0]

print(np.cross(binned_latitude, binned_longitude))

lat = 0.1
lon = 10
# Note: lat must be gt 0 AND <= 10
if 0 < lat <= 10 and 0 < lon <= 15:
    print("The latitude and longitude are within the specified range.")
else:
    print("The latitude and longitude are not within the specified range.")

exit(0)

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

n_clusters = (4, 3)
data, rows, columns = make_checkerboard(
    shape=(300, 300), n_clusters=n_clusters, noise=10, shuffle=False, random_state=0
)

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Original dataset")

# for test, exit here
plt.show()
# exit(0)

# shuffle clusters
rng = np.random.RandomState(0)
row_idx = rng.permutation(data.shape[0])
col_idx = rng.permutation(data.shape[1])
data = data[row_idx][:, col_idx]

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Shuffled dataset")

# for test,exit here
plt.show()
exit(0)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    # np.fft.fftn()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
