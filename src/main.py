import random

from file_handler import read_from_file
from network1 import Network1
from network2 import Network2
from plot import categorize_data, plot

data = read_from_file()
x0, y0, x1, y1 = categorize_data(data)
plot(x0, y0, x1, y1)

random.shuffle(data)
k = int(len(data) * 4 / 5)
training_data = data[0:k]
test_data = data[k + 1:len(data)]
print("network 1")
net = Network1(training_data, test_data)
net.SGD(3000, 3)
print("network 2")
net = Network2(training_data, test_data)
net.SGD(3000, 3)