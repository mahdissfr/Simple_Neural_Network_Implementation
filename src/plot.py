from matplotlib.pyplot import figure, show

from file_handler import read_from_file


def categorize_data(d):
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    for i in range(len(d) - 1):
        if d[i][2] == 0:
            x0.append(d[i][0])
            y0.append(d[i][1])
        else:
            x1.append(d[i][0])
            y1.append(d[i][1])
    return x0, y0, x1, y1

def categorize_result(d):
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    for data in d:
        if data[2] == 0:
            x0.append(data[0])
            y0.append(data[1])
        else:
            x1.append(data[0])
            y1.append(data[1])
    return x0, y0, x1, y1


def plot(x0, y0, x1, y1):
    fig = figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.scatter(x0, y0, label="0", color="black", marker="x")
    ax.tick_params(axis='x', colors="black")
    ax.tick_params(axis='y', colors="black")

    ax2.scatter(x1, y1, label="1", color="blue", marker="x")
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.tick_params(axis='x', colors="black")
    ax2.tick_params(axis='y', colors="black")

    show()



