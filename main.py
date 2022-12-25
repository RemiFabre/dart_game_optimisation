import math
import numpy as np
import matplotlib.pyplot as plt
import pickle

numbers = [20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10, 6, 13, 4, 18, 1]

angle_step = 2*math.pi/20.0
bull_eye_diam = 12.7
bull_green_diam = 32.0
border = 8.0
triple_ext_diam = 107.0
total_diam = 170.0


# Calculate distance from center of board


def get_score(x, y):
    # between -0.5 and 0.5 to be inside
    # (0,0) it's the bull's eye, x+ is up, y+ is left
    # Normalizing between +-0.5
    x = x * (total_diam/2)/0.5
    y = y * (total_diam/2)/0.5
    distance = math.sqrt(x**2 + y**2)
    name = "ERROR"

    if distance <= bull_eye_diam/2:
        return 50.0, "bulls_eye"
    elif distance <= bull_green_diam/2:
        return 25.0, "green_bull"
    elif distance > total_diam / 2:
        return 0, "out"
    else:
        # inside
        theta = math.atan2(y, x)
        angle_index = int((((theta+angle_step/2) % (2*math.pi))/(2*math.pi))*20)
        number = numbers[angle_index]
        value = number
        name = "simple"
        # print(f"number={number}, x={x}, y={y}")
        if distance >= ((total_diam/2)-border):
            # double
            name = "double"
            value = value * 2
        elif distance < triple_ext_diam/2 and distance >= (triple_ext_diam/2 - border):
            # triple
            name = "triple"
            value = value*3
        name = name + " " + str(number)
        return value, name


def explore_darts(points_per_line):
    scores = {}
    for i in range(points_per_line):
        for j in range(points_per_line):
            x = -0.5+i*(1/(points_per_line-1))
            y = -0.5+j*(1/(points_per_line-1))
            score, name = get_score(x, y)
            scores[(x, y)] = (score, name)

    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1][0]))
    for coords, values in sorted_scores.items():
        print(f"{values} at {coords}")


def explore_ev_area(points_per_line, sigma_x, sigma_y, size=50):
    scores = {}
    size = 10
    filename = f"sorted_spots{points_per_line**2}_size{size**2}_sx{sigma_x}_sy{sigma_y}"
    for i in range(points_per_line):
        for j in range(points_per_line):
            x = -0.5+i*(1/(points_per_line-1))
            y = -0.5+j*(1/(points_per_line-1))
            ev = ev_area(x, y, sigma_x, sigma_y, size=size, plot=False)
            scores[(x, y)] = ev
            # print(f"EV={ev} for x,y= {x}, {y} and sigma_x={sigma_x} sigma_y={sigma_y}")

    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    pickle.dump(sorted_scores, open(filename, "wb"))
    return sorted_scores


def plot_explore_ev_area(sorted_scores):
    final_xs = []
    final_ys = []
    final_scores = []
    i = 0
    for coords, value in sorted_scores.items():
        print(f"{value} at {coords}")
        final_xs.append(coords[0])
        final_ys.append(coords[1])
        final_scores.append(value)
        i += 1
        if i >= 500:
            # pass
            break

    # plt.scatter(final_xs, final_ys, final_scores, cmap='hot')
        # Set the figure size
    plt.rcParams["figure.figsize"] = [10.0, 10.0]
    plt.rcParams["figure.autolayout"] = True
    # Inverting x->y and y->-x because the referentials in matpotlib are different that the one we choose for our board
    # plt.scatter(final_ys*-1.0, final_xs, marker=".")
    plt.scatter(np.array(final_ys)*-1.0, np.array(final_xs), c=np.array(final_scores), cmap='seismic')

    draw_board()
    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])

    plt.show()


def ev_area(x_avg, y_avg, sigma_x, sigma_y, size=100, plot=True):
    mean = [x_avg, y_avg]
    cov = [[sigma_x**2, 0], [0, sigma_y**2]]

    final_xs, final_ys = np.random.multivariate_normal(mean, cov, size**2).T
    final_scores = []
    sum = 0
    for i in range(len(final_xs)):
        x = final_xs[i]
        y = final_ys[i]
        value, _ = get_score(x, y)
        final_scores.append(value)
        sum += value
    sum = sum/(size**2)

    if plot:
        # plt.scatter(final_xs, final_ys, final_scores, cmap='hot')
        # Set the figure size
        plt.rcParams["figure.figsize"] = [10.0, 10.0]
        plt.rcParams["figure.autolayout"] = True
        # Inverting x->y and y->-x because the referentials in matpotlib are different that the one we choose for our board
        plt.scatter(final_ys*-1.0, final_xs, marker=".")
        draw_board()
        plt.text(0, 0, f"{sum:.1f}", fontsize=40)
        plt.xlim([-0.5, 0.5])
        plt.ylim([-0.5, 0.5])

        plt.show()
    return sum


def draw_board():
    # Bull's eye
    circle1 = plt.Circle((0.0, 0.0), bull_eye_diam/(total_diam*2), color='red',
                         clip_on=False, fill=False, linestyle="--", linewidth=3)
    # Green bull
    circle2 = plt.Circle((0.0, 0.0), bull_green_diam/(total_diam*2), color='red',
                         clip_on=False, fill=False, linestyle="--", linewidth=3)
    # Triples
    circle3 = plt.Circle((0.0, 0.0), triple_ext_diam/(total_diam*2), color='red',
                         clip_on=False, fill=False, linestyle="--", linewidth=3)
    circle4 = plt.Circle((0.0, 0.0), (triple_ext_diam/2 - border)/total_diam, color='red',
                         clip_on=False, fill=False, linestyle="--", linewidth=3)

    # Doubles
    circle5 = plt.Circle((0.0, 0.0), total_diam/(total_diam*2), color='red',
                         clip_on=False, fill=False, linestyle="--", linewidth=3)
    circle6 = plt.Circle((0.0, 0.0), (total_diam/2 - border)/total_diam, color='red',
                         clip_on=False, fill=False, linestyle="--", linewidth=3)

    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle2)
    plt.gca().add_patch(circle3)
    plt.gca().add_patch(circle4)
    plt.gca().add_patch(circle5)
    plt.gca().add_patch(circle6)

# Testing area

# print(get_score(0.25, 0.0))
# explore_darts(100)
# ev_area(0.0, 0.0, 0.2, 0.1, size=50)

# ## Testing perfect accuracy to check coordinates and stuff
# # Aiming at triple 20
# ev_area((triple_ext_diam/2 - border/2)/total_diam, 0.0, 0, 0, size=2)
# # Aiming at triple 14
# ev_area(0.0, (triple_ext_diam/2 - border/2)/total_diam, 0, 0, size=2)
# # Aiming at ~~triple 11
# ev_area(0.1, (triple_ext_diam/2 - border/2)/total_diam, 0, 0, size=2)

# # Probably what my throws look like:
ev_area(0.0, 0.0, 0.15, 0.09, size=50)


# sorted_scores = explore_ev_area(50, 0.15, 0.09, size=100)
# sorted_scores = pickle.load(open("sorted_spots2500_size10000_sx0.15_sy0.09", "rb"))
# plot_explore_ev_area(sorted_scores)
