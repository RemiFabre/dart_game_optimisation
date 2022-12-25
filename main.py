import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable

numbers = [20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10, 6, 13, 4, 18, 1]

# "Official" board found here:
# https://www.dimensions.com/element/dartboard
angle_step = 2*math.pi/20.0
bull_eye_diam = 12.7
bull_green_diam = 32.0
border = 8.0
triple_ext_diam = 214.0
total_diam = 340.0

# Measurements made on my board.
# bull_eye_diam = 13.5
# bull_green_diam = 23.5
# border = 10.0
# triple_ext_diam = 212.5
# total_diam = 340.05


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


def plot_explore_ev_area(sorted_scores, only_top_x=None, save_img=True, name="youforgottonamethis"):
    final_xs = []
    final_ys = []
    final_scores = []
    i = 0
    min_val = 9999999
    max_val = 0
    for coords, value in sorted_scores.items():
        print(f"{value} at {coords}")
        final_xs.append(coords[0])
        final_ys.append(coords[1])
        final_scores.append(value)
        if value > max_val:
            max_val = value
        if value < min_val:
            min_val = value
        i += 1
        if only_top_x is not None:
            if i >= only_top_x:
                break

    plt.rcParams["figure.figsize"] = [10.0, 10.0]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax0 = fig.add_subplot(111, aspect='equal')
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    # plt.xlabel('$x axis$', fontsize=20)
    # plt.ylabel('$y axis$', fontsize=20)
    # Scatter plot.
    sc = ax0.scatter(np.array(final_ys)*-1.0, np.array(final_xs), c=np.array(final_scores), cmap='inferno')
    the_divider = make_axes_locatable(ax0)
    color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
    # Colorbar.
    ticks = np.linspace(min_val, max_val, 10, endpoint=True)
    cbar = plt.colorbar(sc, cax=color_axis, ticks=ticks)
    cbar.set_label('Average score for aiming spot', fontsize=21, labelpad=-2)

    draw_board(ax0)
    if save_img:
        filename = name
        if only_top_x is not None:
            filename += f"_top{only_top_x}"
        plt.savefig("img/"+filename, format="png",  bbox_inches='tight')
    plt.show()


def ev_area(x_avg, y_avg, sigma_x, sigma_y, size=100, plot=True, save_img=False, name="forgottoname"):
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

    if plot or save_img:
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
        if save_img:
            plt.savefig("img/"+name, format="png",  bbox_inches='tight')

        if plot:
            plt.show()
    return sum


def draw_board(ax=None):
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

    if ax is None:
        plt.gca().add_patch(circle1)
        plt.gca().add_patch(circle2)
        plt.gca().add_patch(circle3)
        plt.gca().add_patch(circle4)
        plt.gca().add_patch(circle5)
        plt.gca().add_patch(circle6)
    else:
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.add_patch(circle3)
        ax.add_patch(circle4)
        ax.add_patch(circle5)
        ax.add_patch(circle6)


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



# ## Average player 0.15, 0.09
# ev_area(0.0, 0.0, 0.15, 0.09, size=50, save_img=True, name="sx0.15_sy0.09_center")
# # The ideal shot apparently
# ev_area(-0.06, 0.24, 0.15, 0.09, size=50, save_img=True, name="sx0.15_sy0.09_ideal")
# # What happens when aiming at the triple 20
# ev_area((triple_ext_diam/2 - border/2)/total_diam, 0.0, 0.15, 0.09, size=50, save_img=True, name="sx0.15_sy0.09_triple_20")

# # sorted_scores = explore_ev_area(51, 0.15, 0.09, size=100)

# filename = "sorted_spots2601_size10000_sx0.15_sy0.09"
# sorted_scores = pickle.load(open(filename, "rb"))
# plot_explore_ev_area(sorted_scores, only_top_x=None, save_img=True, name=filename)
# plot_explore_ev_area(sorted_scores, only_top_x=100, save_img=True, name=filename)
# plot_explore_ev_area(sorted_scores, only_top_x=1, save_img=True, name=filename)


# # Good player 0.07, 0.07
# ev_area(0.0, 0.0, 0.07, 0.07, size=50, save_img=True, name="sx0.07_sy0.07_center")
# # The ideal shot apparently
# ev_area(-0.3, 0.12, 0.07, 0.07, size=50, save_img=True, name="sx0.07_sy0.07_ideal")
# # What happens when aiming at the triple 20
# ev_area((triple_ext_diam/2 - border/2)/total_diam, 0.0, 0.07, 0.07, size=50, save_img=True, name="sx0.07_sy0.07_triple_20")

# # sorted_scores = explore_ev_area(51, 0.07, 0.07, size=100)

# filename = "sorted_spots2601_size10000_sx0.07_sy0.07"
# sorted_scores = pickle.load(open(filename, "rb"))
# plot_explore_ev_area(sorted_scores, only_top_x=None, save_img=True, name=filename)
# plot_explore_ev_area(sorted_scores, only_top_x=100, save_img=True, name=filename)
# plot_explore_ev_area(sorted_scores, only_top_x=1, save_img=True, name=filename)


# ## Bad player 0.2, 0.2
# ev_area(0.0, 0.0, 0.2, 0.2, size=50, save_img=True, name="sx0.2_sy0.2_center")
# # The ideal shot apparently
# ev_area(-0.020, 0.0999999, 0.2, 0.2, size=50, save_img=True, name="sx0.2_sy0.2_ideal")
# # What happens when aiming at the triple 20
# ev_area((triple_ext_diam/2 - border/2)/total_diam, 0.0, 0.2, 0.2, size=50, save_img=True, name="sx0.2_sy0.2_triple_20")

# # sorted_scores = explore_ev_area(51, 0.2, 0.2, size=100)

# filename = "sorted_spots2601_size10000_sx0.2_sy0.2"
# sorted_scores = pickle.load(open(filename, "rb"))
# plot_explore_ev_area(sorted_scores, only_top_x=None, save_img=True, name=filename)
# plot_explore_ev_area(sorted_scores, only_top_x=100, save_img=True, name=filename)
# plot_explore_ev_area(sorted_scores, only_top_x=1, save_img=True, name=filename)


## Excellent player 0.02, 0.02
ev_area(0.0, 0.0, 0.02, 0.02, size=50, save_img=True, name="sx0.02_sy0.02_center")
# The ideal shot apparently
ev_area(0.3, 0.0, 0.02, 0.02, size=50, save_img=True, name="sx0.02_sy0.02_ideal")
# What happens when aiming at the triple 20
ev_area((triple_ext_diam/2 - border/2)/total_diam, 0.0, 0.02, 0.02, size=50, save_img=True, name="sx0.02_sy0.02_triple_20")

# sorted_scores = explore_ev_area(51, 0.02, 0.02, size=100)

filename = "sorted_spots2601_size10000_sx0.02_sy0.02"
sorted_scores = pickle.load(open(filename, "rb"))
plot_explore_ev_area(sorted_scores, only_top_x=None, save_img=True, name=filename)
plot_explore_ev_area(sorted_scores, only_top_x=100, save_img=True, name=filename)
plot_explore_ev_area(sorted_scores, only_top_x=1, save_img=True, name=filename)