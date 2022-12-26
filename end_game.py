import numpy as np
import pickle
import traceback
import sys

import aiming_spots

"""
    Idea : 
    - generate a dict for each aiming spot that give the probability of scoring X points.
    - Sort the dict, higher probabilities first (will be used to optimize by skipping low probas)
    - Brute force the case 300. i.e try all aiming spots and answer the question: how many throws on average to reach 301?
    - Solve for 299, but stop when reaching 300 since that value is known
    - Continue until 241?

    This gives an optimal strategy to minimize the number of throws to reach 301 (i.e where to aim for each end game scenario) for a given player.
    Caveat: the game is not played 1 throw each, but 3 throws each. We ignore this here as it's complex and not interesting enough.
    """


def create_empty_proba_dict():
    # All possible scores in 1 throw
    proba = {}
    proba[0] = 0.0
    proba[25] = 0.0
    proba[50] = 0.0
    for v in aiming_spots.numbers:
        proba[v] = 0.0
        proba[2*v] = 0.0
        proba[3*v] = 0.0
    return proba


def proba_area(x_avg, y_avg, sigma_x, sigma_y, size=100, cutoff_proba=0.001):
    mean = [x_avg, y_avg]
    cov = [[sigma_x**2, 0], [0, sigma_y**2]]
    proba = get_empty_proba()

    nb_throws = size**2
    final_xs, final_ys = np.random.multivariate_normal(mean, cov, nb_throws).T
    sum = 0
    for i in range(nb_throws):
        x = final_xs[i]
        y = final_ys[i]
        value, _ = aiming_spots.get_score(x, y)
        proba[int(value)] += 1
    # Transforming the results into probabilities and setting to 0 the very low ones
    for value, _ in proba.items():
        proba[value] = proba[value]/nb_throws
        if proba[value] <= cutoff_proba:
            proba[value] = 0
    sorted_proba = dict(sorted(proba.items(), key=lambda item: item[1], reverse=True))
    return sorted_proba


def explore_proba_area(points_per_line, sigma_x, sigma_y, size=100, cutoff_proba=0.001):
    proba_map = {}
    filename = f"proba{points_per_line**2}_size{size**2}_sx{sigma_x}_sy{sigma_y}"
    for i in range(points_per_line):
        for j in range(points_per_line):
            x = -0.5+i*(1/(points_per_line-1))
            y = -0.5+j*(1/(points_per_line-1))
            proba = proba_area(x, y, sigma_x, sigma_y, size=size, cutoff_proba=cutoff_proba)
            proba_map[(x, y)] = proba
            print(f"proba={proba} for x,y= {x}, {y} and sigma_x={sigma_x} sigma_y={sigma_y}")

    # Can't sort this
    # sorted_proba_map = dict(sorted(proba_map.items(), key=lambda item: item[1], reverse=True))
    pickle.dump(proba_map, open(filename, "wb"))
    return proba_map


def get_empty_proba():
    return {0: 0.0, 25: 0.0, 50: 0.0, 20: 0.0, 40: 0.0, 60: 0.0, 5: 0.0, 10: 0.0, 15: 0.0, 12: 0.0, 24: 0.0, 36: 0.0,
            9: 0.0, 18: 0.0, 27: 0.0, 14: 0.0, 28: 0.0, 42: 0.0, 11: 0.0, 22: 0.0, 33: 0.0, 8: 0.0, 16: 0.0, 32: 0.0,
            48: 0.0, 7: 0.0, 21: 0.0, 19: 0.0, 38: 0.0, 57: 0.0, 3: 0.0, 6: 0.0, 17: 0.0, 34: 0.0, 51: 0.0, 2: 0.0,
            4: 0.0, 30: 0.0, 45: 0.0, 13: 0.0, 26: 0.0, 39: 0.0, 54: 0.0, 1: 0.0}


def optimal_spot_for_score(current_score, points_per_line, sigma_x, sigma_y, size=100, proba_coverage=0.99):
    goal = 301
    # To be saved on disk and updated for each score. This is the final solution that will be updated score by score
    # by descending order
    scores_ev_and_pos = {}
    scores_ev_and_pos[301] = [0.0, "N/A"]
    filename = f"proba{points_per_line**2}_size{size**2}_sx{sigma_x}_sy{sigma_y}"
    print(f"Looking for a pickle file at: {filename}")
    try:
        with open(filename, "rb") as f:
            proba_map = pickle.load(f)
            print("File found! No need to redo the calculations")
    except Exception as e:
        # Catching all because we print the traceback and the action is always the same
        print(traceback.format_exc(e))
        proba_map = None

    if proba_map is None:
        print(f"No pickle file containing the calculations was found, doing the calculations (can take a while)")
        proba_map = explore_proba_area(points_per_line, sigma_x, sigma_y, size=size)
    # Scoring each aiming spot. The score is the expected number of throws to reach the goal score exactly
    spot_scores = {}
    for coords, proba in proba_map.items():
        spot_ev = 0.0
        # We need the probability of each value outcome. e.g if current_score==299, then we need :
        # P_301, P_300, P_f (proba of busting)
        # We also need the expected number of throws for each score other than the current one, so :
        # EV_300
        # then the expected number of throws to reach the goal is this infinite formula:
        # EV = P_301 * (1 + 2*P_f + 3*P_f**2 + 4*P_f**3 + ...) +
        #      P_300 * ( (1 + EV_300) + (2 + EV_300)*P_f + (3 + EV_300)*P_f**2 + (4 + EV_300)*P_f**3 )
        # The values P_301, P_300, P_f and EV_301 (==0), EV_300 will be stored in the dictionnary:
        proba_and_value = {}
        proba_and_value["bust"] = [0.0, "N/A"]
        for i in range(current_score, goal+1):
            # recovering the EV of a known score
            if i in scores_ev_and_pos:
                ev = scores_ev_and_pos[i]
            else:
                print(f"ERROR: score {i} is unknown in scores_ev_and_pos")
                sys.exit()
            proba_and_value[i] = [0.0, ev]
        # Updating the probability of each of these potential values:
        for value, p in proba.items():
            potential_value = current_score + value
            # Getting the EV of this score
            if potential_value > goal:
                proba_and_value["bust"][0] += p
            else:
                if potential_value in proba_and_value:
                    proba_and_value[potential_value][0] += p
                else:
                    print(f"ERROR: score {potential_value} is unknown in proba_and_value")
                    sys.exit()
        spot_ev = approximate_spot_ev(proba_and_value)
        spot_scores[coords] = spot_ev

    sorted_spot_scores = dict(sorted(spot_scores.items(), key=lambda item: item[1], reverse=False))
    for coords, ev in sorted_spot_scores:
        print(f"Best ev: {ev} at coords: {coords}")
        scores_ev_and_pos[current_score] = [ev, coords]
        break

    return scores_ev_and_pos, sorted_spot_scores


def approximate_spot_ev(proba_and_value, depth=5):
    # We need the probability of each value outcome. e.g if current_score==299, then we need :
    # P_301, P_300, P_f (proba of busting)
    # We also need the expected number of throws for each score other than the current one, so :
    # EV_300
    # then the expected number of throws to reach the goal is this infinite formula:
    # EV = P_301 * (1 + 2*P_f + 3*P_f**2 + 4*P_f**3 + ...) +
    #      P_300 * ( (1 + EV_300) + (2 + EV_300)*P_f + (3 + EV_300)*P_f**2 + (4 + EV_300)*P_f**3 )
    # The values P_301, P_300, P_f and EV_301 (==0), EV_300 are stored in proba_and_value
    ev = 0.0
    # To avoid recalculating:
    pf = proba_and_value["bust"][0]
    pfs = []
    for i in range(depth+1):
        # First one is 1, this is intended
        pfs.append(pf**i)
    for score in proba_and_value:
        if score == "bust":
            continue
        proba = proba_and_value[score][0]
        value = proba_and_value[score][1]
        for i in range(depth+1):
            ev += proba*pfs[i]*(i+1+value)
    return ev


def test_approximate_spot_ev():
    proba_and_value = {}
    proba_and_value["bust"] = [0.6, "N/A"]
    proba_and_value[301] = [0.4, 0.0]
    ev = approximate_spot_ev(proba_and_value, depth=0)
    print(f"result={ev}, expected={0.4}")
    ev = approximate_spot_ev(proba_and_value, depth=1)
    print(f"result={ev}, expected={0.8800000000000001}")
    ev = approximate_spot_ev(proba_and_value, depth=2)
    print(f"result={ev}, expected={1.3120000000000003}")


if __name__ == '__main__':
    # print(create_empty_proba_dict())

    # Testing on a good player that aims for triple 20:
    # print(proba_area((aiming_spots.triple_ext_diam/2 - aiming_spots.border/2) /
    #       aiming_spots.total_diam, 0.0, 0.07, 0.07, size=100, cutoff_proba=0.001))

    # Full proba map on a good player
    # print(explore_proba_area(51, 0.07, 0.07, size=100, cutoff_proba=0.001))

    # Testing approximate_spot_ev
    # test_approximate_spot_ev()

    scores_ev_and_pos, sorted_spot_scores = optimal_spot_for_score(300, 51, 0.07, 0.07, size=100, proba_coverage=0.99)
    print(sorted_spot_scores)
    print(scores_ev_and_pos)
