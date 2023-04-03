# Almost all these shots were aiming at the center
points_r = [21, 92, 145, 217, 240, 91, 111, 134, 169, 206, 240, 30, 90, 155, 166, 200, 240, 49, 83, 127, 171, 211, 261,
            64, 138, 169, 190, 195, 228, 265, 36, 73, 102, 142, 208, 273, 45, 74, 115, 176, 205, 246, 35, 110, 146,
            221, 258, 43, 88, 122, 174, 217, 288, 85, 120, 138, 179, 209, 247, 41, 63, 89, 134, 149, 187, 292, 33,
            113, 143, 173, 189, 212, 265, 33, 57, 100, 183, 192, 231, 265
            ]

points_g = [67, 102, 125, 149, 176, 228, 240, 36, 84, 101, 130, 157, 182, 221, 246, 21, 53, 78, 131, 185, 203, 247, 19,
            63, 107, 139, 145, 171, 209, 226, 265, 43, 78, 97, 159, 186, 243, 31, 69, 107, 138, 218, 268, 29, 70, 97,
            120, 165, 201, 273, 8, 69, 155, 195, 226, 259, 57, 85, 109, 136, 196, 207, 249, 27, 41, 63, 90, 112, 151,
            164, 201, 253, 29, 55, 89, 125, 161, 188, 216, 262, 47, 107, 151, 195, 210, 258]

# 13 games, r 7 wins g 5 wins
# Results points_r: avg_set=42.25, avg_throw=14.083333333333334 => Equivalent variances = 0.0735, 0.0735
# Results points_g: avg_set=35.616279069767444, avg_throw=11.872093023255815 => Equivalent variances = 0.2, 0.2


def scores_to_avg_throw(scores):
    prev_score = 0
    nb_sets = 0
    sum = 0
    for score in scores:
        nb_sets += 1
        if score < prev_score:
            # New game
            diff = score
            prev_score = score
        else:
            diff = score - prev_score
            prev_score = score
        sum += diff
        print(diff)
    avg_set = sum/nb_sets
    avg_throw = avg_set/3.0
    print(f"avg_set={avg_set}, avg_throw={avg_throw}")

if __name__ == '__main__':
    # scores_to_avg_throw(points_r).
    scores_to_avg_throw(points_g)
