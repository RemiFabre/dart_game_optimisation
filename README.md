# Description
A simple Python program to solve the game of darts. More precisely, if the accuracy (2x2 covariance matrix) of a player is known, then the optimal aiming position can be calculated. This is achieved through a brute force method.

Should you aim for triple 20? For the bull's eye? Something else? The answer actually depends on your skill level.


# Results
Below several simulations for several types of players

## Average player with sigma_x=0.15 and sigma_y=0.09

With sigma_x=0.15 and sigma_y=0.09, a player that aims at the middle would get something like this after throwing 2601 darts:
![shots](img/sx0.15_sy0.09_center)

Knowing this, where should the player aim at to maximize the score per throw?
Below, 2601 aiming positions were tried, for each position 10 000 darts were "thrown" in simulation. The expected value for each of the 2061 aiming positions is then calculated and showcased below:
![2601 aiming positions scored](img/sorted_spots2601_size10000_sx0.15_sy0.09)
![best 100 shots](img/sorted_spots2601_size10000_sx0.15_sy0.09_top100)
The best aiming spot is at (-0.06, 0.24):
![best shot](img/sorted_spots2601_size10000_sx0.15_sy0.09_top1)

How much better is the optimal aiming spot compared to other common aiming spots?
Ideal shot 13.5 points per shot:
![best shot](img/sx0.15_sy0.09_ideal)
Center shot 12.5 points per shot:
![center shot](img/sx0.15_sy0.09_center)
triple 20 shot 11.9 points per shot:
![triple 20 shot](img/sx0.15_sy0.09_triple_20)


# References and articles about this subject
https://pages.cs.wisc.edu/~bolo/darts/x01.html

https://commons.wikimedia.org/wiki/File:Optimal_Dart_Throw_Location_Skill_Level.svg

https://www.stat.cmu.edu/~ryantibs/darts/

https://www.stat.cmu.edu/~ryantibs/papers/darts.pdf

