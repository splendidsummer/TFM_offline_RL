# Real Robot Challenge 2022 

[![license](https://img.shields.io/badge/license-GPLv2-blue.svg)](https://opensource.org/licenses/GPL-2.0)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/ait-bsc.svg?logo=python&logoColor=FFE873)](https://pypi.org/project/ait-bsc/)

## About this chanllenge
The goal of [challenge in 2022](https://real-robot-challenge.com/) is to solve dexterous manipulation tasks with offline reinforcement learning (RL) or imitation learning. The participants are provided with datasets containing dozens of hours of robotic data and can evaluate their policies remotely on a cluster of real TriFinger robots.
Participants can tackle two tasks during the real-robot stage:
* Pushing a cube to a target location on the ground and
* lifting the cube to match a target position and orientation in the air.
Like last year’s challenge, the Real Robot Challenge III is featured in the NeurIPS 2022 Competition Track.
![trifinger](trifingerpro_with_cube.jpg =500)

## Methodology 
### Offline RL 
There are 2 kinds of approaches in RL to solve this task, one is online RL(including on-policy & off-police methods), the other one is offline RL, which means the policy of actions is generate to mimic the behaviors of offline dataset. It should be noted that offline RL is surely a different notation from off-policy, in the offline setting, the agent no longer has the ability to interact with the environment and collect additional transitions using the behaviour policy. The learning algorithm is provided with a static dataset of fixed interaction, and must learn the best policy it can using this dataset.
We select 3 offline algorithms to solve this chanllenge, which are:
* **Standard BC (Bahavior Cloning)**
* **IQL (Implicit Q-Learning)**
* **TD3+BC (Twin Delayed DDPG)**
### Data augmentation by using discrete group symmetry

#### Morphorlogical Symmtry Analysis 

#### Data augmentation




## Offline Dataset
### Observation Space 



 
