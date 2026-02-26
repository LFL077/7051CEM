# -*- coding: utf-8 -*-
"""
Created on Mon Jun  61 2022

@author: LeandroLandgraf
"""

#%% Running optimiser and training using bayes_opt (22.06.2023)
'''
Reward function:
    angular_velocity = np.linalg.norm(self.ang_vel)
    angular_position = np.linalg.norm(self.ang_pos)
    distance_to_pad = np.linalg.norm(self.distance)
    linear_velocity = np.linalg.norm(self.lin_vel)
    
    self.reward += (
        - self.reward_options[0] # negative offset to discourage staying in the air
        + (self.reward_options[1] / offset_to_pad)  # encourage being near the pad
        + (self.reward_options[2] * progress_to_pad)  # encourage progress to landing pad
        - (self.reward_options[3] * abs(self.ang_vel[-1]))  # minimize spinning
        - (self.reward_options[4] * np.linalg.norm(self.ang_pos[:2]))  # penalize aggressive angles
        - (self.reward_options[5] * angular_velocity)  # not spinning
        - (self.reward_options[6] * angular_position)  # and upright
        - (self.reward_options[7] * linear_velocity)   # basically stopped
        - (self.reward_options[8] * distance_to_pad)   # we want to be at the pad
        )
    
Fitness:
    # check if we touched the landing pad
    if self.env.contact_array[self.env.drones[0].Id, self.landing_pad_id]:
        self.landing_pad_contact = 1.0
        self.reward += self.reward_options[9]
        self.fitness += 1 # improves fitness if touching pad
    else:
        self.landing_pad_contact = 0.0
        return

    # if our both velocities are less than 0.02 m/s and we upright, we LANDED!
    if (
        np.linalg.norm(self.previous_ang_vel) < 0.02
        and np.linalg.norm(self.previous_lin_vel) < 0.02
        and np.linalg.norm(self.ang_pos[:2]) < 0.1
    ):
        self.reward += self.reward_options[10] # completion bonus
        self.info["env_complete"] = True
        self.termination |= True
        self.fitness += 10 # greatly improve fitness if successful landing
        return
'''
import os
target_dir = 'C:\\Users\\LeandroLandgraf\\Documents\\10_TechnicalMastersProgramme\\5-6_Semester\\LandgrafL\\Git\\rocket_landing_sac'
os.chdir(target_dir)

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

import src.main
from wingman import Wingman

from time import time

def write_result(version, reward_gains, fitness):
    with open(f'weights/{results_file}', 'a') as f:
        print(f'Version number: {version}', file=f)
        print(f'Reward function vector: {reward_gains}', file=f)
        print(f'Fitness result: {fitness}', file=f)
        print('  ', file=f)
        
def black_box_function(x0, x1, x10, x2, x3, x4, x5, x6, x7, x8, x9):
    wm = Wingman(config_yaml="./src/settings.yaml")
    cfg = wm.cfg
    
    reward_gains = f"{x0}, {x1}, {x2}, {x3}, {x4}, {x5}, {x6}, {x7}, {x8}, {x9}, {x10}"
    
    cfg.reward_options = reward_gains
    src.main.train(wm)
    
    cfg.eval_fitness = True
    cfg.eval_num_episodes = 500
    fitness = src.main.eval_fitness(wm)
    
    write_result(cfg.version_number, cfg.reward_options, fitness)

    return fitness

start = time()

# Bounded region of parameter space
pbounds = {'x0': (0, 10), 'x1': (0, 10), 'x10': (500, 1000), 'x2': (0, 500), 
           'x3': (0, 10), 'x4': (0, 10), 'x5': (0, 10), 'x6': (0, 10), 
           'x7': (0, 10), 'x8': (0, 100), 'x9': (0, 100)}

reward_prior1 = [5.0, 2.0, 500.0, 100.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 20.0]
reward_prior2 = [3.0, 0.6, 930.0, 186.0, 6.8, 2.6, 3.5, 0.1, 3.6, 94.9, 21.8]

optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds,
                                 random_state=13)

# Saving progress
results_file = f'results_{time():.0f}.txt'
logger = JSONLogger(path=f"weights/{results_file}_bayes_opt.log")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
with open(f'weights/{results_file}', 'w') as f:
    print('Results', file=f)
    # print(optimizer.space.keys, file=f)
    print('   ', file=f)

# print(optimizer.space.keys)
# ['x0', 'x1', 'x10', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']

optimizer.maximize(init_points=3, n_iter=5)

optimizer.probe(params=reward_prior1, lazy=True)

optimizer.probe(params=reward_prior2, lazy=True)

optimizer.maximize(init_points=0, n_iter=15)

end = time()

print(optimizer.max)

with open(f'weights/{results_file}', 'a') as f:
    print('Final results', file=f)
    print(f'{optimizer.max}', file=f)
    print(f'Results obtained in {(end-start):.0f}s ({((end-start)/3600):.2f})h with 300k steps and 500 eval episodes', file=f)
    print('   ', file=f)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
    print('   ', file=f)

# Loading
load_logs(optimizer, logs=[f"weights/{results_file}_bayes_opt.log"]);
print("Optimizer is now aware of {} points.".format(len(optimizer.space)))

#%% Running optimiser and training using bayes_opt (25.06.2023)
'''
Reward function:
    ang_vel = self.state[:3]
    ang_pos = self.state[3:7]
    lin_vel = self.state[7:10]
    lin_pos = self.state[10:13]
    # action = self.state[13:20] # we don't need that for the rewards
    aux_state = self.state[20:29]
    landing_pad_contact_obs = self.state[29:30]
    rotated_distance = self.state[30:]
    
    
    angular_velocity = np.linalg.norm(ang_vel)
    angular_position = np.linalg.norm(ang_pos[:2])
    linear_velocity = np.linalg.norm(lin_vel)
    distance_to_pad = np.linalg.norm(lin_pos)
    lift_surface_0 = np.linalg.norm(aux_state[0])
    lift_surface_1 = np.linalg.norm(aux_state[1])
    lift_surface_2 = np.linalg.norm(aux_state[2])
    lift_surface_3 = np.linalg.norm(aux_state[3])
    ignition_steate = np.linalg.norm(aux_state[4])
    remaining_fuel = np.linalg.norm(aux_state[5])
    current_throttle = np.linalg.norm(aux_state[6])
    gimbal_state_0 = np.linalg.norm(aux_state[7])
    gimbal_state_1 = np.linalg.norm(aux_state[8])
    # landing_pad_contact = np.linalg.norm(landing_pad_contact_obs) # also don't need this, already included
    # distance_to_pad_rotated = np.linalg.norm(rotated_distance) # ans this one repeats the other


    angular_velocity = np.linalg.norm(self.ang_vel)
    angular_position = np.linalg.norm(self.ang_pos)
    distance_to_pad = np.linalg.norm(self.distance)
    linear_velocity = np.linalg.norm(self.lin_vel)
    
    self.reward += (
        - self.reward_options[0] 
        - (self.reward_options[1] * angular_velocity) 
        - (self.reward_options[2] * angular_position) 
        - (self.reward_options[3] * linear_velocity) 
        - (self.reward_options[4] * distance_to_pad) 
        - (self.reward_options[5] * lift_surface_0) 
        - (self.reward_options[6] * lift_surface_1) 
        - (self.reward_options[7] * lift_surface_2) 
        - (self.reward_options[8] * lift_surface_3) 
        - (self.reward_options[9] * ignition_steate) 
        - (self.reward_options[10] * remaining_fuel) 
        - (self.reward_options[11] * current_throttle) 
        - (self.reward_options[12] * gimbal_state_0) 
        - (self.reward_options[13] * gimbal_state_1) 
        )
    
Fitness:
    # check if we touched the landing pad
    if self.env.contact_array[self.env.drones[0].Id, self.landing_pad_id]:
        self.landing_pad_contact = 1.0
        self.reward += 50
        self.fitness += 1 # improves fitness if touching pad
    else:
        self.landing_pad_contact = 0.0
        return

    # if our both velocities are less than 0.02 m/s and we upright, we LANDED!
    if (
        np.linalg.norm(self.previous_ang_vel) < 0.02
        and np.linalg.norm(self.previous_lin_vel) < 0.02
        and np.linalg.norm(self.ang_pos[:2]) < 0.1
    ):
        self.reward += 1000 # completion bonus
        self.info["env_complete"] = True
        self.termination |= True
        self.fitness += 100 # greatly improve fitness if successful landing
        return
'''
import os
target_dir = 'C:\\Users\\LeandroLandgraf\\Documents\\10_TechnicalMastersProgramme\\5-6_Semester\\LandgrafL\\Git\\rocket_landing_sac'
os.chdir(target_dir)

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

import src.main
from wingman import Wingman

from time import time
import datetime

def write_result(version, reward_gains, fitness):
    with open(f'weights/{results_file}.txt', 'a') as f:
        print(f'Version number: {version}', file=f)
        print(f'Reward function vector: {reward_gains}', file=f)
        print(f'Fitness result: {fitness}', file=f)
        print('  ', file=f)
        
def black_box_function(x0, x1, x10, x11, x12, x13, x2, x3, x4, x5, x6, x7, x8, x9):
    wm = Wingman(config_yaml="./src/settings.yaml")
    cfg = wm.cfg
    
    reward_gains = f"{x0}, {x1}, {x2}, {x3}, {x4}, {x5}, {x6}, {x7}, {x8}, {x9}, {x10}, {x11}, {x12}, {x13}"
    
    cfg.reward_options = reward_gains
    src.main.train(wm)
    
    cfg.eval_fitness = True
    cfg.eval_num_episodes = 500
    cfg.eval_steps_ratio = 10000
    fitness = src.main.eval_fitness(wm)
    
    write_result(cfg.version_number, cfg.reward_options, fitness)

    return fitness

# Initial run, 200k steps and 50 eval during training - just to have an initial set of points
start = time()

# Bounded region of parameter space
pbounds = {'x0': (0, 10), 'x1': (0, 10), 'x10': (0, 10), 'x11': (0, 10), 
           'x12': (0, 10), 'x13': (0, 10), 'x2': (0, 10), 'x3': (0, 10), 
           'x4': (0, 100), 'x5': (0, 10), 'x6': (0, 10), 'x7': (0, 10), 
           'x8': (0, 100), 'x9': (0, 100)}

reward_prior = [5, 1, 0, 0, 0, 0, 1, 1, 10, 0, 0, 0, 0, 0] 

optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds,
                                 random_state=13)

# Saving progress
results_file = f'results_{datetime.date.today()}'
logger = JSONLogger(path=f"weights/{results_file}_bayes_opt.log")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
with open(f'weights/{results_file}.txt', 'w') as f:
    print('Results', file=f)
    # print(optimizer.space.keys, file=f)
    print(' ', file=f)

optimizer.probe(params=reward_prior, lazy=True)

optimizer.maximize(init_points=5, n_iter=10)

end = time()

print(optimizer.max)

with open(f'weights/{results_file}.txt', 'a') as f:
    print('Final results', file=f)
    print(f'{optimizer.max}', file=f)
    print('', file=f)
    print(f'Results obtained in {(end-start):.0f}s ({((end-start)/3600):.2f})h with 200k steps and 500 eval episodes', file=f)
    print('', file=f)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res), file=f)
    print('', file=f)

# # Loading
# load_logs(optimizer, logs=[f"weights/{results_file}_bayes_opt.log"]);
# print("Optimizer is now aware of {} points.".format(len(optimizer.space)))

# Running again with more steps, 300k, and 100 eval during training, using best results as priors
_ = input('Change total_steps = 300k and eval_num_episodes = 100 in the config file.')
_ = input('Check if bounds need updating and priors need including.')
start = time()

# Bounded region of parameter space
pbounds = {'x0': (0, 10), 'x1': (0, 10), 'x10': (0, 10), 'x11': (0, 10), 
           'x12': (0, 10), 'x13': (0, 10), 'x2': (0, 10), 'x3': (0, 10), 
           'x4': (0, 100), 'x5': (0, 10), 'x6': (0, 10), 'x7': (0, 10), 
           'x8': (0, 100), 'x9': (0, 100)}

reward_prior1 = [8.1, 2.8, 8.7, 7.9, 6.0, 0.9, 9.9, 7.3, 16.0, 5.6, 0.9, 5.3, 96.3, 88.8] #fitness=16
reward_prior2 = [1.9, 6.9, 9.0, 6.7, 5.2, 5.1, 5.0, 7.5, 6.1, 1.6, 7.9, 9.9, 19.2, 81.1] #fitness=23
reward_prior3 = [2.92, 8.90, 3.45, 9.09, 6.30, 1.63, 4.07, 1.87, 21.58, 3.71, 0.16, 5.31, 90.42, 30.56] #fitness=25

optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds,
                                 random_state=13)

optimizer.probe(params=reward_prior1, lazy=True)
optimizer.probe(params=reward_prior2, lazy=True)
optimizer.probe(params=reward_prior3, lazy=True)

optimizer.maximize(init_points=0, n_iter=10)

end = time()

print(optimizer.max)

with open(f'weights/{results_file}.txt', 'a') as f:
    print('Final results', file=f)
    print(f'{optimizer.max}', file=f)
    print('', file=f)
    print(f'Results obtained in {(end-start):.0f}s ({((end-start)/3600):.2f})h with 300k steps and 500 eval episodes', file=f)
    print('', file=f)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res), file=f)
    print('', file=f)

#%% Running optimiser and training using bayes_opt (28.06.2023)
'''
    # LfL: 28.06.2023
    # variable reward function 3 for optimisation: including all states and original function

    ang_vel = self.state[:3]
    ang_pos = self.state[3:7]
    lin_vel = self.state[7:10]
    lin_pos = self.state[10:13]
    # action = self.state[13:20] # we don't need that for the rewards
    aux_state = self.state[20:29]
    # landing_pad_contact_obs = self.state[29:30]
    # rotated_distance = self.state[30:]
    
    
    angular_velocity = np.linalg.norm(ang_vel)
    angular_position = np.linalg.norm(ang_pos[:2])
    linear_velocity = np.linalg.norm(lin_vel)
    distance_to_pad = np.linalg.norm(lin_pos)
    lift_surface_0 = np.linalg.norm(aux_state[0])
    lift_surface_1 = np.linalg.norm(aux_state[1])
    lift_surface_2 = np.linalg.norm(aux_state[2])
    lift_surface_3 = np.linalg.norm(aux_state[3])
    ignition_state = np.linalg.norm(aux_state[4])
    remaining_fuel = np.linalg.norm(aux_state[5])
    current_throttle = np.linalg.norm(aux_state[6])
    gimbal_state_0 = np.linalg.norm(aux_state[7])
    gimbal_state_1 = np.linalg.norm(aux_state[8])

    angular_velocity = np.linalg.norm(self.ang_vel)
    angular_position = np.linalg.norm(self.ang_pos)
    distance_to_pad = np.linalg.norm(self.distance)
    linear_velocity = np.linalg.norm(self.lin_vel)
    
    self.reward += (
        - (self.reward_options[0])
        - (self.reward_options[1] * angular_velocity) 
        - (self.reward_options[2] * angular_position) 
        - (self.reward_options[3] * linear_velocity) 
        - (self.reward_options[4] * distance_to_pad) 
        - (self.reward_options[5] * lift_surface_0) 
        - (self.reward_options[6] * lift_surface_1) 
        - (self.reward_options[7] * lift_surface_2) 
        - (self.reward_options[8] * lift_surface_3) 
        - (self.reward_options[9] * ignition_state) 
        + (self.reward_options[10] * remaining_fuel) 
        - (self.reward_options[11] * current_throttle) 
        - (self.reward_options[12] * gimbal_state_0) 
        - (self.reward_options[13] * gimbal_state_1) 
        )
    
    self.reward += (
        -self.reward_options[14] # negative offset to discourage staying in the air
        + (self.reward_options[15] / offset_to_pad)  # encourage being near the pad
        + (self.reward_options[16] * progress_to_pad)  # encourage progress to landing pad
        -(self.reward_options[17] * abs(self.ang_vel[-1]))  # minimize spinning
        - (self.reward_options[18] * np.linalg.norm(self.ang_pos[:2]))  # penalize aggressive angles
    )
    
Fitness:
    # check if we touched the landing pad
    if self.env.contact_array[self.env.drones[0].Id, self.landing_pad_id]:
        self.landing_pad_contact = 1.0
        self.reward += 50
        self.fitness += 1 # improves fitness if touching pad
    else:
        self.landing_pad_contact = 0.0
        return

    # if our both velocities are less than 0.02 m/s and we upright, we LANDED!
    if (
        np.linalg.norm(self.previous_ang_vel) < 0.02
        and np.linalg.norm(self.previous_lin_vel) < 0.02
        and np.linalg.norm(self.ang_pos[:2]) < 0.1
    ):
        self.reward += 1000 # completion bonus
        self.info["env_complete"] = True
        self.termination |= True
        self.fitness += 100 # greatly improve fitness if successful landing
        return
'''
import os
target_dir = 'C:\\Users\\LeandroLandgraf\\Documents\\10_TechnicalMastersProgramme\\5-6_Semester\\LandgrafL\\Git\\rocket_landing_sac'
os.chdir(target_dir)

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

import src.main
from wingman import Wingman

from time import time
import datetime

def write_result(version, reward_gains, fitness):
    with open(f'weights/{results_file}.txt', 'a') as f:
        print(f'Version number: {version}', file=f)
        print(f'Reward function vector: {reward_gains}', file=f)
        print(f'Fitness result: {fitness}', file=f)
        print('  ', file=f)
        
def black_box_function(x0, x1, x10, x11, x12, x13, x14, x15, x16, x17, x18, 
                       x2, x3, x4, x5, x6, x7, x8, x9):
    wm = Wingman(config_yaml="./src/settings.yaml")
    cfg = wm.cfg
    
    reward_gains = f"{x0:.3f}, {x1:.3f}, {x2:.3f}, {x3:.3f}, {x4:.3f}, {x5:.3f}, {x6:.3f}, {x7:.3f}, {x8:.3f}, {x9:.3f}, {x10:.3f}, {x11:.3f}, {x12:.3f}, {x13:.3f}, {x14:.3f}, {x15:.3f}, {x16:.3f}, {x17:.3f}, {x18:.3f}"
    
    cfg.reward_options = reward_gains
    src.main.train(wm)
    
    cfg.eval_fitness = True
    cfg.eval_num_episodes = 500
    fitness = src.main.eval_fitness(wm)
    
    write_result(cfg.version_number, cfg.reward_options, fitness)

    return fitness

# Initial run, 200k steps and 50 eval during training - just to have an initial set of points
start = time()

# Bounded region of parameter space
pbounds = {'x0': (0, 5), 'x1': (0, 5), 'x2': (0, 5), 'x3': (0, 5), 
           'x4': (0, 5), 'x5': (0, 5), 'x6': (0, 5), 'x7': (0, 5), 
           'x8': (0, 5), 'x9': (0, 50), 'x10': (0, 100), 'x11': (0, 50), 
           'x12': (0, 5), 'x13': (0, 5), 'x14': (0, 5), 'x15': (0, 5), 
           'x16': (0, 250), 'x17': (0, 5), 'x18': (0, 5)}

reward_prior0 = {'x0': 0, 'x1': 0, 'x2': 0, 'x3': 0, 'x4': 0, 'x5': 0, 'x6': 0, 
                 'x7': 0, 'x8': 0, 'x9': 0, 'x10': 0, 'x11': 0, 'x12': 0, 'x13': 0, 
                 'x14': 5, 'x15': 2, 'x16': 100, 'x17': 1, 'x18': 1} #original

reward_prior1 = {'x0': 1.9257769271036862, 'x1': 6.86484012963911, 
                 'x10': 8.988015042187246, 'x11': 6.663636522943425, 
                 'x12': 5.166503472018982, 'x13': 5.0502102417213655, 
                 'x2': 5.040880116516298, 'x3': 7.530186019248948, 
                 'x4': 6.115177489852308, 'x5': 1.5985614860103015, 
                 'x6': 7.89301764944393, 'x7': 9.8890505355358, 
                 'x8': 19.22820015208312, 'x9': 81.07714086490283, 
                 'x14': 0, 'x15': 0, 'x16': 0, 'x17': 0, 'x18': 0} 

reward_prior2 = {'x0': 2.9226253859788343, 'x1': 8.900371868960455, 
                 'x10': 3.4508019055848282, 'x11': 9.085173941784252, 
                 'x12': 6.3025213220839005, 'x13': 1.6305521810167412, 
                 'x2': 4.0726885285650685, 'x3': 1.8693250700265407, 
                 'x4': 21.582489361874003, 'x5': 3.705363882058086, 
                 'x6': 0.1634110526303889, 'x7': 5.3069114460626965, 
                 'x8': 90.42188726166759, 'x9': 30.555502735965277, 
                 'x14': 0, 'x15': 0, 'x16': 0, 'x17': 0, 'x18': 0} 

reward_prior3 = {'x0': 4.95940805942586, 'x1': 3.248659035012796, 
                 'x10': 6.172541826645568, 'x11': 1.7322395125035384, 
                 'x12': 7.609296040356076, 'x13': 4.601148831767579, 
                 'x2': 1.49410315290199, 'x3': 2.634930327198697, 
                 'x4': 94.49299243219, 'x5': 7.384849141086901, 
                 'x6': 1.4324114083856399, 'x7': 5.3925131329436224, 
                 'x8': 72.7965115872734, 'x9': 58.58802962819243, 
                 'x14': 0, 'x15': 0, 'x16': 0, 'x17': 0, 'x18': 0} 

reward_prior4 = {'x0': 2.5, 'x1': 1, 'x2': 1, 'x3': 1, 'x4': 1, 'x5': 1, 
                 'x6': 1, 'x7': 1, 'x8': 1, 'x9': 1, 'x10': 50, 'x11': 1, 
                 'x12': 1, 'x13': 1, 'x14': 2.5, 'x15': 2, 'x16': 100, 
                 'x17': 1, 'x18': 1}

optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds,
                                 random_state=13)

# Saving progress
results_file = f'results_{datetime.date.today()}'
logger = JSONLogger(path=f"weights/{results_file}_bayes_opt.log")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
with open(f'weights/{results_file}.txt', 'w') as f:
    print('Results', file=f)
    print(' ', file=f)

optimizer.probe(params=reward_prior0, lazy=True)
optimizer.probe(params=reward_prior1, lazy=True)
optimizer.probe(params=reward_prior2, lazy=True)
optimizer.probe(params=reward_prior3, lazy=True)
optimizer.probe(params=reward_prior4, lazy=True)

optimizer.maximize(init_points=10, n_iter=15)

optimizer.maximize(init_points=0, n_iter=15)

end = time()

print(optimizer.max)

with open(f'weights/{results_file}.txt', 'a') as f:
    print('Final results', file=f)
    print(f'{optimizer.max}', file=f)
    print('', file=f)
    print(f'Results obtained in {(end-start):.0f}s ({((end-start)/3600):.2f})h with 500k steps and 500 eval episodes', file=f)
    print('', file=f)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res), file=f)
    print('', file=f)

# os.system("shutdown /s /t 1")

# import subprocess
# subprocess.run(["shutdown", "-s"])

#%% Running optimiser and training using bayes_opt (05.07.2023)
'''
    # LfL: 28.06.2023
    # variable reward function 3 for optimisation: including all states and original function
    # same reward function, only adding negative terms for failure

    ang_vel = self.state[:3]
    ang_pos = self.state[3:7]
    lin_vel = self.state[7:10]
    lin_pos = self.state[10:13]
    # action = self.state[13:20] # we don't need that for the rewards
    aux_state = self.state[20:29]
    # landing_pad_contact_obs = self.state[29:30]
    # rotated_distance = self.state[30:]
    
    
    angular_velocity = np.linalg.norm(ang_vel)
    angular_position = np.linalg.norm(ang_pos[:2])
    linear_velocity = np.linalg.norm(lin_vel)
    distance_to_pad = np.linalg.norm(lin_pos)
    lift_surface_0 = np.linalg.norm(aux_state[0])
    lift_surface_1 = np.linalg.norm(aux_state[1])
    lift_surface_2 = np.linalg.norm(aux_state[2])
    lift_surface_3 = np.linalg.norm(aux_state[3])
    ignition_state = np.linalg.norm(aux_state[4])
    remaining_fuel = np.linalg.norm(aux_state[5])
    current_throttle = np.linalg.norm(aux_state[6])
    gimbal_state_0 = np.linalg.norm(aux_state[7])
    gimbal_state_1 = np.linalg.norm(aux_state[8])

    angular_velocity = np.linalg.norm(self.ang_vel)
    angular_position = np.linalg.norm(self.ang_pos)
    distance_to_pad = np.linalg.norm(self.distance)
    linear_velocity = np.linalg.norm(self.lin_vel)
    
    self.reward += (
        - (self.reward_options[0])
        - (self.reward_options[1] * angular_velocity) 
        - (self.reward_options[2] * angular_position) 
        - (self.reward_options[3] * linear_velocity) 
        - (self.reward_options[4] * distance_to_pad) 
        - (self.reward_options[5] * lift_surface_0) 
        - (self.reward_options[6] * lift_surface_1) 
        - (self.reward_options[7] * lift_surface_2) 
        - (self.reward_options[8] * lift_surface_3) 
        - (self.reward_options[9] * ignition_state) 
        + (self.reward_options[10] * remaining_fuel) 
        - (self.reward_options[11] * current_throttle) 
        - (self.reward_options[12] * gimbal_state_0) 
        - (self.reward_options[13] * gimbal_state_1) 
        )
    
    self.reward += (
        -self.reward_options[14] # negative offset to discourage staying in the air
        + (self.reward_options[15] / offset_to_pad)  # encourage being near the pad
        + (self.reward_options[16] * progress_to_pad)  # encourage progress to landing pad
        -(self.reward_options[17] * abs(self.ang_vel[-1]))  # minimize spinning
        - (self.reward_options[18] * np.linalg.norm(self.ang_pos[:2]))  # penalize aggressive angles
    )
    
Fitness:
    # check if we touched the landing pad
    if self.env.contact_array[self.env.drones[0].Id, self.landing_pad_id]:
        self.landing_pad_contact = 1.0
        # self.reward += 5000 # optimisation will maximize this, so giving a high value anyway
        self.fitness += 1 # improves fitness if touching pad
    else:
        self.landing_pad_contact = 0.0
        self.reward -= 5000
        return

    # if collision has more than 0.35 rad/s angular velocity, we dead
    # truthfully, if collision has more than 0.55 m/s linear velocity, we dead
    # number taken from here:
    # https://cosmosmagazine.com/space/launch-land-repeat-reusable-rockets-explained/
    # but doing so is kinda impossible for RL, so I've lessened the requirement to 1.0
    if (
        np.linalg.norm(self.previous_ang_vel) > 0.35
        or np.linalg.norm(self.previous_lin_vel) > 1.0
    ):
        self.info["fatal_collision"] = True
        self.termination |= True
        self.reward -= 10000
        return

    # if our both velocities are less than 0.02 m/s and we upright, we LANDED!
    if (
        np.linalg.norm(self.previous_ang_vel) < 0.02
        and np.linalg.norm(self.previous_lin_vel) < 0.02
        and np.linalg.norm(self.ang_pos[:2]) < 0.1
    ):
        # self.reward += 100000 # just giving a very high completion bonus
        self.info["env_complete"] = True
        self.termination |= True
        self.fitness += 100 # greatly improve fitness if successful landing
        return
'''
import os
target_dir = 'C:\\Users\\LeandroLandgraf\\Documents\\10_TechnicalMastersProgramme\\5-6_Semester\\LandgrafL\\Git\\rocket_landing_sac'
os.chdir(target_dir)

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

import src.main
from wingman import Wingman

from time import time
import datetime

def write_result(version, reward_gains, fitness):
    with open(f'weights/{results_file}.txt', 'a') as f:
        print(f'Version number: {version}', file=f)
        print(f'Reward function vector: {reward_gains}', file=f)
        print(f'Fitness result: {fitness}', file=f)
        print('  ', file=f)
        
def black_box_function(x0, x1, x10, x11, x12, x13, x14, x15, x16, x17, x18, 
                       x2, x3, x4, x5, x6, x7, x8, x9):
    wm = Wingman(config_yaml="./src/settings.yaml")
    cfg = wm.cfg
    
    reward_gains = f"{x0:.3f}, {x1:.3f}, {x2:.3f}, {x3:.3f}, {x4:.3f}, {x5:.3f}, {x6:.3f}, {x7:.3f}, {x8:.3f}, {x9:.3f}, {x10:.3f}, {x11:.3f}, {x12:.3f}, {x13:.3f}, {x14:.3f}, {x15:.3f}, {x16:.3f}, {x17:.3f}, {x18:.3f}"
    
    cfg.reward_options = reward_gains
    src.main.train(wm)
    
    cfg.eval_fitness = True
    cfg.eval_num_episodes = 200
    fitness = src.main.eval_fitness(wm)
    
    write_result(cfg.version_number, cfg.reward_options, fitness)

    return fitness

# Initial run, 200k steps and 50 eval during training - just to have an initial set of points
start = time()

# Bounded region of parameter space
pbounds1 = {'x0': (0, 5), 'x1': (0, 5), 'x2': (0, 5), 'x3': (0, 5), 
           'x4': (0, 5), 'x5': (0, 5), 'x6': (0, 5), 'x7': (0, 5), 
           'x8': (0, 5), 'x9': (0, 50), 'x10': (0, 100), 'x11': (0, 50), 
           'x12': (0, 5), 'x13': (0, 5), 'x14': (0, 10), 'x15': (0, 5), 
           'x16': (0, 250), 'x17': (0, 5), 'x18': (0, 5)}

optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds1,
                                 random_state=13)
# Saving progress
results_file = f'results_{datetime.date.today()}'
logger = JSONLogger(path=f"weights/{results_file}_bayes_opt.log")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
with open(f'weights/{results_file}.txt', 'w') as f:
    print('Results', file=f)
    print(' ', file=f)
    
    
reward_prior0 = {'x0': 0, 'x1': 0, 'x2': 0, 'x3': 0, 'x4': 0, 'x5': 0, 'x6': 0, 
                 'x7': 0, 'x8': 0, 'x9': 0, 'x10': 0, 'x11': 0, 'x12': 0, 'x13': 0, 
                 'x14': 5, 'x15': 2, 'x16': 100, 'x17': 1, 'x18': 1} #Fitness result: 57.0, version 634396

# optimizer.probe(params=reward_prior0, lazy=True)
# optimizer.maximize(init_points=0, n_iter=0)

# #new bounds
# pbounds2 = {'x0': (0, 5), 'x1': (0, 5), 'x2': (0, 5), 'x3': (0, 5), 
#             'x4': (0, 5), 'x5': (0, 5), 'x6': (0, 5), 'x7': (0, 5), 
#             'x8': (0, 5), 'x9': (0, 10), 'x10': (0, 50), 'x11': (0, 10), 
#             'x12': (0, 5), 'x13': (0, 5), 'x14': (0, 10), 'x15': (0, 5), 
#             'x16': (0, 150), 'x17': (0, 2), 'x18': (0, 2)}
# optimizer.set_bounds(new_bounds=pbounds2)

reward_prior1 = {"x0": 3.8885120528691006, "x1": 1.1877061001745615, 
                 "x2": 1.492247354445897, "x3": 0.29256245941037373, 
                 "x4": 4.285304712935995, "x5": 1.8642701393740375, 
                 "x6": 3.3992397578904847, "x7": 1.2813997466331506, 
                 "x8": 1.7379060757624598, "x9": 0.47063850404847063,
                 "x10": 82.42785326613685, "x11": 48.28745990214999, 
                 "x12": 4.863005569524467, "x13": 2.267246237086561, 
                 "x14": 3.0452123138063896, "x15": 3.8776325730242336, 
                 "x16": 160.4033361897673, "x17": 3.6100911475847357, 
                 "x18": 0.17518262050718658} #Fitness result: 27.0, version 570305

reward_prior2 = {"x0": 0.9499672366878598, "x1": 0.3904399759291255, 
                 "x2": 0.8209499411646332, "x3": 3.7037588508746353, 
                 "x4": 1.6766913055634536, "x5": 1.1700183562691246, 
                 "x6": 4.601512155099819, "x7": 0.8614565286488268, 
                 "x8": 2.9620019999547242, "x9": 41.18575494560382,
                 "x10": 58.828344818244105, "x11": 48.56458723069614, 
                 "x12": 0.27626760240260173, "x13": 4.017453416278955, 
                 "x14": 4.615813073764067, "x15": 0.7320372939009356, 
                 "x16": 218.2621922814655, "x17": 4.478961272014756, 
                 "x18": 1.492132428953688} #Fitness result: 32.0, version 475780

optimizer.probe(params=reward_prior1, lazy=True)
optimizer.probe(params=reward_prior2, lazy=True)

optimizer.maximize(init_points=0, n_iter=15)

end = time()

print(optimizer.max)

with open(f'weights/{results_file}.txt', 'a') as f:
    print('Final results', file=f)
    print(f'{optimizer.max}', file=f)
    print('', file=f)
    print(f'Results obtained in {(end-start):.0f}s ({((end-start)/3600):.2f})h with 200k steps and 100 eval episodes', file=f)
    print('', file=f)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res), file=f)
    print('', file=f)

# os.system("shutdown /s /t 1")

# import subprocess
# subprocess.run(["shutdown", "-s"])

#%% Running optimiser and training using bayes_opt (06.07.2023)
'''
    # LfL: 28.06.2023
    # variable reward function 3 for optimisation: including all states and original function
    # same reward function, only adding negative terms for failure

    ang_vel = self.state[:3]
    ang_pos = self.state[3:7]
    lin_vel = self.state[7:10]
    lin_pos = self.state[10:13]
    # action = self.state[13:20] # we don't need that for the rewards
    aux_state = self.state[20:29]
    # landing_pad_contact_obs = self.state[29:30]
    # rotated_distance = self.state[30:]
    
    
    angular_velocity = np.linalg.norm(ang_vel)
    angular_position = np.linalg.norm(ang_pos[:2])
    linear_velocity = np.linalg.norm(lin_vel)
    distance_to_pad = np.linalg.norm(lin_pos)
    lift_surface_0 = np.linalg.norm(aux_state[0])
    lift_surface_1 = np.linalg.norm(aux_state[1])
    lift_surface_2 = np.linalg.norm(aux_state[2])
    lift_surface_3 = np.linalg.norm(aux_state[3])
    ignition_state = np.linalg.norm(aux_state[4])
    remaining_fuel = np.linalg.norm(aux_state[5])
    current_throttle = np.linalg.norm(aux_state[6])
    gimbal_state_0 = np.linalg.norm(aux_state[7])
    gimbal_state_1 = np.linalg.norm(aux_state[8])

    angular_velocity = np.linalg.norm(self.ang_vel)
    angular_position = np.linalg.norm(self.ang_pos)
    distance_to_pad = np.linalg.norm(self.distance)
    linear_velocity = np.linalg.norm(self.lin_vel)
    
    self.reward += (
        - (self.reward_options[0])
        - (self.reward_options[1] * angular_velocity) 
        - (self.reward_options[2] * angular_position) 
        - (self.reward_options[3] * linear_velocity) 
        - (self.reward_options[4] * distance_to_pad) 
        - (self.reward_options[5] * lift_surface_0) 
        - (self.reward_options[6] * lift_surface_1) 
        - (self.reward_options[7] * lift_surface_2) 
        - (self.reward_options[8] * lift_surface_3) 
        - (self.reward_options[9] * ignition_state) 
        + (self.reward_options[10] * remaining_fuel) 
        - (self.reward_options[11] * current_throttle) 
        - (self.reward_options[12] * gimbal_state_0) 
        - (self.reward_options[13] * gimbal_state_1) 
        )
    
    self.reward += (
        -self.reward_options[14] # negative offset to discourage staying in the air
        + (self.reward_options[15] / offset_to_pad)  # encourage being near the pad
        + (self.reward_options[16] * progress_to_pad)  # encourage progress to landing pad
        -(self.reward_options[17] * abs(self.ang_vel[-1]))  # minimize spinning
        - (self.reward_options[18] * np.linalg.norm(self.ang_pos[:2]))  # penalize aggressive angles
    )
    
Fitness:
    # check if we touched the landing pad
    if self.env.contact_array[self.env.drones[0].Id, self.landing_pad_id]:
        self.landing_pad_contact = 1.0
        self.reward += 50 # optimisation will maximize this, so giving a high value anyway
        self.fitness += 1 # improves fitness if touching pad
    else:
        self.landing_pad_contact = 0.0
        self.reward -= 500
        return

    # if collision has more than 0.35 rad/s angular velocity, we dead
    # truthfully, if collision has more than 0.55 m/s linear velocity, we dead
    # number taken from here:
    # https://cosmosmagazine.com/space/launch-land-repeat-reusable-rockets-explained/
    # but doing so is kinda impossible for RL, so I've lessened the requirement to 1.0
    if (
        np.linalg.norm(self.previous_ang_vel) > 0.35
        or np.linalg.norm(self.previous_lin_vel) > 1.0
    ):
        self.info["fatal_collision"] = True
        self.termination |= True
        self.reward -= 10000
        return

    # if our both velocities are less than 0.02 m/s and we upright, we LANDED!
    if (
        np.linalg.norm(self.previous_ang_vel) < 0.02
        and np.linalg.norm(self.previous_lin_vel) < 0.02
        and np.linalg.norm(self.ang_pos[:2]) < 0.1
    ):
        self.reward += 1000 # just giving a very high completion bonus
        self.info["env_complete"] = True
        self.termination |= True
        self.fitness += 100 # greatly improve fitness if successful landing
        return
'''
import os
target_dir = 'C:\\Users\\LeandroLandgraf\\Documents\\10_TechnicalMastersProgramme\\5-6_Semester\\LandgrafL\\Git\\rocket_landing_sac'
os.chdir(target_dir)

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

import src.main
from wingman import Wingman

from time import time
import datetime

def write_result(version, reward_gains, fitness):
    with open(f'weights/{results_file}.txt', 'a') as f:
        print(f'Version number: {version}', file=f)
        print(f'Reward function vector: {reward_gains}', file=f)
        print(f'Fitness result: {fitness}', file=f)
        print('  ', file=f)
        
def black_box_function(x0, x1, x10, x11, x12, x13, x14, x15, x16, x17, x18, 
                       x2, x3, x4, x5, x6, x7, x8, x9):
    wm = Wingman(config_yaml="./src/settings.yaml")
    cfg = wm.cfg
    
    reward_gains = f"{x0:.3f}, {x1:.3f}, {x2:.3f}, {x3:.3f}, {x4:.3f}, {x5:.3f}, {x6:.3f}, {x7:.3f}, {x8:.3f}, {x9:.3f}, {x10:.3f}, {x11:.3f}, {x12:.3f}, {x13:.3f}, {x14:.3f}, {x15:.3f}, {x16:.3f}, {x17:.3f}, {x18:.3f}"
    
    cfg.reward_options = reward_gains
    src.main.train(wm)
    
    cfg.eval_fitness = True
    cfg.eval_num_episodes = 500
    fitness = src.main.eval_fitness(wm)
    
    write_result(cfg.version_number, cfg.reward_options, fitness)

    return fitness

# Initial run, 200k steps and 50 eval during training - just to have an initial set of points
start = time()

# Bounded region of parameter space
pbounds1 = {'x0': (0, 5), 'x1': (0, 5), 'x2': (0, 5), 'x3': (0, 5), 
           'x4': (0, 5), 'x5': (0, 5), 'x6': (0, 5), 'x7': (0, 5), 
           'x8': (0, 5), 'x9': (0, 50), 'x10': (0, 100), 'x11': (0, 50), 
           'x12': (0, 5), 'x13': (0, 5), 'x14': (0, 10), 'x15': (0, 5), 
           'x16': (0, 250), 'x17': (0, 5), 'x18': (0, 5)}

optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds1,
                                 random_state=13)
# Saving progress
results_file = f'results_{datetime.date.today()}'
logger = JSONLogger(path=f"weights/{results_file}_bayes_opt.log")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
with open(f'weights/{results_file}.txt', 'w') as f:
    print('Results', file=f)
    print(' ', file=f)
    
    
reward_prior0 = {'x0': 0, 'x1': 0, 'x2': 0, 'x3': 0, 'x4': 0, 'x5': 0, 'x6': 0, 
                 'x7': 0, 'x8': 0, 'x9': 0, 'x10': 0, 'x11': 0, 'x12': 0, 'x13': 0, 
                 'x14': 5, 'x15': 2, 'x16': 100, 'x17': 1, 'x18': 1} #Fitness result: 57.0, version 634396

reward_prior1 = {"x0": 3.8885120528691006, "x1": 1.1877061001745615, 
                 "x2": 1.492247354445897, "x3": 0.29256245941037373, 
                 "x4": 4.285304712935995, "x5": 1.8642701393740375, 
                 "x6": 3.3992397578904847, "x7": 1.2813997466331506, 
                 "x8": 1.7379060757624598, "x9": 0.47063850404847063,
                 "x10": 82.42785326613685, "x11": 48.28745990214999, 
                 "x12": 4.863005569524467, "x13": 2.267246237086561, 
                 "x14": 3.0452123138063896, "x15": 3.8776325730242336, 
                 "x16": 160.4033361897673, "x17": 3.6100911475847357, 
                 "x18": 0.17518262050718658} #Fitness result: 27.0, version 570305

reward_prior2 = {"x0": 0.9499672366878598, "x1": 0.3904399759291255, 
                 "x2": 0.8209499411646332, "x3": 3.7037588508746353, 
                 "x4": 1.6766913055634536, "x5": 1.1700183562691246, 
                 "x6": 4.601512155099819, "x7": 0.8614565286488268, 
                 "x8": 2.9620019999547242, "x9": 41.18575494560382,
                 "x10": 58.828344818244105, "x11": 48.56458723069614, 
                 "x12": 0.27626760240260173, "x13": 4.017453416278955, 
                 "x14": 4.615813073764067, "x15": 0.7320372939009356, 
                 "x16": 218.2621922814655, "x17": 4.478961272014756, 
                 "x18": 1.492132428953688} #Fitness result: 32.0, version 475780

optimizer.probe(params=reward_prior0, lazy=True)
optimizer.probe(params=reward_prior1, lazy=True)
optimizer.probe(params=reward_prior2, lazy=True)
optimizer.maximize(init_points=0, n_iter=0)

# #new bounds
pbounds2 = {'x0': (0, 1), 'x1': (0, 1), 'x2': (0, 1), 'x3': (0, 1), 
            'x4': (0, 1), 'x5': (0, 1), 'x6': (0, 1), 'x7': (0, 1), 
            'x8': (0, 1), 'x9': (0, 2), 'x10': (0, 10), 'x11': (0, 2), 
            'x12': (0, 1), 'x13': (0, 1), 'x14': (0, 0), 'x15': (0, 1), 
            'x16': (0, 300), 'x17': (0, 0), 'x18': (0, 0)}
optimizer.set_bounds(new_bounds=pbounds2)

optimizer.maximize(init_points=10, n_iter=15)

end = time()

print(optimizer.max)

with open(f'weights/{results_file}.txt', 'a') as f:
    print('Final results', file=f)
    print(f'{optimizer.max}', file=f)
    print('', file=f)
    print(f'Results obtained in {(end-start):.0f}s ({((end-start)/3600):.2f})h with 200k steps and 100 eval episodes', file=f)
    print('', file=f)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res), file=f)
    print('', file=f)

# os.system("shutdown /s /t 1")

# import subprocess
# subprocess.run(["shutdown", "-s"])

#%% Running optimiser and training using pyro
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to

import pyro
import pyro.contrib.gp as gp # Gaussian Process

pyro.set_rng_seed(13)

def f(x):
    '''
    This function has both a local minimum and a global minimum. The global 
    minimum is at x* = 0.75725
    '''
    return (6 * x - 2)**2 * torch.sin(12 * x - 4)

x = torch.linspace(0, 1, 100)
plt.figure(figsize=(8, 4))
plt.plot(x.numpy(), f(x).numpy())
plt.show()

# initialize the model with four input points: 0.0, 0.33, 0.66, 1.0
X = torch.tensor([0.0, 0.33, 0.66, 1.0])
y = f(X)

'''
We choose the Matern kernel with v = 5/2. Note that the popular RBF kernel, 
which is used in many regression tasks, results in a function prior whose samples 
are infinitely differentiable; this is probably an unrealistic assumption for 
most ‘black-box’ objective functions.
'''
gpmodel = gp.models.GPRegression(X, y, gp.kernels.Matern52(input_dim=1), noise=torch.tensor(0.1), jitter=1.0e-4)

def update_posterior(x_new):
    ''' 
    This helper function will take care of updating the gpmodel each time we
    evaluate f at a new value x
    '''
    y = f(x_new) # evaluate f at new point.
    X = torch.cat([gpmodel.X, x_new]) # incorporate new evaluation
    y = torch.cat([gpmodel.y, y])
    gpmodel.set_data(X, y)
    # optimize the GP hyperparameters using Adam with lr=0.001
    optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.001)
    gp.util.train(gpmodel, optimizer)

def lower_confidence_bound(x, kappa=2):
    ''' 
    Define an acquisition function
    i) mu is small (exploitation); or
    ii) sigma (variance) is large (exploration)
    A large value of kappa means that we place more weight on exploration because 
    we prefer candidates x in areas of high uncertainty. A small value of kappa 
    encourages exploitation because we prefer candidates x that minimize mu, which 
    is the mean of our surrogate objective function. 
    '''
    mu, variance = gpmodel(x, full_cov=False, noiseless=False)
    sigma = variance.sqrt()
    return mu - kappa * sigma

def find_a_candidate(x_init, lower_bound=0, upper_bound=1):
    '''The final component we need is a way to find (approximate) minimizing 
    points of the acquisition function. There are several ways to proceed, 
    including gradient-based and non-gradient-based techniques. Here we will 
    follow the gradient-based approach. One of the possible drawbacks of gradient 
    descent methods is that the minimization algorithm can get stuck at a local 
    minimum.
    '''
    # transform x to an unconstrained domain
    constraint = constraints.interval(lower_bound, upper_bound)
    unconstrained_x_init = transform_to(constraint).inv(x_init)
    unconstrained_x = unconstrained_x_init.clone().detach().requires_grad_(True)
    minimizer = optim.LBFGS([unconstrained_x], line_search_fn='strong_wolfe')

    def closure():
        minimizer.zero_grad()
        x = transform_to(constraint)(unconstrained_x)
        y = lower_confidence_bound(x)
        autograd.backward(unconstrained_x, autograd.grad(y, unconstrained_x))
        return y

    minimizer.step(closure)
    # after finding a candidate in the unconstrained domain,
    # convert it back to original domain.
    x = transform_to(constraint)(unconstrained_x)
    return x.detach()

def next_x(lower_bound=0, upper_bound=1, num_candidates=5):
    ''' With the various helper functions defined above, we can now encapsulate 
    the main logic of a single step of Bayesian Optimization
    '''
    candidates = []
    values = []

    x_init = gpmodel.X[-1:]
    for i in range(num_candidates):
        x = find_a_candidate(x_init, lower_bound, upper_bound)
        y = lower_confidence_bound(x)
        candidates.append(x)
        values.append(y)
        x_init = x.new_empty(1).uniform_(lower_bound, upper_bound)

    argmin = torch.min(torch.cat(values), dim=0)[1].item()
    return candidates[argmin]

def plot(gs, xmin, xlabel=None, with_title=True):
    xlabel = "xmin" if xlabel is None else "x{}".format(xlabel)
    Xnew = torch.linspace(-0.1, 1.1, 100)
    ax1 = plt.subplot(gs[0])
    ax1.plot(gpmodel.X.numpy(), gpmodel.y.numpy(), "kx")  # plot all observed data
    with torch.no_grad():
        loc, var = gpmodel(Xnew, full_cov=False, noiseless=False)
        sd = var.sqrt()
        ax1.plot(Xnew.numpy(), loc.numpy(), "r", lw=2)  # plot predictive mean
        ax1.fill_between(Xnew.numpy(), loc.numpy() - 2*sd.numpy(), loc.numpy() + 2*sd.numpy(),
                         color="C0", alpha=0.3)  # plot uncertainty intervals
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_title("Find {}".format(xlabel))
    if with_title:
        ax1.set_ylabel("Gaussian Process Regression")

    ax2 = plt.subplot(gs[1])
    with torch.no_grad():
        # plot the acquisition function
        ax2.plot(Xnew.numpy(), lower_confidence_bound(Xnew).numpy())
        # plot the new candidate point
        ax2.plot(xmin.numpy(), lower_confidence_bound(xmin).numpy(), "^", markersize=10,
                 label="{} = {:.5f}".format(xlabel, xmin.item()))
    ax2.set_xlim(-0.1, 1.1)
    if with_title:
        ax2.set_ylabel("Acquisition Function")
    ax2.legend(loc=1)

#------------------------------------------------------------------------------

plt.figure(figsize=(12, 30))
outer_gs = gridspec.GridSpec(5, 2)
optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.001)
gp.util.train(gpmodel, optimizer)
for i in range(8):
    xmin = next_x()
    gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[i])
    plot(gs, xmin, xlabel=i+1, with_title=(i % 2 == 0))
    update_posterior(xmin)
plt.show()






