# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 2023

@author: LeandroLandgraf
"""

#%% 
'''
    if distance_to_pad > 200:
        self.reward += (
            - self.reward_options[0] # negative offset to discourage staying in the air
            + (self.reward_options[1] * progress_to_pad)  # encourage progress to landing pad
            - (self.reward_options[2] * (1 - remaining_fuel))
            )
    else:
        self.reward += (
            + (self.reward_options[3] * progress_to_pad) # encourage progress to landing pad
            + (self.reward_options[4] * (self.ang_vel - self.previous_ang_vel)) # minimize spinning
            - (self.reward_options[5] * angular_position) # penalize aggressive angles
            + (self.reward_options[6] * (self.lin_vel - self.previous_lin_vel)) # penalize if moving fast
            )

    # composite reward together
    self.reward += (
        - self.reward_options[14] # negative offset to discourage staying in the air
        + (self.reward_options[15] / offset_to_pad)  # encourage being near the pad
        + (self.reward_options[16] * progress_to_pad)  # encourage progress to landing pad
        - (self.reward_options[17] * abs(self.ang_vel[-1]))  # minimize spinning
        - (self.reward_options[18] * np.linalg.norm(self.ang_pos[:2]))  # penalize aggressive angles
    )
'''
import os
target_dir = 'C:\\Users\\LeandroLandgraf\\Documents\\10_TechnicalMastersProgramme\\5-6_Semester\\LandgrafL\\Git\\rocket_landing_sac'
os.chdir(target_dir)

import src.main
from wingman import Wingman
# import wandb

from time import time
import datetime

def create_log(results_file):
    with open(f'weights/{results_file}', 'w') as f:
        print('Results', file=f)
        print('', file=f)

def write_result(version, reward_gains, fitness, results_file):
    with open(f'weights/{results_file}', 'a') as f:
        print(f'Version number: {version}', file=f)
        print(f'Reward function vector: {reward_gains}', file=f)
        print(f'Fitness result: {fitness}', file=f)
        print('  ', file=f)

def run_training(reward_options, total_steps, eval_num_episodes, results_file, 
                 version=False):
    
    wm = Wingman(config_yaml="./src/settings.yaml")
    cfg = wm.cfg
    # wandb.config.update({"allow_val_change":True})
    
    if version:
        print(version)
        cfg.version_number = version
    cfg.reward_options = reward_options
    cfg.total_steps = total_steps
    cfg.eval_num_episodes = 50
    src.main.train(wm)

    cfg.eval_num_episodes = eval_num_episodes    
    cfg.eval_fitness = True
    fitness = src.main.eval_fitness(wm)
    write_result(cfg.version_number, cfg.reward_options, fitness, results_file)
    
start = time()
# Saving progress
results_file = f'results_{datetime.date.today()}.txt'
if not os.path.isfile(target_dir+'\\weights\\'+results_file):
    create_log(results_file)
else:
    print('     Log file exists')
    print('=========================')
    print('')

total_steps = 300000
eval_num_episodes = 200

reward_options0 = "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 2, 100, 1, 3, 20, 0, 0, 500" # original
run_training(reward_options0, total_steps, eval_num_episodes, results_file)

reward_options1 = "1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 50, 250, 5000" # best PHI #1
run_training(reward_options1, total_steps, eval_num_episodes, results_file)

reward_options2 = "1, 0, 0, 0, 0, 0, 10, 2, 0.35, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 100, 50, 250, 5000" # best PHI #2
run_training(reward_options2, total_steps, eval_num_episodes, results_file)

reward_options3 = "5, 5, 1, 2, 2, 2, 5, 1, 0.2, 0.1, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 50, 10, 100, 5000" # whatever
run_training(reward_options3, total_steps, eval_num_episodes, results_file)

reward_options4 = "5, 1, 1, 1, 1, 2, 10, 2, 0.35, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 20, 10, 50, 500" # whatever
run_training(reward_options4, total_steps, eval_num_episodes, results_file)

end = time()

with open(f'weights/{results_file}', 'a') as f:
    print(f'Results obtained in {(end-start):.0f}s ({((end-start)/3600):.2f}h) \
with {total_steps/1000:.0f}k steps and {eval_num_episodes} evaluation episodes'
    , file=f)
    print('', file=f)

#%% Generating data
import wandb

# # baseline
reward_gains = "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 2, 100, 1, 3, 20, 0, 0, 500"
version_number = 20231607
wandb_notes = "baseline_20231607"

# PHI0
reward_gains = "0, 1, 1, 10, 0.5, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 500"
version_number = 20232107
wandb_notes = "PHI0_20232107"


total_runs = 10

# train
for i in range(total_runs):
    print(f'      Run number {i}')
    print('=============================')
    print('')
    runfile('src/main.py', args=f'--train --reward_options="{reward_gains}" \
            --version_number={version_number}{i} --wandb --wandb_notes={wandb_notes}')
    wandb.finish(quiet=True)

#%% Runnig all R_0 and PHI_1
import os
import wandb

target_dir = 'C:\\Users\\LeandroLandgraf\\Documents\\10_TechnicalMastersProgramme\\5-6_Semester\\LandgrafL\\Git\\rocket_landing_sac'
os.chdir(target_dir)

total_runs = 3
buffer_size = 50000
total_steps = 300000
eval_num_episodes = 50

# R_0: original
reward_gains = ["0, 5, 2, 0, 100, 0, 1, 0, 3, 0, 0, 0, 20, 0, 0, 500",
                "0, 5, 1, 0, 100, 0, 2, 0, 5, 0, 0, 0, 50, 0, -50, 1000"]
version_number = [20230701, 20230702]
wandb_notes = ["R0_20230701", "R0_20230702"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

# PHI_1: potential-based difference of potentials function PHI_0=-||(s)||
reward_gains = ["1, 3, 100, 0, 1, 0, 3, 0, 1, 0, 0, 0, 20, 0, 0, 500",
                "1, 5, 5, 0, 2, 0, 5, 0, 3, 0, 0, 0, 50, 0, -50, 1000",
                "1, 5, 10, 0, 5, 0, 10, 0, 5, 0, 0, 0, 50, 0, -50, 1000",
                "1, 5, 25, 0, 0, 0, 10, 0, 0, 0, 25, 0, 0, 0, 10, 0, 50, 0, 0, 0, -50, 1000"]
version_number = [20230711, 20230712, 20230713, 20230714]
wandb_notes = ["PHI1_20230711", "PHI1_20230712", "PHI1_20230713", "PHI1_20230714"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

# PHI_1: potential-based difference of potentials function PHI_0=-||(s)||
reward_gains = ["1, 5, 0.05, 0, 0.2, 0, 6.667, 0, 0.1, 0, 0, 0, 50, 0, -50, 1000", # alpha_2 normalized
                "1, 5, 7.5, 0, 3.5, 0, 7.5, 0, 4, 0, 0, 0, 50, 0, -50, 1000", # average alpha_1 and alpha_2
                "1, 50, 100, 0, 50, 0, 100, 0, 50, 0, 0, 0, 50, 0, -50, 1000", # 10 * alpha_2
                "1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 50, 0, -50, 1000"]
version_number = [20230811, 20230812, 20230813, 20230814]
wandb_notes = ["PHI1_20230811", "PHI1_20230812", "PHI1_20230813", "PHI1_20230814"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

# normalisation reward vector: 1, 1, 200, 25, 1.5, 50, 1, 1, 1, 1, 1

# PHI_1: potential-based difference of potentials function PHI_0=-||(s)||
reward_gains = ["1, 5, 0.05, 0, 5, 0, 7.1, 0, 3, 0, 0, 0, 50, 0, -50, 1000", # best alpha(i): 4, 2, 4/5, 2/7
                "1, 5, 0.01, 0, 4.25, 0, 6.8, 0, 5, 0, 0, 0, 50, 0, -50, 1000"] # push the best
version_number = [20230815, 20230816]
wandb_notes = ["PHI1_20230815", "PHI1_20230816"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)


#%% Runnig all PHI_2
import os
import wandb

target_dir = 'C:\\Users\\LeandroLandgraf\\Documents\\10_TechnicalMastersProgramme\\5-6_Semester\\LandgrafL\\Git\\rocket_landing_sac'
os.chdir(target_dir)

total_runs = 3
buffer_size = 50000
total_steps = 300000
eval_num_episodes = 50

# PHI_2: potential-based harmonic function PHI=1/||(s)||
reward_gains = ["2, 3, 1000, 0, 300, 0, 10, 0, 500, 0, 0, 0, 20, 0, 0, 500",
                "2, 5, 500, 0, 150, 0, 5, 0, 250, 0, 0, 0, 50, 0, -50, 1000",
                "2, 5, 750, 0, 500, 0, 5, 0, 750, 0, 0, 0, 50, 0, -50, 1000",
                "2, 5, 300, 0, 600, 0, 3, 0, 1000, 0, 0, 0, 50, 0, -50, 1000"]
version_number = [20230721, 20230722, 20230723, 20230724]
wandb_notes = ["PHI2_20230721", "PHI2_20230722", "PHI2_20230723", "PHI2_20230724"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

# PHI_2: potential-based harmonic function PHI=1/||(s)||
reward_gains = ["2, 5, 2.5, 0, 20, 0, 3.333, 0, 15, 0, 0, 0, 50, 0, -50, 1000", # alpha_1/2/1/2 normalized
                "2, 5, 625, 0, 325, 0, 5, 0, 500, 0, 0, 0, 50, 0, -50, 1000", # average alpha_1/2/1/2
                "2, 50, 5000, 0, 50, 0, 7500, 0, 50, 0, 0, 0, 50, 0, -50, 1000", # 10*alpha_1/2/1/2
                "2, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 50, 0, -50, 1000"]
version_number = [20230821, 20230822, 20230823, 20230824]
wandb_notes = ["PHI2_20230821", "PHI2_20230822", "PHI2_20230823", "PHI2_20230824"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

# normalisation reward vector: 1, 1, 200, 25, 1.5, 50, 1, 1, 1, 1, 1

# PHI_2: potential-based harmonic function PHI=1/||(s)||
reward_gains = ["2, 5, 1, 0, 750, 0, 20, 0, 750, 0, 0, 0, 50, 0, -50, 1000", # best alpha(i): 7, 2, 1, 2
                "2, 5, 0.005, 0, 0.04, 0, 0.667, 0, 0.02, 0, 0, 0, 50, 0, -50, 1000"] # 1/x normalized
version_number = [20230825, 20230826]
wandb_notes = ["PHI2_20230825", "PHI2_20230826"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

#%% Runnig all PHI_3
import os
import wandb

target_dir = 'C:\\Users\\LeandroLandgraf\\Documents\\10_TechnicalMastersProgramme\\5-6_Semester\\LandgrafL\\Git\\rocket_landing_sac'
os.chdir(target_dir)

total_runs = 3
buffer_size = 50000
total_steps = 300000
eval_num_episodes = 50

# # PHI_3: potential-based log10 function PHI_2=-log(s+1)
reward_gains = ["3, 3, 2, 0, 1, 0, 5, 0, 2, 0, 0, 0, 20, 0, 0, 500",
                "3, 5, 3, 0, 3, 0, 2, 0, 3, 0, 0, 0, 50, 0, -50, 1000",
                "3, 5, 10, 0, 5, 0, 10, 0, 10, 0, 0, 0, 50, 0, -50, 1000",
                "3, 5, 3, 0, 10, 0, 5, 0, 5, 0, 0, 0, 50, 0, -50, 1000"]
version_number = [20230731, 20230732, 20230733, 20230734]
wandb_notes = ["PHI3_20230731", "PHI3_20230732", "PHI3_20230733", "PHI3_20230734"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

# PHI_3: potential-based log10 function PHI_2=-log(s+1)
reward_gains = ["3, 5, 0.15, 0, 0.04, 0, 3.333, 0, 0.5, 0, 0, 0, 50, 0, -50, 1000", # alpha_1/0/0/?? normalized
                "3, 5, 2.5, 0, 2, 0, 3.5, 0, 0.75, 0, 0, 0, 50, 0, -50, 1000", # average alpha_1/0
                "3, 50, 1.5, 0, 0.4, 0, 33.33, 0, 5, 0, 0, 0, 50, 0, -50, 1000", # 10*alpha_1/0/0/??
                "3, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 50, 0, -50, 1000"]
version_number = [20230831, 20230832, 20230833, 20230834]
wandb_notes = ["PHI3_20230831", "PHI3_20230832", "PHI3_20230833", "PHI3_20230834"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

# normalisation reward vector: 1, 1, 200, 25, 1.5, 50, 1, 1, 1, 1, 1

# PHI_3: potential-based log10 function PHI_2=-log(s+1)
reward_gains = ["3, 5, 1, 0, 10, 0, 5, 0, 50, 0, 0, 0, 50, 0, -50, 1000", # best alpha(i): 7, 3, 3, ??
                "3, 5, 100, 0, 1000, 0, 500, 0, 5000, 0, 0, 0, 50, 0, -50, 1000"] # *100
version_number = [20230835, 20230836]
wandb_notes = ["PHI3_20230835", "PHI3_20230836"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

#%% Runnig all PHI_4
import os
import wandb

target_dir = 'C:\\Users\\LeandroLandgraf\\Documents\\10_TechnicalMastersProgramme\\5-6_Semester\\LandgrafL\\Git\\rocket_landing_sac'
os.chdir(target_dir)

total_runs = 3
buffer_size = 50000
total_steps = 300000
eval_num_episodes = 50

# PHI_4: potential-based log2 function PHI_4=-log(s+1)
reward_gains = ["4, 3, 2, 0, 1, 0, 5, 0, 2, 0, 0, 0, 20, 0, 0, 500",
                "4, 5, 3, 0, 3, 0, 2, 0, 3, 0, 0, 0, 50, 0, -50, 1000",
                "4, 5, 10, 0, 5, 0, 10, 0, 10, 0, 0, 0, 50, 0, -50, 1000",
                "4, 5, 3, 0, 10, 0, 5, 0, 5, 0, 0, 0, 50, 0, -50, 1000"]
version_number = [20230741, 20230742, 20230743, 20230744]
wandb_notes = ["PHI4_20230741", "PHI4_20230742", "PHI4_20230743", "PHI4_20230744"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

# PHI_4: potential-based log2 function PHI_4=-log(s+1)
reward_gains = ["4, 5, 0.05, 0, 0.4, 0, 3.333, 0, 0.5, 0, 0, 0, 50, 0, -50, 1000", # alpha_2/3/3/?? normalized
                "4, 5, 6.5, 0, 7.5, 0, 7.5, 0, 0.05, 0, 0, 0, 50, 0, -50, 1000", # average alpha_2/3
                "4, 50, 0.5, 0, 4, 0, 33.33, 0, 5, 0, 0, 0, 50, 0, -50, 1000", # 10*alpha_2/3/3/?? normalized
                "4, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 50, 0, -50, 1000"]
version_number = [20230841, 20230842, 20230843, 20230844]
wandb_notes = ["PHI4_20230841", "PHI4_20230842", "PHI4_20230843", "PHI4_20230844"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

# normalisation reward vector: 1, 1, 200, 25, 1.5, 50, 1, 1, 1, 1, 1

# PHI_4: potential-based log2 function PHI_4=-log(s+1)
reward_gains = ["4, 5, 1, 0, 10, 0, 7.5, 0, 0.05, 0, 0, 0, 50, 0, -50, 1000", # best alpha(i): 7, 3, 5, 5
                "4, 5, 1, 0, 5, 0, 8.75, 0, 0.025, 0, 0, 0, 50, 0, -50, 1000"] # pushing it
version_number = [20230845, 20230846]
wandb_notes = ["PHI4_20230845", "PHI4_20230846"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

#%% #Mixed PHI(s)
import os
import wandb

target_dir = 'C:\\Users\\LeandroLandgraf\\Documents\\10_TechnicalMastersProgramme\\5-6_Semester\\LandgrafL\\Git\\rocket_landing_sac'
os.chdir(target_dir)

total_runs = 3
buffer_size = 50000
total_steps = 300000
eval_num_episodes = 50

# Mix_1: best for each variable
reward_gains = ["5, 5, 1, 100, 750, 0, 7.1, 0, 750, 0, 0, 0, 50, 0, -50, 1000",
                "5, 5, 0.5, 50, 375, 0, 3.55, 0, 375, 0, 0, 0, 50, 0, -50, 1000",
                "5, 5, 2, 200, 1500, 0, 14.2, 0, 1500, 0, 0, 0, 50, 0, -50, 1000"]
version_number = [20230851, 20230852, 20230853]
wandb_notes = ["Mix1_20230851", "Mix1_20230852", "Mix1_20230853"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

# Mix_2: second best for each variable
reward_gains = ["6, 5, 3, 0, 10, 0, 7.5, 0, 1, 0, 0, 0, 50, 0, -50, 1000",
                "6, 5, 1.5, 0, 5, 0, 3.75, 0, 0.5, 0, 0, 0, 50, 0, -50, 1000",
                "6, 5, 6, 0, 20, 0, 15, 0, 2, 0, 0, 0, 50, 0, -50, 1000"]
version_number = [20230861, 20230862, 20230863]
wandb_notes = ["Mix2_20230861", "Mix2_20230862", "Mix2_20230863"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

# Mix_3: both Mix_1 and Mix_2
reward_gains = ["7, 5, 1, 3, 750, 10, 7.1, 7.5, 750, 1, 0, 0, 50, 0, -50, 1000",
                "7, 5, 0.5, 1.5, 375, 5, 3.55, 3.75, 375, 0.5, 0, 0, 50, 0, -50, 1000",
                "7, 5, 2, 6, 1500, 20, 14.2, 15, 1500, 2, 0, 0, 50, 0, -50, 1000"]
version_number = [20230871, 20230872, 20230873]
wandb_notes = ["Mix3_20230871", "Mix3_20230872", "Mix3_20230873"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

# Mix_4: the best of each Mix?
reward_gains = ["8, 5, 1, 100, 10, 0, 15, 0, 1500, 0, 0, 0, 50, 0, -50, 1000",
                "8, 5, 0.5, 50, 5, 0, 7.5, 0, 750, 0, 0, 0, 50, 0, -50, 1000",
                "8, 5, 2, 200, 20, 0, 30, 0, 3000, 0, 0, 0, 50, 0, -50, 1000"]
version_number = [20230881, 20230882, 20230883]
wandb_notes = ["Mix4_20230881", "Mix4_20230882", "Mix4_20230883"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

# Mix_5: Phi_2 alpha_0 and alpha_8 + Phi_4 alpha_3 and alpha_5
reward_gains = ["9, 5, 1000, 3, 300, 10, 10, 5, 500, 5, 0, 0, 50, 0, -50, 1000",
                "9, 5, 1, 3, 750, 10, 20, 5, 750, 5, 0, 0, 50, 0, -50, 1000",
                "9, 5, 1000, 6.5, 300, 7.5, 10, 7.5, 500, 0.05, 0, 0, 50, 0, -50, 1000",
                "9, 5, 1, 6.5, 750, 7.5, 20, 7.5, 750, 0.05, 0, 0, 50, 0, -50, 1000"]
version_number = [20230891, 20230892, 20230893, 20230894]
wandb_notes = ["Mix5_20230891", "Mix5_20230892", "Mix5_20230893", "Mix5_20230894"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

#%% Run the best Phi and Mix for 1M episodes
import os
import wandb

target_dir = 'C:\\Users\\LeandroLandgraf\\Documents\\10_TechnicalMastersProgramme\\5-6_Semester\\LandgrafL\\Git\\rocket_landing_sac'
os.chdir(target_dir)
total_runs = 3
buffer_size = 150000
total_steps = 1000000
eval_num_episodes = 150

# Mix 4 alpha 1 for 1M episodes
reward_gains = ["8, 5, 0.5, 50, 5, 0, 7.5, 0, 750, 0, 0, 0, 50, 0, -50, 1000"]
version_number = [20230982]
wandb_notes = ["Mix4_1M_20230982"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

# Phi 4 alpha 3 for 1M episodes
reward_gains = ["4, 5, 3, 0, 10, 0, 5, 0, 5, 0, 0, 0, 50, 0, -50, 1000"]
version_number = [20230944]
wandb_notes = ["Phi4_1M_20230944"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

#%% Two last functions, graph-based
import os
import wandb

target_dir = 'C:\\Users\\LeandroLandgraf\\Documents\\10_TechnicalMastersProgramme\\5-6_Semester\\LandgrafL\\Git\\rocket_landing_sac'
os.chdir(target_dir)
total_runs = 3
buffer_size = 50000
total_steps = 300000
eval_num_episodes = 50

# normalisation reward vector: 200, 25, 1.5, 50
# normalised factors: 0.005, 0.04, 0.667, 0.02
# sucess: dpad < 4, ang_vel < 0.35; ang_pos < 0.2; lin_vel < 5.56
# factors: a = 1, b = 1; a = success, b = 100*norm; x2, /2 

# y1 = a/(x+0.01) - b * x
reward_gains = ["10, 5, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 50, 0, -50, 1000",
                "10, 5, 2, 0.5, 0.35, 4, 0.2, 66.667, 5.56, 2, 0, 0, 50, 0, -50, 1000",
                "10, 5, 4, 1, 0.7, 8, 0.4, 133.334, 11.12, 4, 0, 0, 50, 0, -50, 1000",
                "10, 5, 1, 0.25, 0.175, 2, 0.1, 33.334, 2.78, 1, 0, 0, 50, 0, -50, 1000"]
version_number = [202309101, 202309102, 202309103, 202309104]
wandb_notes = ["Y1_202309101", "Y1_202309102", "Y1_202309103", "Y1_202309104"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

# y2 = a/(x+0.01) - b * log2(x+1)
reward_gains = ["11, 5, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 50, 0, -50, 1000",
                "11, 5, 2, 0.5, 0.35, 4, 0.2, 66.667, 5.56, 2, 0, 0, 50, 0, -50, 1000",
                "11, 5, 4, 1, 0.7, 8, 0.4, 133.334, 11.12, 4, 0, 0, 50, 0, -50, 1000",
                "11, 5, 1, 0.25, 0.175, 2, 0.1, 33.334, 2.78, 1, 0, 0, 50, 0, -50, 1000"]
version_number = [202309111, 202309112, 202309113, 202309114]
wandb_notes = ["Y2_202309111", "Y2_202309112", "Y2_202309113", "Y2_202309114"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

#-------------------------------------------------------------------------------
# I am stubborn and don't give up, running two more ideas before I am done
#-------------------------------------------------------------------------------

# y3,dpad from Mix4 and y2 for the rest, gains are Mix 4 alpha 1 and Y2 alpha 1
reward_gains = ["12, 5, 0.5, 50, 1, 1, 1, 1, 1, 1, 0, 0, 50, 0, -50, 1000",
                "12, 5, 0.25, 25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 50, 0, -50, 1000",
                "12, 5, 1, 100, 2, 2, 2, 2, 2, 2, 0, 0, 50, 0, -50, 1000",
                "12, 5, 0.5, 50, 1, 1, 1, 1, 0.1, 0.1, 0, 0, 50, 0, -50, 1000"]
version_number = [202309121, 202309122, 202309123, 202309124]
wandb_notes = ["Y3_202309121", "Y3_202309122", "Y3_202309123", "Y3_202309124"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

# y3,dpad from Mix4 and y2 for the rest, gains are Mix 4 alpha 1 and Y2 alpha 1
reward_gains = ["12, 5, 0.5, 50, 2, 2, 2, 2, 0.1, 0.1, 0, 0, 50, 0, -50, 1000",
                "12, 5, 0.5, 50, 2, 4, 2, 4, 0.1, 0.2, 0, 0, 100, 0, -50, 10000"]
version_number = [202309125, 202309126]
wandb_notes = ["Y3_202309125", "Y3_202309126"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

# y3,dpad from Mix4 and y2 for the rest, gains are Mix 4 alpha 1 and Y2 alpha 1
reward_gains = ["12, 5, 0.25, 25, 0.25, 0.25, 0.25, 0.25, 0.1, 0.1, 0, 0, 50, 0, -50, 1000",
                "12, 5, 0.25, 25, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 50, 0, -50, 1000"]
version_number = [202309127, 202309128]
wandb_notes = ["Y3_202309127", "Y3_202309128"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)

#%% Running a few more for 1M episodes
import os
import wandb

target_dir = 'C:\\Users\\LeandroLandgraf\\Documents\\10_TechnicalMastersProgramme\\5-6_Semester\\LandgrafL\\Git\\rocket_landing_sac'
os.chdir(target_dir)
total_runs = 3
buffer_size = 150000
total_steps = 1000000
eval_num_episodes = 150

reward_gains = ["11, 5, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 50, 0, -50, 1000"]
version_number = [202310111]
wandb_notes = ["Y2_1M_202309111"]

for i, alfa in enumerate(reward_gains): 
    for j in range(total_runs):
        runfile('src/main.py', args=f'--train --reward_options="{alfa}" \
                --buffer_size={buffer_size} --total_steps={total_steps} \
                --eval_num_episodes={eval_num_episodes} \
                --version_number={version_number[i]}{j} --wandb --wandb_notes={wandb_notes[i]}')
        wandb.finish(quiet=True)



#%% Running training command prompt style for testing

reward_gains = "11, 5, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 50, 0, -50, 1000"
eval_num_episodes = 50
total_steps = 100000
v = 2023091110


# # train
runfile('src/main.py', args=f'--train --reward_options="{reward_gains}" \
        --total_steps={total_steps} --eval_num_episodes={eval_num_episodes}')

# # eval_fitness
# runfile('src/main.py', args=f'--eval_fitness --version={v} --reward_options="{reward_gains}" \
#         --eval_num_episodes={eval_num_episodes}')

# display
runfile('src/main.py', args=f'--display --version={v} --reward_options="{reward_gains}"')

# # gif
# runfile('src/main.py', args=f'--display --version={v} --reward_options="{reward_gains}" --render_gif')

#%%

print(f'Results obtained in {(end-start):.0f}s ({((end-start)/3600):.2f}h) \
with {total_steps/1000:.0f}k steps and {eval_num_episodes} evaluation episodes')

a = ["1,2,3", "2,3,4", "3,4,5"]
b=[1,2,3,4]

for i, item in enumerate(a):
    print(item)
    print(b[i])
    
    
    