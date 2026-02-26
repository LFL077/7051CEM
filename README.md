---
title: Readme
---

Repository for files and resources related to the final dissertation on Data Science and Computational Intelligence of the Faculty of Engineering, Environment and Computing.

<br>

# Title (provisional): "Development and evaluation of Reinforcement Learning strategies for landing a simulated 1:10th scale SpaceX Falcon 9 using the `PyFlyt Rocket-Landing-v0` environment."

<br>

## Project Summary

Using Reinforcement Learning algorithms and the `PyFlyt/Rocket-Landing-v0` Gymnasium Environment, land a rocket falling at terminal velocity on a landing pad, with only 1% of fuel remaining.

## What are the aims and objectives of the project?

This project is aimed at solving the `PyFlyt/Rocket-Landing-v0` Gymnasium Environment, with the definition of the reward functions (both dense and sparse rewards), and implementation of several Reinforcement Learning algorithms such as Proximal Policy Gradient (PPO) and Soft Actor-Critic (SAC).

The environment is part of the PyFlyt - UAV Flight Simulator Gymnasium Environments for Reinforcement Learning Research developed by Coventry University researchers, and its goal is to land a rocket falling at terminal velocity on a landing pad, with only 1% of fuel remaining.

<br>

<br>
## Planning:

|**Sprint**|**When**   |**What**                                                  |**Requisite(s)**                            |**Completion**|
|----------|-----------|----------------------------------------------------------|--------------------------------------------|--------------|
|    11    | 31.07.23  | Definition of reward functions - PBRS                    | Environment installation and commissioning | 100%         |
|    12    | 14.08.23  | Definition of reward functions - Mixed $\Phi(s)$         | Reward functions - PBRS                    | 100%         |
|    13    | 28.08.23  | Definition of reward functions - Other approach?         | Reward functions - PBRS                    | 100%         |
| ***14*** | 11.09.23  | Definition of reward functions - Other approach?         | Reward functions - PBRS                    | 25%          |
|    15    | 25.09.23  | SAC agent training and results                           | Definition of reward functions             | 50%          |
|  ~~16~~  | 09.10.23  | ~~PPO - initial implementation~~                         | Definition of reward functions             | 10%          |
|  ~~17~~  | 23.10.23  | ~~PPO - final implementation and hyperparameter def.~~   | PPO - initial implementation               | 00%          |
|  ~~18~~  | 06.11.23  | ~~PPO agent training and results~~                       | PPO - final implementation                 | 00%          |
|    19    | 20.11.23  | Project Report finishing (originally due 08.08.23)       | EVERYTHING!!                               | 00%          |
|    20    | 04.12.23  |                                                          | \--                                        | \--          |

## Old Planning:

|**Sprint**|**When**|**What**                                                  |**Requisite(s)**                            |**Completion**|
|----------|--------|----------------------------------------------------------|--------------------------------------------|--------------|
|     1    | 22.05  | Ethics application (due ~~31.05~~ 09.06)                 | \--                                        | 100%         |
|     2    | 29.05  | Project and requirements definition and planning         | Ethics application                         | 100%         |
|     3    | 05.06  | Environment installation and commissioning               | Project and requirements definition        | 100%         |
|     4    | 12.06  | Project Proposal (due 19.06)                             | Project and requirements definition        | 100%         |
|     5    | 19.06  | Definition of reward functions                           | Environment installation and commissioning | 50%          |
|     6    | 26.06  | RL method 1 (SAC)                                        | Definition of reward functions             | 75%          |
|     7    | 03.07  | RL method 2 (PPO)                                        | Definition of reward functions             | 05%          |
|     8    | 10.07  | ~~RL method 3 (DDPG)~~ Still working on reward function  | Definition of reward functions             | 00%          |
|     9    | 17.07  | Project Presentation (due 24.07)                         | All RL methods and results                 | 100%         |
|    10    | 24.07  |                                                          | \--                                        | \--          |
|  ~~11~~  |~~31.07~~| ~~Project Report (due 08.08)~~                          | ~~EVERYTHING!!~~                           | ~~25%~~      |

## Supervisor meetings:

### 30.05.2023 (Prof. James Brusey)

- CW1 (Project Proposal) assignment brief shared

- How to do a master's project: https://github.coventry.ac.uk/aa3172/masters-projects

- Suggestions:
    - Request access to high performance computers, instead of using Google Colab
    - Try using Zotero with Better BibTex for references management
    - Recommended max. 8000 words
    - Use recommended settings from scrbook, set fontsize to either 10 or 11pt and bigger margins
        - The elements of typographic style - reference on how to design a page layout
        - ClassicThesis - A "classically styled" thesis package: https://www.ctan.org/pkg/classicthesis
	 - Use IMRaD for Chapter structure: Intro and Background / Method / Results / Conclusions
	     - Introduction
	     - Methods
	     - Results
	     - Discussion

- ToDo
    - Contact Jun Jet Tai and use his expertise for the technical topics (e-mail sent on 30.05, meeting happened 01.06)
    - Rewrite planning using AGILE features (which include reading and writing on each sprint)
    

### 01.06.2023 (Jun Jet Tai)

- Fork the environment (the place to change the code will probably be the environment itself, and we will have to work on the reward function, done on 01.06):
    - https://github.com/jjshoots/PyFlyt/blob/master/PyFlyt/gym_envs/rocket_envs/rocket_landing_env.py (Lines 206-214)
    - https://github.com/jjshoots/rocket_landing_sac
- Challenge: not to find a RL method which works, but making sure the environment is solvable
- Implementation uses Weights & Biases (https://docs.wandb.ai/quickstart)
    - https://github.com/jjshoots/Wingman

- ToDo
  - Look for references on how to design reward functions
  - Install all libraries in Python environment (done 04.06, it's alive and rocket_landing_sac works!)


### 03.07.2023 (Prof. James Brusey)
- Discussion about current status and difficulties in defining successfully the reward function
  - Suggestion about reward shaping: review and fully understand basic concept of reward shaping and avoid providing hints in a way that changes what is optimal
  - Provide function based on state, using ***potential based*** shaping function:
    - *(Ng, A. Y., Harada, D., & Russell, S. (1999, June). Policy invariance under reward transformations: Theory and application to reward shaping)*
    - $F(s, a, s') = \gamma \Phi(s') - \Phi(s)$
  - Thinking not in terms of individual reward in a given point of time, but the integrated reward at the end of the episode
  - Perhaps not having a positive reward when successful, but a very big negative reward when crashing
  - Start from closer to the end (less steps, scale down the problem) to validate approach, do things that are smaller first to ensure it works

Next: put together any questions, queries, issues as e-mail for next week


### 12.07.2023 (Jun Jet Tai)
- Discussion about potential-based reward shaping and implementation in the code
- Proposed simplification of task: 
  - Disable randomized and accelerated drop:
    - `options = dict(randomize_drop=True, accelerate_drop=True)`
  - Spawn landing platform in the same place, by setting angle and distance to zero:
    - `theta = self.np_random.uniform(0.0, 2.0 * np.pi)`
    - `distance = self.np_random.uniform(0.0, 0.05 * self.ceiling)`
  - Initial altitude already decreased to 300m
- Jet suggested logging everything to identify what the agent is exploring/learning and where to focus changes
- Observed behavior may be due to negative reqrad when landing outside platform or chrashing being too big
- Observe entropy for indication of exploration vs exploitation




## Master ToDo list
- ~~Annoyingly pybullet crashes sometimes, and I have to reset the environment/kernel.~~
  - Updating numpy seems to work
  - Alternatively:
    - `os.environ["KMP_DUPLICATE_LIB_OK"]="True"`
- ~~Making reward_options passable at start of learning, optimisation algorithm calling RL.~~
- Create a rocket_landing_ppo
- Create a rocket_landing_ddpg

## Running log

### rocket_landing_sac

- Version366877: first run, no changes to reward function, testing the environment:
  - Result: 
    - All libraries and dependencies working correctly.
      - `wingman: Step 1270253; Average Loss 2961.60621; Lowest Average Loss 3205.45730`
      - `wingman: New lowest point, saving weights to: weights\Version366877\weights0.pth`

- Version36779: new reward function, also testing passing reward parameters
  - Result:
    - After some debugging and changing `rocket_landing_sac`, it runs as expected
      - `wingman: Step 500549; Average Loss 274756.00464; Lowest Average Loss 266906.59604`

```
reward_options = {0:5, 1:1.0, 2:1.0, 3:5.0, 4:2.0, 5:20.0, 6:500.0}

self.reward += (
  - self.reward_options[0] # negative offset to discourage staying in the air
  - (self.reward_options[1] * angular_velocity)  # not spinning
  - (self.reward_options[2] * angular_position)  # and upright
  - (self.reward_options[3] * linear_velocity)   # basically stopped
  + (self.reward_options[4] / (distance_to_pad + 0.001))   # we want to be at the pad
  )
```

- Version45483: running with original reward function, but passing arguments this time, and reduced to 500k steps.
  - Result:
    - `wingman: Step 470253; Average Loss 4603.52836; Lowest Average Loss 4948.31568`
    - `wingman: Step 500191; Average Loss 4840.08224; Lowest Average Loss 4603.52836`
    
```
reward_gains = "5.0, 2.0, 100.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 20.0, 500.0"

self.reward += (
    - self.reward_options[0] # negative offset to discourage staying in the air
    + (self.reward_options[1] / offset_to_pad)  # encourage being near the pad
    + (self.reward_options[2] * progress_to_pad)  # encourage progress to landing pad
    - (self.reward_options[3] * abs(self.ang_vel[-1]))  # minimize spinning
    - (self.reward_options[4] * np.linalg.norm(self.ang_pos[:2]))  # penalize aggressive angles
    - (self.reward_options[5] * angular_velocity)  # not spinning
    - (self.reward_options[6] * angular_position)  # and upright
    - (self.reward_options[7] * linear_velocity)   # basically stopped
    + (self.reward_options[8] / (distance_to_pad + 0.001))   # we want to be at the pad
    )

self.landing_pad_contact = 1.0
self.reward += self.reward_options[9]

self.reward += self.reward_options[10] # completion bonus
self.info["env_complete"] = True
self.termination |= True
```

- Version280324 and Version595401: testing back-to-back training and new fitness_eval function
  - Sparse fitness (focused solely on end-goal):
    - `self.fitness += 1 # improves fitness if touching pad`
    - `self.fitness += 10 # greatly improve fitness if successful landing`
  - Results:
    - `280324 fitness results: 2.0 for 280324`
    - `595401 fitness results: 1.0 for 595401`
    - `2 x 200000 steps done in 1.35e+04s with 2 x 100 evaluation`

- Version167302: first run using bayes_opt (22.06.2023) including:
  - reward_priors = [5.0, 2.0, 100.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 20.0, 500.0]
  - total_steps: 300000; eval_num_episodes: 100
    - `wingman: Step 300103; Average Loss 3279.41800; Lowest Average Loss 3374.53150`

- 2023-06-22: several runs, using 2 priors + 3 inits + 15 iterations
  - no better result than before after 60h of running

- 25.06.2023: reward function including ALL observation space
  - Running for 200k steps to initialize optimizer
  - Running with 300k steps during iterations, using best results from initialization as priors 
  - Unsuccessful
  
- 28.06.2023: reward function including ALL observation space plus original terms
  - Including priors based on best results so far and original reward function
  - Running for 500k steps on priors and init
  - Running only 300k on iterations, since it was taking too long to run
  - Results:
    - `Version number: 634396 prior0 (original function, 500k steps)`
   ` Reward function vector: 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 5.000, 2.000, 100.000, 1.000, 1.000`
    `Fitness result: 57.0`
    - `Version number: 475780 iter4 (300k steps)`
    `Reward function vector: 0.950, 0.390, 0.821, 3.704, 1.677, 1.170, 4.602, 0.861, 2.962, 41.186, 58.828, 48.565, 0.276, 4.017, 4.616, 0.732, 218.262, 4.479, 1.492`
    `Fitness result: 32.0`

- 

- 

- 

- 

.
