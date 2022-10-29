# -*- coding: utf-8 -*-
"""Q_learning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hj4PdBxhmk7awamIa8_l8QSK01ZV4id9

<div align=center>
		
<p></p>
<p></p>
<font size=5>
In the Name of God
<font/>
<p></p>
 <br/>
    <br/>
    <br/>
<font color=#FF7500>
Sharif University of Technology - Departmenet of Computer Engineering
</font>
<p></p>
<font color=blue>
Artifical Intelligence - Dr. Mohammad Hossein Rohban
</font>
<br/>
<br/>
Fall 2021

</div>

<hr/>
		<div align=center>
		    <font color=red size=6>
			    <br />
Practical Assignment 5-Q1
            	<br/>
			</font>
    <br/>
    <br/>
<font size=4>
			<br/><br/>
Deadline:  Bahman 17th
                <br/><b>
              Cheating is Strongly Prohibited
                </b><br/><br/>
                <font color=red>
Please run all the cells.
     </font>
</font>
                <br/>
    </div>

# Personal Data
"""

# Set your student number
student_number = 98000000
Name = ''
Last_Name = ''

"""# Rules
- You are not allowed to use provided codes that can be found on the internet. 
- If you want to use a library which is not already imported, you must ask a question on Quera to get the permission of using that.
- Do not hesitate to ask questions on Quera, if you have any.
- This assignment is due Bahman 17th 23:59:59. you can use up to 1 grace day for this assignment and the hard deadline is Bahman 18th 23:59:59.

# Q1 (30Points)

## Mountain Car

The OpenAI Gym library includes a set of Python Reinforcement Learning environments. In this question, we examine the Mountain Car environment of this collection. Mountain Car is a Reinforcement Learning task that aims to learn the policy of climbing a steep hill and reaching the flag-marked goal. Also, the car engine is not powerful enough to climb straight up the hill on the right. Therefore, it must gain enough acceleration by climbing the hill on the left.


In this case, the state of the car is determined by an array containing its position and speed.

| Num | Observation | Min | Max|
| --- | --- | --- | ---|
| 0 | Position | -1.3 | 0.5 |
| 1 | Velocity | -0.07 | 0.07|

The intelligent agent is allowed to perform three movements of right push, no push and left push in each step. The agent's move will be given to the environment and the environment returns the next state along with the movement reward. For each step that the car does not reach the target, a cost of -1 is considered. Use Q-learning to find the optimal policy in each case. To do this, you must complete the functions below.

** Note that you will receive the full score of this question only if you can achieve a score of -115 with your implementation.

<font size=4>
Author: Atoosa Chegini
			<br/>
                <font color=red>
Please run all the cells.
     </font>
</font>
                <br/>
    </div>
"""

import numpy as np
import random
import gym

env = gym.make("MountainCar-v0")
# initialization (parameters such as learning rate, gamma, number of states, number of actions, and q_table)
alpha = 0.1
gamma = 0.9
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n
num_episodes = int(50000)
discretize_size = 16
ep = 0.9
q_table = np.zeros((discretize_size**num_states+1, num_actions))
print("alpha:", alpha)
print("gamma:", gamma)
print("number of states:", num_states)
print("number of actions:", num_actions)
print("number of episodes:", num_episodes)
print("discretize size:", discretize_size)
print("epsilon:", ep)
# You may change the inputs of any function as you desire.

"""Next cell wants you supplement two functions. First for transforming the continuous space into discrete one (in order to make using q_table feasible), second for updating q_values based on the last action done by agent."""

# This is just one example of a discretization function. You can change it as you want.
def discretize_state(x, minn, step):
    return int((x - minn) / step)


def env_state_to_Q_state(state):
    [position, velocity] = state
    pos_low, pos_high = -1.3, 0.5
    vel_low, vel_high =  -0.07, 0.07
    # Complete this function!

    pos_step = (pos_high - pos_low) / discretize_size
    vel_step = (vel_high - vel_low) / discretize_size
    return (discretize_size - 1)*discretize_state(position, pos_low, pos_step) + discretize_state(velocity, vel_low, vel_step)

def update_q(current_state, next_state, action, reward):
    best_future_q = np.max(q_table[next_state])
    # Complete this function!

    q_table[current_state][action] = q_table[current_state][action] + alpha * (reward + gamma * best_future_q - q_table[current_state][action])

"""At the following cell, the ends of two functions are getting current action based on the policy and defining the training process respectively."""

# You may change the inputs of any function as you desire.
from IPython.display import clear_output


def get_action(current_state):
    return np.argmax(q_table[current_state])


def q_learning():
    
    # Complete this funciton!

    global ep
    for e in range(1, num_episodes+1):
        state, done = env.reset(), False

        while not done:
            random_action, random_prob = random.randint(0, num_actions-1), random.random()

            action = random_action if random_prob < ep else get_action(env_state_to_Q_state(state)) 
            
            next_state, reward, done, temp = env.step(action)
            q_state = env_state_to_Q_state(state)
            next_q_state = env_state_to_Q_state(next_state)
            update_q(q_state, next_q_state, action, reward)
            state = next_state
        ep *= 0.99
        clear_output(wait=True)
        print(f'Episode {e}')

    print(f'All {num_episodes} episodes are finished.')


def save_policy():
    policy  = []
    for item in q_table:
        policy.append(np.argmax(item))
    np.save('policy.npy', policy)

q_learning()
save_policy()

# Attention: don't change this function. we will use this to grade your policy which you will hand in with policy.npy
# btw you can use it to see how you are performing. Uncomment two lines which are commented to be able to see what is happening visually.
def score():
    policy, scores = np.load("policy.npy"), []
    print(policy)
    for episode in range(1000):
        print(f"******Episode {episode}")
        state, score, done, step = env_state_to_Q_state(env.reset()), 0, False, 0
        while not done:
            # time.sleep(0.04)
            action = policy[state]
            state, reward, done, _ = env.step(action)
            state = env_state_to_Q_state(state)
            step += 1
            score += int(reward)
            # env.render()
        print(f"Score:{score}")
        scores.append(score)
    print(f"Average score over 1000 run : {np.array(scores).mean()}")

score()