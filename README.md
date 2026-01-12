# Neuro-evolution-of-augmented-topologies

This project applies the NEAT (NeuroEvolution of Augmenting Topologies) algorithm to evolve neural network structures in three Gymnasium environments: CartPole, Pendulum, and Acrobot. It explores how architectures and activation functions adapt to different control tasks and compares graph-based networks with traditional FeedForward models, considering ways to simplify evolved networks into matrix-based forms.

( still in development ... )

## Table of contents

- [CartPole](#cartpole)  
- [Pendulum](#pendulum)  
- [Acrobot](#acrobot)
- [Final Thoughts](#final-thoughts)

## CartPole

The CartPole environment is a classic control problem where a pole is attached to a moving cart.  The goal is to balance the pole upright by applying forces to the cart in left or right directions.  This environment is commonly used to test reinforcement learning algorithms and control strategies.

### How it Works

- The cart moves along a frictionless track, and the pole pivots around a joint.  
- The agent can apply a fixed force either to the left (0) or right (1).  
- The episode ends if the pole angle exceeds ±12° or the cart position exceeds ±2.4 units.

###### Environment Parameters

- **Action Space:** two discrete actions → 0: left, 1: right  
- **Observation Space:** 4-dimensional → [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]  
- **Reward:** +1 for each step the pole remains balanced  
- **Starting State:** Observations initialized randomly in (-0.05, 0.05)



### Rewards

The agent receives **+1 reward** for every step the pole remains upright, including the termination step. Default reward thresholds are **500 for v1** and **200 for v0**. If `sutton_barto_reward=True`, each non-terminating step gives **0 reward**, while each terminating step gives **-1**, resulting in a threshold of **0**.





### Implementation

( ... )  







## Pendulum

## Acrobot

## Final thoughts