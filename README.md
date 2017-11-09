# Model Free Based Reinforcement Learning
This is where I know the most, but it is still hard to keep up with Arxiv (in)sanity.



Policy Gradient Methods

Key Algorithms
A3C
TRPO
PPO
ACER
ACKTR
DDPG
DDDDPG (ICLR)


Value Learning / Q - Learning

Key Algorithms
DQN
Double DQN
Dueling DQN
Distrubtional DQN (new updated paper a few days ago)
Noisy Nets
Prioritized Experience Replay
Rainbow (combines all of them)


Other ideas

UNREAL
Nueral Episodic Control


# Imitation Learning / Behavorial Cloning
Simple approach to solving reinforcement learning problems. Just frame the problem completely as a supervised learning problem to predict the correct action given a dataset of human experience.

Can be surprisingly succesful given how simple it is to use in practice. For example in self driving cars (see Nvidia example), they simple train a deep ConvNet with outputs being the actuators of the car. The disadvantages of this are of course you need very large datasets of human experience which can be hard to collect.

One of the major practical issues is that of compounding errors. When the classifier makes a mistake and begins to see a trajectory and/or states that is hasn't seen in training, all bets are off whether it will work correctly. We hope (as is usual in deep networks), that the network can generalize over the input manifold to areas that is has never seen before, but we know in practice this is rarely the case.

Dataset Aggregation Algorithm (DAgger) is a simple online learning algorithm that looks to iteratively combat the compounding errors issue. You train on the dataset. Then run your policy at test time and record your observations. Then have a human label the observations with the correct state. In one sense, this is supposed to teach your algorithm how to correct itself when it makes errors.

Learning References:
* [Berkeley Deep RL Course Lecture 2](https://www.youtube.com/watch?v=kl_G95uKTHw&list=PLkFD6_40KJIwTmSbCv9OVJB3YaO4sFwkX&index=2) discusses imitation learning, DAgger, and other techniques. Has links to papers and case studies

Key Papers:
* [End to End Learning for Self Driving Cars (Nvidia 2016)](https://arxiv.org/abs/1604.07316) Nvidia trains a self driving car using an end to end convolutional neural network. It is trained directly from human driving footage. They also do neat tricks with using three cameras that view the road from different angles (see berkeley course above for explanation)
* [Deep Imitation Learning for Complex Manipulation Tasks from Virtual Reality Teleoperation](https://arxiv.org/abs/1710.04615) Berkeley crew uses Virtual Reality Gear for Imitation Learning. They are now scaling this at their startup.

# Model Based Reinforcement Learning
At a high level, you use a model of the enviroment (continuous or discrete) to do some sort of planning or simulation in order to generate your policy. 

This model can be given in the simple case (you know the MDP, or maybe you know physics, or you have a simulator or something) or in practice we more generally try to learn the model.

Learning the model can be done with simple techniques such as linearizing the system and using finite differences, or more recently treating the model dynamics as a supervised learning problem and training a deep network.

## Control Systems Based Approaches (Continuous Domains)
If the underlying physical dynamics of a continuous system are known, then we can use control theory to solve these types of problems. 

There is an algorithm known as LQR which can do control. It can output the necessary sequence of actions to do something like push a puck to a target. 

Model Predictive Control (MPC) is an algorithm which repeatedly runs LQR at each timestep to adjust for pertubations and errors.

__TODO__: Finish writing this section after watching rest of Berkeley Lectures

## Planning Based Approaches (Discrete Domains)
MCTS
__TODO__: Finish writing this section after watching rest of Berkeley Lectures

## Model Based & Model Free Combinations
DYNA Algorithm

__TODO__: This is mentioned in David Silver's lectures

Learning References:
* [Berkeley Deep RL Course Lecture 3-6](https://www.youtube.com/watch?v=mZtlW_xtarI&index=3&list=PLkFD6_40KJIwTmSbCv9OVJB3YaO4sFwkX) discusses model based reinforcement learning in depth.
* [Lecture 9 of Berkeley Reinforcement Learning Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures) discusses LQR, MPC, and guided policy search.
* [Lecture 8 of David Silver's Reinforcement Learning Course](https://www.youtube.com/watch?v=ItMutbeOHtc&index=8&list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-) discusses learning models, DYNA, and MCTS

Key Papers:




## Multi-Agent Domains

__TODO__: I don't know anything about this


Key Papers:
* [Learning to Learn with Opponent Learning Awareness (LOLA)](https://arxiv.org/abs/1709.04326)
* [Multi-Agent Actor Critic](https://arxiv.org/abs/1706.02275)

## Derivative Free Optimization
A broad class of algorithms that consists of things like Genetic Algorithms, Evolutionary Algorithms, Evolutionary Search, and closely related others. Most of these algorithms are essentially random parameter search + heuristics. In the case when a derivative can be computed analytically (supervised learning, Q learning, policy gradients using reinforce trick), these algorithms are dumb because they don't move in that direction.

The only main advantage of these algorithms is that they can be scaled really well in parallel. The best paper is OpenAI's work that uses a clever trick where the seperate workers exchange what random seed they used rather than the actual parameters of the system.

Learning References:
[Lecture 8 of Berkeley Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)

Key Papers:
[Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864)



## Sim2Real Transfer
There is a lot of new work in this area from the usual suspects (OpenAI, Berkeley, Deepmind). It is a really promising approach to use a simulator to train a policy and then transfer that policy to a real robot.



