# Model Free Reinforcement Learning

A general class of algorithms that makes no attempt to learn the underlying dynamics of the system, nor do any planning in the future. This can cause large sample inefficiencies, but in practice these algorithms currently learn the best policies out of all classes of algorithms.

## Policy Gradient Methods

Directly optimize the policy by analytically computing the gradient using the "REINFORCE" or likelihood ratio trick. These algorithms are extremely well suited for learning continuous control tasks such as the MuJoCo simulator. They sometimes have worse sample complexity than Q-Learning algorithms as it is difficult to learn off-policy in policy gradient techniques.

__The Gradient Step__:

Your gradient step tries to make actions that led to good results more common. Andrej Karpathy provides a great analogy to supervised learning.

![Gradient Step for Policy Gradients](/images/gae.png)

The high level view of the vanilla Policy Gradient Algorithm looks like the following.

![Vanilla Policy Gradient](/images/vanillaPG.png)

__Estimating the Advantage Function__:

To get a good estimate of the advantage function, this is what is done in practice. 

Your neural network outputs V(s) and you estimate Q(s) either using n-step returns or Generalized Advantage Estimation (exponentially weighted returns just like TD(lambda)). 

![Generalized Advantage Estimation](/images/GAE.png)

![N-Step Return](/images/nstepreturn.png)

[Asynchronous Advantage Actor Critic](https://arxiv.org/abs/1602.01783) Use n-step returns and a neural network to approximate the advantage function. Use shared convolutional weights for the policy network. Train using parallel workers and get really good results.

[Generalized Advantage Estimation (2016)](https://arxiv.org/abs/1506.02438) presents a better method to approximate the advantage function using an exponentially weighted average, similar to TD(lambda) 

![The Adapted Algorithm](/images/a3c.png)

__Continuous Control__:

[Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971) Rather than outputting a probability distribution like a Guassian + Variance term, they have a deterministic policy. They then get gradient information directly from the critic instead 
of from the policy.

[Benchmarking Continuous Control](https://arxiv.org/abs/1604.06778) Provides a fantastic survey and benchmarking of different algorithms. It does not include PPO or ACKTR as it is from 2016. 

[Towards Generalization and Simplicity in Continuous Control](https://arxiv.org/abs/1703.02660) A bit of a provactative paper showing that linear and RBF policies are competitive on a lot of tasks.


__Other Ideas__:

[Sample-Efficient Actor Critic with Experience Replay](https://arxiv.org/abs/1611.01224) Describes a method to adding an experience replay and off-policy learning to actor critic algorithms. They do this through a truncated importance sampling technique that allows you to learn off policy. They also present some interesting things like using an average of past policies as an approximation to TRPO. Though I think that last point is unneccesary when you have ACTKR or PPO.

__Exploration in Policy Gradients__:

The A3C paper (above) adds an entropy term that helps to encourage exploration. In a theoretical sense this term should try to make the policy more uniform (uniform distribution is maximum entropy). I haven't seen this used very often though, and the other techniques (simply adding noise) are simpler and show good results. A mix of the two techniques with low settings on their parameters also might be a promising technique. Entropy is a "state-space" exploration technique because it encourages the network to randomly explore around.  

[Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295) also seen in OpenAI paper which has similar idea. It basically just adds random noise to your policy to help explore more. In Value-Learning e-greedy techniques do "state-space exploration", where this techniques is a "parameter space exploration" technique.


DDDDPG (ICLR)


__Natural Gradient Algorithms__:

There has been a lot of interest in algorithms that simulate a natural gradient step. The main intuition behind these optimization methods in general is that rather than taking a step in parameter space, take a step in function space. [This blog post](http://kvfrans.com/what-is-the-natural-gradient-and-where-does-it-appear-in-trust-region-policy-optimization/) provides a fantastic intuitive explanation. 

Now in supervised learning, the general trend has been to do more "dumb steps" rather than less "smart steps". Don't worry about computing the Hessian, doing L-BFGS, doing any second order information. Hell, you don't even need fresh gradients in an asynchronous setting as noise can be a great regularizer. But in Reinforcement Learning, we have a lot more issues. First off, we want to be more sample efficient. Secondly, and much more importantly, is that in Reinforcement Learning changes to our policy will change the data we see in the future. If we take a step that is too big, we can destroy everything we have learned in the worst case. That is the motivation behind the name "Trust Regions", we want to stay in the regions of the RL problem that we have explored a lot and trust our network in those areas.

There is a lot of math I need to review/learn to understand these techniques. I will leave them here for the future to explore the details and not just the intuition.

[Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) The described work, solves the KL-divergence constrained optimization problem.

[Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) Solves the constrained KL-divergence problem, that is, it adds a penalty for divering the policy too much. The "clipped" objective of the paper is very simple and provides great results in practice (about equal to TRPO) for much less computation time and algorithmic complexity.

[Actor Critic with Kronecker Trust Regions](https://arxiv.org/pdf/1708.05144.pdf) Provides an approximate solution to TRPO using Kronecker Trust Regions which is basically superior in every way to TRPO. Reaches great results and provides only small overhead on top of A2C. Interesting they use the n-step return for the advantage estimation instead of Generalized Advantage Estimation which I do think hurts their performance a bit. They should have added this and tuned the lambda parameter.


Learning Resources:
* [http://karpathy.github.io/2016/05/31/rl/](Andrej Karpathy's Explanation) Very Simple, fantastic explanation of policy gradient. The key intuition is that the advantage A becomes your label if you think of this like a supervised learning problem.
* [Berkeley Deep RL Bootcamp Lecture 4](https://sites.google.com/view/deep-rl-bootcamp/lectures)
* [David Silver Lecture 7](https://www.youtube.com/watch?v=KHZVXao4qXs&t=15s)
* [Berkeley Deep RL Course John Schulman Lectures](https://www.youtube.com/watch?v=8jQIKgTzQd4&list=PLkFD6_40KJIwTmSbCv9OVJB3YaO4sFwkX)


Value Learning / Q - Learning

Key Algorithms
DQN
Double DQN
Dueling DQN
Distrubtional DQN (new updated paper a few days ago)
Noisy Nets
Prioritized Experience Replay
Rainbow (combines all of them)


__Exploration in Value-Learning__:

E-greedy remains the dominant technique. It is very simple and has sublinear regret in the contextual bandit setting (see David Silver's Lecture 9 for more information on this).

The parameter noise paper (above in Policy Gradient Exploration section) also shows really nice. A combination of a small bit of E-greedy and small bit of parameter space noise also might be a nice way to explore.



Other ideas

UNREAL
Nueral Episodic Control




# Imitation Learning / Behavorial Cloning
Simple approach to solving reinforcement learning problems. Just frame the problem completely as a supervised learning problem to predict the correct action given a dataset of human experience.

Can be surprisingly succesful given how simple it is to use in practice. For example in self driving cars (see Nvidia example), they simple train a deep ConvNet with outputs being the actuators of the car. The disadvantages of this are of course you need very large datasets of human experience which can be hard to collect.

One of the major practical issues is that of compounding errors. When the classifier makes a mistake and begins to see a trajectory and/or states that is hasn't seen in training, all bets are off whether it will work correctly. We hope (as is usual in deep networks), that the network can generalize over the input manifold to areas that is has never seen before, but we know in practice this is rarely the case.

Dataset Aggregation Algorithm (DAgger) is a simple online learning algorithm that looks to iteratively combat the compounding errors issue. You train on the dataset. Then run your policy at test time and record your observations. Then have a human label the observations with the correct state. In one sense, this is supposed to teach your algorithm how to correct itself when it makes errors.

[DART](http://bair.berkeley.edu/blog/2017/10/26/dart/) is a technique from berkeley to help do imitation learning by adding noise to the demonstrations. 

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


## Heirarchical Reinforcement Learning
Options framework
Fuedal Networks
Meta learning shared heirachies

## Inverse Reinforcement Learning
