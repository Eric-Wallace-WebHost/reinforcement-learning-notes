This repository contains notes on a number of Reinforcement Learning papers and algorithms. The main focus is on Deep Reinforcement Learning with papers starting from 2013.

If you are new to the field, I recommend taking a look at David Silver's online course, as well as Berkeley's course on Deep RL.

# Model Free Reinforcement Learning

A general class of algorithms that makes no attempt to learn the underlying dynamics of the system, nor do any planning in the future. This can cause large sample inefficiencies, but in practice these algorithms currently learn the best policies out of all classes of algorithms.

## Policy Gradient Methods

Directly optimize the policy by analytically computing the gradient using the "REINFORCE" or likelihood ratio trick. 

In practice, these techniques using a neural network as an approximator for the policy, and follow an algorithm similar to Policy Iteration. They do a "policy evaluation" step where they compute rollouts in the environment to see how the current policy is doing. They then update the policy in order to get more of the good rewards it say.

These algorithms are extremely well suited for learning continuous control tasks such as the MuJoCo simulator. They sometimes have worse sample complexity than Q-Learning algorithms as it is difficult to learn off-policy in policy gradient techniques.

__The Gradient Step__:

Your gradient step tries to make actions that led to good results more common. Andrej Karpathy provides a great analogy to supervised learning.

[Gradient Step for Policy Gradients](https://github.com/Eric-Wallace/reinforcement-learning-notes/blob/master/images/gae.png)

The high level view of the vanilla Policy Gradient Algorithm looks like the following.

[Vanilla Policy Gradient Psuedo-Code/Algorithm](https://github.com/Eric-Wallace/reinforcement-learning-notes/blob/master/images/vanillaPG.png)

__Estimating the Advantage Function__:

To get a good estimate of the advantage function, this is what is done in practice. 

Your neural network outputs V(s) and you estimate Q(s) either using n-step returns or Generalized Advantage Estimation (exponentially weighted returns just like TD(lambda)). 

[Slide showing Generalized Advantage Estimation](https://github.com/Eric-Wallace/reinforcement-learning-notes/blob/master/images/GAE.png)

[Slide showing N-Step Return](https://github.com/Eric-Wallace/reinforcement-learning-notes/blob/master/images/nstepreturn.png)

[Asynchronous Advantage Actor Critic](https://arxiv.org/abs/1602.01783) Use n-step returns and a neural network to approximate the advantage function. Use shared convolutional weights for the policy network. Train using parallel workers and get really good results.

[Generalized Advantage Estimation (2016)](https://arxiv.org/abs/1506.02438) presents a better method to approximate the advantage function using an exponentially weighted average, similar to TD(lambda) 

[The Adapted Algorithm/Psuedo-Code](https://github.com/Eric-Wallace/reinforcement-learning-notes/blob/master/images/a3cClean.PNG)

[Another image of Psuedo-Code](https://github.com/Eric-Wallace/reinforcement-learning-notes/blob/master/images/a3c.PNG)

__Continuous Control Focused__:

[Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971) Rather than outputting a probability distribution like a Guassian + Variance term, they have a deterministic policy. They then get gradient information directly from the critic instead 
of from the policy.

[Benchmarking Continuous Control](https://arxiv.org/abs/1604.06778) Provides a fantastic survey and benchmarking of different algorithms. It does not include PPO or ACKTR as it is from 2016. 

[Towards Generalization and Simplicity in Continuous Control](https://arxiv.org/abs/1703.02660) A bit of a provactative paper showing that linear and RBF policies are competitive on a lot of tasks.

[Distributed Distributional Deep Deterministic Policy Gradients](https://openreview.net/pdf?id=SyZipzbCb) uses DDPG + distributional Bellman idea + distributed system from Ape-X (paper in parallel section) for continuous control tasks.

[Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/abs/1603.00748) not a policy gradient method. Shows how to apply Deep Q Learning to Continuous Control Tasks.

[Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286) use a distributed PPO to train locomotion on hard domains like parkour and get some really cool results.

__Other Ideas__:

[Sample-Efficient Actor Critic with Experience Replay](https://arxiv.org/abs/1611.01224) Describes a method to adding an experience replay and off-policy learning to actor critic algorithms. They do this through a truncated importance sampling technique that allows you to learn off policy. They also present some interesting things like using an average of past policies as an approximation to TRPO. Though I think that last point is unneccesary when you have ACTKR or PPO.

__Exploration in Policy Gradients__:

The A3C paper (above) adds an entropy term that helps to encourage exploration. In a theoretical sense this term should try to make the policy more uniform (uniform distribution is maximum entropy). I haven't seen this used very often though, and the other techniques (simply adding noise) are simpler and show good results. A mix of the two techniques with low settings on their parameters also might be a promising technique. Entropy is a "state-space" exploration technique because it encourages the network to randomly explore around.  

[Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295) also seen in OpenAI paper which has similar idea. It basically just adds random noise to your policy to help explore more. In Value-Learning e-greedy techniques do "state-space exploration", where this techniques is a "parameter space exploration" technique.

__Natural Gradient Algorithms__:

There has been a lot of interest in algorithms that simulate a natural gradient step. The main intuition behind these optimization methods in general is that rather than taking a step in parameter space, take a step in function space. [This blog post](http://kvfrans.com/what-is-the-natural-gradient-and-where-does-it-appear-in-trust-region-policy-optimization/) provides a fantastic intuitive explanation. 

Now in supervised learning, the general trend has been to do more "dumb steps" rather than less "smart steps". Don't worry about computing the Hessian, doing L-BFGS, doing any second order information. Hell, you don't even need fresh gradients in an asynchronous setting as noise can be a great regularizer. But in Reinforcement Learning, we have a lot more issues. First off, we want to be more sample efficient. Secondly, and much more importantly, is that in Reinforcement Learning changes to our policy will change the data we see in the future. If we take a step that is too big, we can destroy everything we have learned in the worst case. That is the motivation behind the name "Trust Regions", we want to stay in the regions of the RL problem that we have explored a lot and trust our network in those areas.

[Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) The described work, solves the KL-divergence constrained optimization problem.

[Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) Solves the constrained KL-divergence problem, that is, it adds a penalty for divering the policy too much. The "clipped" objective of the paper is very simple and provides great results in practice (about equal to TRPO) for much less computation time and algorithmic complexity.

[Actor Critic with Kronecker Trust Regions](https://arxiv.org/pdf/1708.05144.pdf) Provides an approximate solution to TRPO using Kronecker Trust Regions which is basically superior in every way to TRPO. Reaches great results and provides only small overhead on top of A2C. Interesting they use the n-step return for the advantage estimation instead of Generalized Advantage Estimation which I do think hurts their performance a bit. They should have added this and tuned the lambda parameter.

__Using Reinforce-Like Methods in NLP and Computer Vision__:

When using non-differentiable blocks in other systems like "hard attention" models in Image Captioning, you can use Reinforce like algorithms to provide gradient information. John Schulman provided a generalized framework for computation graphs that involve "stochastic computation" like hard attention models. [Stochastic Computation Graphs](https://arxiv.org/abs/1506.05254) is the paper, which I need to review as well as his slides from the Bootcamp/Course describing this. 

## Value Learning / Q - Learning

This class of algorithm is a generalization of the Value Iteration algorithm. Runs the current policy to collect information about the Q-Values of the various states. The policy is implicit, you simply select the best Q value in each state. 

In practice, these can be more sample efficient that policy gradient methods because you can learn off-policy. That is, rather than using SARSA-like algorithms, Q-Learning is used in practice. This gives the advantage that we can use an experience replay and train our neural network completely off policy. The one disadvantage of Q-Learning is that you can't easily use eligibility traces as is done in SARSA(lambda). Thats why the original DQN paper simply used the one-step Q estimate to train. This has disadvantages because the reward don't directly flow backwards through the states. Though this can be overcome by using n-step returns in the forward view as is done in Rainbow.

__The Gradient Step__:

Q-Learning is simply a regression problem, where the neural network attempts to predict the n-step (normally n = 1) return by taking that step. 

[Psuedo-Code/Algorithm for Deep Q-Learning](/images/dqn.png)

__Algorithmic Improvements__:

There has been a whole host of improvements to Deep Q-learning. Many of these algorithms provide orthogonal improvements to one another. They have been combined in the Rainbow paper. 

[Deep Q Networks Original Paper (Nature Version)](http://www.davidqiu.com:8888/research/nature14236.pdf) The paper that started it all.

[Double Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461) describes using theoretical and empirical results that because Q-Learning takes the max of Q over a series of states, it can have a large over estimation of the Q value. They show that this is harmful to learning and propose a very slight tweak to the network target in the gradient step equation. This provides good results, adds no computation time, and is probably a few lines of code change (aka it should definitely be used!).

[Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) describes a way to structure your neural network architecture to implicility model the Q value as (V(s) + A(s)). This has nice properties as you can implicility model the Value function even though you are training for Q values. This doesn't require any obvious algorithmic changes, but you need to change you network structure to have two heads. You need to combine those two heads using a network they describe in the paper. 

[Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) is an idea that fixes the naive uniform random sampling from the experience replay used in DQN. The paper talks at length at how you can make this more efficient, because otherwise you will have to do computation and search over all elements in the experience replay (~ 1,000,000 elements) which will be super expensive for every step in the network. They use some nice ideas like KD trees to do this.  

N-Step Q-Learning doesn't have a paper as it is a very idea but it is used in the Rainbow paper. Due to the fact that we are using e-greedy Q-Learning and learning off-policy, we can't easily use eligibility traces. You can get around this and doing Q(lambda) by doing something like keep track of whenever you acted randomly, and stop updating the rewards at that point. This can be quite awkward though. A simple idea is just use an n-step target for the update. This is also a little awkward as the first state will have n-steps of reward to approximate and the final non-terminal state will just have a 1-step target, but that is okay. This should get nice gains by reducing variance, especially early in the training phase.

[Rainbow: Combining improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298) combines all of the above techniques and the distributional c-51 algorithm below into one algorithm that does super well.

[Prioritized Distributed Experience Replay](https://openreview.net/forum?id=H1Dy---0Z&noteId=H1Dy---0Z) uses some of the improvements to Rainbow in a distributed manner. There are a ton of workers who collect rollouts (each with different e-greedy values), and then one learner on a GPU who does minibatches on a massive experience replay shared amongst all the workers. The learner samples in a prioritized way according to the absolute bellman error. It does really well. You could imagine this + distributional bellman just crushing it.

[Recurrent Q-Networks](https://arxiv.org/abs/1609.05521) uses an LSTM to model the Q-value. One of the central ideas is how do you use an experience replay, because the hidden state will be 0 when you start replaying when it wouldn't be 0 when your on policy. Some ideas are to store X frames (i.e. like 10) and then use the first 5 to warm up your LSTM, then only compute gradients for the last five. Another idea is to store the LSTM hidden state in the experience replay and initialize with that. But then you can't really compute graidents into that stored hidden state. You could finally just pretend it doesn't happen and intialize with all 0's and this paper tries that and it works fine (yay deep learning!!).

[Pop-Art DQN](https://arxiv.org/pdf/1602.07714.pdf) applies an adaptive reward scaling algorithm to allow DQN to function with rewards of various scales (i.e. not clipped to -1, 1). 

__Distributional Bellman Equations__:

The intuition behind distributional Q-Learning is that rather than outputting the expected value of Q (i.e. a single scalar), we should output a distribution of Q values (in practice this is a bucketed distribution, like output 51 probability buckets of Q values). This has the advantage of being able to estimate multi-modal or skewed distributions where we might want to act differently than just the expected Q-Value.

_TODO_: I still need to go through these papers and perhaps read the older literature to get the math working out.

[A Distributional Perspective on Reinforcement Learning](https://arxiv.org/pdf/1707.06887.pdf) is the original paper to propose this method in Deep Reinforcement Learning. 

[Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/pdf/1710.10044.pdf) is an improvement to the above paper. 

[Distributed Distributional Deep Deterministic Policy Gradients](https://openreview.net/pdf?id=SyZipzbCb) uses DDPG + distributional Bellman idea + distributed system from Ape-X (paper in parallel section) for continuous control tasks.

__Exploration in Value-Learning__:

_State Space Exploration_:

E-greedy remains the dominant technique. It is very simple and has sublinear regret in the contextual bandit setting (see David Silver's Lecture 9 for more information on this).

E-greedy
Boltzmann Sampling
[The Uncertainty Bellman Equation and Exploration](https://arxiv.org/abs/1709.05380) creates a new backup operator that expresses the agent's uncertainty about certain states. The agent is then rewarded for exploring the states it is uncertain about.
Count Based Exploration
Instrinsic Motivation

https://arxiv.org/pdf/1709.05380.pdf

https://arxiv.org/pdf/1606.01868.pdf
Follow up work to ^ https://arxiv.org/abs/1703.01310

https://arxiv.org/pdf/1602.04621.pdf


_Parameter Space Exploration_:

The parameter noise paper (above in Policy Gradient Exploration section) also shows really nice. A combination of a small bit of E-greedy and small bit of parameter space noise also might be a nice way to explore.


__Aditional Exploratory Work__:

[Sample-Efficient Actor Critic with Experience Replay](https://arxiv.org/abs/1611.01224) Describes a method to adding an experience replay and off-policy learning to actor critic algorithms. They do this through a truncated importance sampling technique that allows you to learn off policy. They also present some interesting things like using an average of past policies as an approximation to TRPO. Though I think that last point is unneccesary when you have ACTKR or PPO.

[Reinforcement Learning with Unsupervised Auxiliary Tasks (UNREAL)](https://arxiv.org/abs/1611.05397) adds auxiliary tasks to a reinforcement learning agent such as pixel control. The agent reuses some of the neural network architecture for these auxiliary tasks, so it can help to do stuff like do additional training of convolutional layers. 

I wonder if anyone has tried to do pretraining of convolution layers. Say do object detection on another tasks like ImageNet, then transfer the lower layers to your agent and initialize them from there.

[Neural Episodic Control](https://arxiv.org/abs/1703.01988) uses the same ideas you would see in a memory-augmented LSTM like "Differentiable Neural Computer" or "End to End Memory Networks". Rather than storing the Q-Values in the network parameters, the convolution layers produce a vector that is used to read/write from a memory that stores the Q-Values. It learns very fast but doesn't each as high performance as other networks over many episodes.

[Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) Use Q functions that also have a goal associated with them. Then, even when you make a mistake, you can get reward by changing the goal in hindsight and still getting reward.

# Parallel Training of Reinforcement Learning:

[Gorila](https://arxiv.org/pdf/1507.04296.pdf) demonstrates training the DQN algorithm in a distributed computation setting across many parallel workers. It uses a shared parameter server as is common in the Google Brain research work. It shows nice speed ups and good results.

[Distributed Prioritized Experience Replay](https://openreview.net/pdf?id=H1Dy---0Z) gets amazing results. They have a bunch of workers that sample from the environment and store results in a central prioritized replay. One learner on a GPU samples from that replay and computes gradient updates. The workers refresh their parameters every now and then. They get to about double the median performance as Rainbow did using this method.

[Distributed Distributional Deep Deterministic Policy Gradients](https://openreview.net/pdf?id=SyZipzbCb) uses DDPG + distributional Bellman idea + distributed system from Ape-X (paper directly above this) for continuous control tasks.

# Imitation Learning / Behavorial Cloning / Learning from Demonstrations
Simple approach to solving reinforcement learning problems. Just frame the problem completely as a supervised learning problem to predict the correct action given a dataset of human experience.

Can be surprisingly succesful given how simple it is to use in practice. For example in self driving cars (see Nvidia example), they simple train a deep ConvNet with outputs being the actuators of the car. The disadvantages of this are of course you need very large datasets of human experience which can be hard to collect.

One of the major practical issues is that of compounding errors. When the classifier makes a mistake and begins to see a trajectory and/or states that is hasn't seen in training, all bets are off whether it will work correctly. We hope (as is usual in deep networks), that the network can generalize over the input manifold to areas that is has never seen before, but we know in practice this is rarely the case.

Dataset Aggregation Algorithm (DAgger) is a simple online learning algorithm that looks to iteratively combat the compounding errors issue. You train on the dataset. Then run your policy at test time and record your observations. Then have a human label the observations with the correct state. In one sense, this is supposed to teach your algorithm how to correct itself when it makes errors.

[DART](http://bair.berkeley.edu/blog/2017/10/26/dart/) is a technique from berkeley to help do imitation learning by adding noise to the demonstrations. 

Learning References:
* [Berkeley Deep RL Course Lecture 2](https://www.youtube.com/watch?v=kl_G95uKTHw&list=PLkFD6_40KJIwTmSbCv9OVJB3YaO4sFwkX&index=2) discusses imitation learning, DAgger, and other techniques. Has links to papers and case studies

Key Papers:
* [End to End Learning for Self Driving Cars (Nvidia 2016)](https://arxiv.org/abs/1604.07316) Nvidia trains a self driving car using an end to end convolutional neural network. It is trained directly from human driving footage. They also do neat tricks with using three cameras that view the road from different angles (see berkeley course above for explanation)
* [Learning from Demonstrations for Real World Reinforcement Learning](https://arxiv.org/pdf/1704.03732.pdf) DQfD. They use a small set of demonstrations on Atari games to initialize a Deep Q Learning algorithm, then run the algorithm in the environment and let it learn. Learns really quick using a combination of the two. 

* [Deep Imitation Learning for Complex Manipulation Tasks from Virtual Reality Teleoperation](https://arxiv.org/abs/1710.04615) This paper is really good. They use a VR headset to directly demonstrate how a robot should do things like grasping. It is able to learn directly from pixels to actions in only about 30 minutes or less of human demonstrations.

__Third Person Imitation Learning__:
Third person imitation learning is obviously a bold goal, and would have fantastic implications. Like perhaps a robot could just watch youtube all day at 10000x speed and then learn how to do everything. 

The general approach people have tried is to learn some sort of embedding that is invariant to different viewpoints. For example, when you are from a third person or first person view of someone pouring water in a cup, the embedding space should be the same. Some techniques have used domain confusion to train a discriminator that attempts to classify whether the embedding space is from one view or another. Once the discriminator has been fooled, we have invariant embeddings. You can then try to match the third person imitator by making your first person attempt match their third person embedding.

[Time-Contrastive Networks](https://arxiv.org/abs/1704.06888z) show some really strong results. They learn invariant embeddings by training on both first and third person video. They have a clever idea where they use a triplet loss, frames that happen at the same time step (i.e. water is about to exit a cup) from different angles must have the same embedding; frames from different time steps but the same angle must have different embeddings. Then have the robot uses the learned embedding and reinforcement learning to try to match the demonstration.

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
_TODO_: 
Finish writing this section after watching rest of Berkeley Lectures

## Model Based & Model Free Combinations
DYNA Algorithm

__TODO__: This is mentioned in David Silver's lectures

Learning References:
* [Berkeley Deep RL Course Lecture 3-6](https://www.youtube.com/watch?v=mZtlW_xtarI&index=3&list=PLkFD6_40KJIwTmSbCv9OVJB3YaO4sFwkX) discusses model based reinforcement learning in depth.
* [Lecture 9 of Berkeley Reinforcement Learning Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures) discusses LQR, MPC, and guided policy search.
* [Lecture 8 of David Silver's Reinforcement Learning Course](https://www.youtube.com/watch?v=ItMutbeOHtc&index=8&list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-) discusses learning models, DYNA, and MCTS

Guided Policy Search


Predictron

Value Iteration Networks

## Multi-Agent Domains

# Self Play
AlphaGo Zero of course
OpenAI Dota

[Emergent Complexity via Multi-Agent Competition](https://arxiv.org/abs/1710.03748) shows that self play can learn really complex stuff

[Opponent Modeling in Deep Reinforcement Learning](https://arxiv.org/abs/1609.05559) models the opponent by generating an additional feature vector from their actions. 

# Cooperation Amongst Agents

[Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926) 
[Multi-Agent Actor Critic](https://arxiv.org/abs/1706.02275)
These two papers use actor critic with a centralized critic for mulit-agent domains

[Learning to Learn with Opponent Learning Awareness (LOLA)](https://arxiv.org/abs/1709.04326) In a cooperative setting, accounts for the fact that the other agent is also learning

https://arxiv.org/pdf/1703.10069.pdf

Key Papers:

## Derivative Free Optimization
A broad class of algorithms that consists of things like Genetic Algorithms, Evolutionary Algorithms, Evolutionary Search, and closely related others. Most of these algorithms are essentially random parameter search + heuristics. In the case when a derivative can be computed analytically (supervised learning, Q learning, policy gradients using reinforce trick), these algorithms are dumb because they don't move in that direction.

The only main advantage of these algorithms is that they can be scaled really well in parallel. The best paper is OpenAI's work that uses a clever trick where the seperate workers exchange what random seed they used rather than the actual parameters of the system.

Learning References:
[Lecture 8 of Berkeley Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)

Key Papers:
[Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864)
[Evolution Strategies Package and Blog Post](http://blog.otoro.net/2017/11/12/evolving-stable-strategies/)

## Sim2Real Transfer
Most of the exciting recent work has shown that your simulator does not need to be accurate, but rather, it needs to have a high degree of variability for things you want your model to be invariant to. For example, random lighting, colors, shapes, etc. will help your model generalize well. Recent work has shown that if you randomize the dynamics and/or visualize environment you can transfer to the real world with no real world training. 

[Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/pdf/1703.06907.pdf) uses random changes to the environment for better generalization. Such as changes in lighting and colors.

[Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/pdf/1710.06537.pdf) show how using randomization of the dynamics of the simulation can help. For example, the friction of the surface the object is on. They use an LSTM which is suppose to be able to use its memory to learn how the dynamics behaves to make decisions.

[Asymmetric Actor Critic for Image-Based Robot Learning](https://arxiv.org/pdf/1710.06542.pdf) proposes a nice idea that since we are in a simulator, you can cheat and give the critic full state information. The actor won't get anything special, but the critic will. This allows you to train and learn a lot faster.


## Heirarchical Reinforcement Learning

Most of the approaches here use a high level controller that selects which low level controller to do the job. One famous approach is the options framework. The options are low level policies which run until some termination condition.   

[Option Critic](https://arxiv.org/abs/1609.05140) framework selects low level option pieces from a high level policy.
[Fuedal Networks](https://arxiv.org/abs/1703.01161) use a high level manager that issues goals to its workers. Depending on the goal, workers are selected and then they execute their sub-policy. The workers are trained for the reward and back propagated through using normal policy gradient methods. The high level manager is trained based on what the underlying workers did, not what the goal it sent was. It gets some really nice results and seems overall better and cleaner than option-critic.
[Meta Learning Shared Heirachies](https://arxiv.org/abs/1710.09767) incorporates meta learning into heirarchical structure.

## Inverse Reinforcement Learning

Alternative to Imitation Learning, where instead we try to learn the reward function from expert demonstrations. Once we have the reward function, we can then use more general Reinforcement Learning techniques to solve the problem.

## Meta Learning

The general approach is to learn a policy that is good at learning in new environments. So "Meta" learning is that we are training an agent that is good at learning. 

Popular Papers
RL^2 -> Uses an RNN as the meta learner
Model Agnostic Meta Learning -> Make your parameters so that they are a few gradients steps away from a good policy
Simple Neural Attentive Meta learner -> Replaces RL^2 with a dilated convolution

[One-Shot Visual Imitation Learning via Meta-Learning](http://proceedings.mlr.press/v78/finn17a/finn17a.pdf) they run numerous imitation learning tasks and then use MAML to find settings of the parameters that makes the network good at Imitation Learning. 




It seems like doing Imitation Learning via a VR system, and running meta learning to get good parameters, and then maybe using the learned model and fine tuning it in simulation could get some really nice results. The next steps would be how do you combine many learned policies into one system that can actually be intelligent. Heirarchies? Who knows.

## Transfering Policies / Multi-Task Reinforcement Learning

[Policy Distillation](https://arxiv.org/abs/1511.06295) trains a number of different agents for individual Atari games. Then they run each of the different agents for a while and collect an experience replay for each one. They then train a single agent that emulates the results of all of the individual agents. The single agent is basically able to do as well as all the individual agents.
[Actor Mimic](https://arxiv.org/abs/1511.06342) is basically the exact same approach as policy distillation. 
[Progressive Reinforcement Learning with Distillation for Multi-Skilled Motion Control](https://openreview.net/pdf?id=B13njo1R-) applies policy distillation to the continuous setting and adds a few additions.
[Distral](http://papers.nips.cc/paper/7036-distral-robust-multitask-reinforcement-learning.pdf) they use the same idea as policy distillation, but it runs in a more online fashion and uses the distilled policy to help regularize the individual experts. 


## Learning Resources

* [Andrej Karpathy's Explanation](http://karpathy.github.io/2016/05/31/rl) Very Simple, fantastic explanation of policy gradient. The key intuition is that the advantage A becomes your label if you think of this like a supervised learning problem.
* [Berkeley Deep RL Bootcamp Lectures](https://sites.google.com/view/deep-rl-bootcamp/lectures)
* [David Silver Lectures](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PL7-jPKtc4r78-wCZcQn5IqyuWhBZ8fOxT)
* [John Schulman Lectures](https://www.youtube.com/watch?v=aUrX-rP_ss4)
* [Berkeley Deep Reinforcement Learning Course](http://rll.berkeley.edu/deeprlcourse/)

