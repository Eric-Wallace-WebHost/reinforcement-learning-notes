# Model Free Based Reinforcement Learning
This is where I know the most, but it is still hard to keep up with Arxiv (in)sanity.






# Imitation Learning / Behavorial Cloning
Simple approach to solving reinforcement learning problems. Just frame the problem completely as a supervised learning problem to predict the correct action given a dataset of human experience.

Can be surprisingly succesful given how simple it is to use in practice. For example in self driving cars (see Nvidia example), they simple train a deep ConvNet with outputs being the actuators of the car. The disadvantages of this are of course you need very large datasets of human experience which can be hard to collect.

One of the major practical issues is that of compounding errors. When the classifier makes a mistake and begins to see a trajectory and/or states that is hasn't seen in training, all bets are off whether it will work correctly. We hope (as is usual in deep networks), that the network can generalize over the input manifold to areas that is has never seen before, but we know in practice this is rarely the case.

Dataset Aggregation Algorithm (DAgger) is a simple online learning algorithm that looks to iteratively combat the compounding errors issue. You train on the dataset. Then run your policy at test time and record your observations. Then have a human label the observations with the correct state. In one sense, this is supposed to teach your algorithm how to correct itself when it makes errors.

Learning References:
* [Berkeley Deep RL Course Lecture 2](https://www.youtube.com/watch?v=kl_G95uKTHw&list=PLkFD6_40KJIwTmSbCv9OVJB3YaO4sFwkX&index=2) discusses imitation learning, DAgger, and other techniques. Has links to papers and case studies

Key Papers:
* [End to End Learning for Self Driving Cars (Nvidia 2016)](https://arxiv.org/abs/1604.07316) Nvidia trains a self driving car using an end to end convolutional neural network. It is trained directly from human driving footage. They also do neat tricks with using three cameras that view the road from different angles (see berkeley course above for explanation)
* [Deep Imitation Learning for Complex Manipulation Tasks from Virtual Reality Teleoperation](https://arxiv.org/abs/1710.04615) Berkeley crew uses Virtual Reality Gear for Imitation Learning.

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

References:
* [Berkeley Deep RL Course Lecture 3-6](https://www.youtube.com/watch?v=mZtlW_xtarI&index=3&list=PLkFD6_40KJIwTmSbCv9OVJB3YaO4sFwkX) discusses model based reinforcement learning in depth.
* [Lecture 9 of Berkeley Reinforcement Learning Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures) discusses LQR, MPC, and guided policy search.
* [Lecture 8 of David Silver's Reinforcement Learning Course](https://www.youtube.com/watch?v=ItMutbeOHtc&index=8&list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-) discusses learning models, DYNA, and MCTS

Key Papers:




## Multi-Agent Domains

__TODO__: I don't know anything about this


Key Papers:
* [Learning to Learn with Opponent Learning Awareness (LOLA)](https://arxiv.org/abs/1709.04326)



A barebones CUDA-enabled PyTorch implementation of the CapsNet architecture in the paper "Dynamic Routing Between Capsules" by [Kenta Iwasaki](https://github.com/iwasaki-kenta) on behalf of Gram.AI.

Training for the model is done using [TorchNet](https://github.com/pytorch/tnt), with MNIST dataset loading and preprocessing done with [TorchVision](https://github.com/pytorch/vision).

## Description

> A capsule is a group of neurons whose activity vector represents the instantiation parameters of a specific type of entity such as an object or object part. We use the length of the activity vector to represent the probability that the entity exists and its orientation to represent the instantiation paramters. Active capsules at one level make predictions, via transformation matrices, for the instantiation parameters of higher-level capsules. When multiple predictions agree, a higher level capsule becomes active. We show that a discrimininatively trained, multi-layer capsule system achieves state-of-the-art performance on MNIST and is considerably better than a convolutional net at recognizing highly overlapping digits. To achieve these results we use an iterative routing-by-agreement mechanism: A lower-level capsule prefers to send its output to higher level capsules whose activity vectors have a big scalar product with the prediction coming from the lower-level capsule.

Paper written by Sara Sabour, Nicholas Frosst, and Geoffrey E. Hinton. For more information, please check out the paper [here](https://arxiv.org/abs/1710.09829).

__Note__: Affine-transformations for the data augmentation stage have not been implemented yet. This implementation only provides an efficient implementation for the dynamic routing procedure, example CapsNet architecture, and squashing functions mentioned in the paper.

## Requirements

* Python 3
* PyTorch
* TorchVision
* TorchNet
* TQDM
* Visdom

## Usage

**Step 1** Adjust the number of training epochs, batch sizes, etc. inside `capsule_network.py`.

```python
BATCH_SIZE = 100
NUM_CLASSES = 10
NUM_EPOCHS = 30
NUM_ROUTING_ITERATIONS = 3
```

**Step 2** Start training. The MNIST dataset will be downloaded if you do not already have it in the same directory the script is run in. Make sure to have Visdom Server running!

```console
$ sudo python3 -m visdom.server & python3 capsule_network.py
```

## Benchmarks

Highest accuracy was 99.48% after 30 epochs. The model may achieve a higher accuracy as shown by the trend of the test accuracy/loss graphs below.

![Training progress.](media/Benchmark.png)

Default PyTorch Adam optimizer hyperparameters were used with no learning rate scheduling. Epochs with batch size of 100 takes ~3 minutes on a Razer Blade w/ GTX 1050. 

## TODO

* Extension to other datasets apart from MNIST.

## Credits

Primarily referenced these two TensorFlow and Keras implementations:
1. [Keras implementation by @XifengGuo](https://github.com/XifengGuo/CapsNet-Keras)
2. [TensorFlow implementation by @naturomics](https://github.com/naturomics/CapsNet-Tensorflow)

Many thanks to [@InnerPeace-Wu](https://github.com/InnerPeace-Wu) for a [discussion on the dynamic routing procedure](https://github.com/XifengGuo/CapsNet-Keras/issues/1) outlined in the paper.

## Contact/Support

Gram.AI is currently heavily developing a wide number of AI models to be either open-sourced or released for free to the community, hence why we cannot guarantee complete support for this work.

If any issues come up with the usage of this implementation however, or if you would like to contribute in any way, please feel free to send an e-mail to [kenta@gram.ai](kenta@gram.ai) or open a new GitHub issue on this repository.
