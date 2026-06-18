
### Reinforcement Learning

The problems in reinforcement learning involve an agent interacting with an environment, which provides numeric reward signals.

Goal: Learn how to take actions in order to maximize the rewards

![[Pasted image 20260617234139.png]]

### Cart Pole Problem

Goal: Balance a pole on top of a moveable cart
States: angle, angular speed, position, horizontal velocity
Action: Horizontal force applied on the cart
Reward: 1 at each time step if the pole is upright

![[Pasted image 20260617234410.png]]

### Robot Locomotion

Objective: Make the robot move forward
State: Angle and position of the joints
Reward: 1 at each time step of the movement upright + forward movement

![[Pasted image 20260617234642.png]]

### Atari Games

Objective: Complete the game with the highest score
State: Raw pixel inputs of the game state
Action: Game Controls (left, right, up, down)
Reward: Score increase/decrease at each time step 

![[Pasted image 20260617234839.png]]

### Go

Objective: Win the game
State: Position of all the pieces
Action: Where to put the next piece down
Reward: 1 if win at the end of the game or 0

![[Pasted image 20260617235020.png]]

### How can we mathematically formalize this problem ?

##### Markov Decision Process
- Mathematical formulation of the RL problem
- Markov Property: Current state completely characterizes the state of the world.

a tuple of objects...
![[Pasted image 20260617235237.png]]

![[Pasted image 20260617235347.png]]

### Example --> A simple MDP: Grid World

![[Pasted image 20260617235523.png]]

![[Pasted image 20260617235644.png]]

- Random policy is nothing but, in any given state or cell you randomly sample which direction to proceed in, so all of these have equal probability.
- Optimal policy as the name suggests is nothing but taking the action that will move us closest to a terminal state.

![[Pasted image 20260618000042.png]]

![[Pasted image 20260618000221.png]]

![[Pasted image 20260618000411.png]]

![[Pasted image 20260618004246.png]]

What is the solution to this ?
- use a function approximator to estimate Q(s, a) E.g. Neural Network

![[Pasted image 20260618004531.png]]

![[Pasted image 20260618005115.png]]

### Case Study: Playing Atari Games

![[Pasted image 20260618010428.png]]

#### Experience Replay

Learning from batches of consecutive samples is problematic:
- Samples are correlated --> inefficient learning
- Current Q-network parameters determine the next training samples
	e.g: if maximizing action is to move left, training samples will be dominated by samples from left hand size and can lead to bad feedback loops

The way that we are going to address these problems is by using something called the experience relay

- continually update a replay memory table of transitions (s_t, a_t, r_t, s_t+1) as game episodes are being replayed (experience)
- Train Q-network on random minibatches of transitions from the replay memory, instead of consecutive samples.

Added benefit: each transition can also contribute to multiple weight updates --> greater data efficiency

![[Screenshot 2026-06-18 011446.png]]

### Policy Gradients

What is a problem with Q-learning ?
The Q-function can be very complicated

For example a robot grasping an object has a very high-dimensional state --> hard to learn the exact value of every state-action pair

But on the other hand your policy can be much simpler, something like just close your hand
The question here arises: Can we learn a policy directly, e.g. finding the best policy from a collection of policies without calculating your Q-value??

![[Pasted image 20260618012354.png]]

**What we can do is gradient ascent on our policy parameters**

![[Pasted image 20260618012712.png]]

![[Pasted image 20260618013655.png]]

![[Pasted image 20260618224752.png]]

![[Pasted image 20260618224821.png]]

![[Pasted image 20260618224844.png]]

![[Pasted image 20260618224953.png]]

![[Pasted image 20260618225024.png]]

![[Pasted image 20260618225042.png]]

![[Pasted image 20260618225106.png]]

![[Pasted image 20260618225123.png]]

![[Pasted image 20260618225156.png]]

![[Pasted image 20260618225219.png]]

![[Pasted image 20260618225238.png]]


