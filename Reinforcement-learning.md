# Introduction
This branch of machine learning is about training an agent by giving it rewards for performing correct actions

diiferent methods exist. we'll studyQ-learning

# Q-learning

## Definitions
- States: One particular arrangement of all the objects in the environment
- Actions: They are the edges between the states. They are the agent changing states.
- cue table: Each row correspond to a state. Each column correspond to an action. The values of this table represent the long term future expected rewards.
- Q-learning: is about letting the agent explores its environment and update the cue table with its experiences.
- Bellman equation: The long term valus for a cue table is represented by this equation

For very complex environement (many states and actions), we'll use deep learning to create cue tables.
The basic idea behind Q-learning is that we replace the Q-table with a neural net.



