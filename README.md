# Reinforcement Learning using Keras
The networks for the RL algorithms are coded entirely in Keras, including choosing the optimizers and calculation of the loss functions. <br /><br />


This library has been heavily inspired by: <br />
https://github.com/higgsfield/RL-Adventure <br />
https://github.com/higgsfield/RL-Adventure-2 <br />
https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python <br />
https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python <br />
https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence <br />
https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code <br />
https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code <br />

Based from the above inspiration, I have created various deep RL agents including DDPG and TD3. On top of all of this, I have developed agents using Decision Trees where actions are determined based on the current state. On top of all of this, implementation has been added for these agents to learn with the assistance of a heuristic policy as Deep RL alone may not work quickly as possible. 
The intent is for this library to work with OpenAI Gym style environments <br />

Setting up the library is simple as follows <br />
1. Navigate to the root of the keras_rl directory on your local machine on the terminal<br />
2. Run python setup.py sdist <br />
3. Run python -m pip install dist/keras-rl-1.0.tar.gz <br />
