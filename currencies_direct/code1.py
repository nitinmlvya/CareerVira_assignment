import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

ENV_NAME = 'CartPole-v0'

# Get the environment and extract the number of actions available in the Cartpole problem
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n


class ModelTrainer:
    def __init__(self):
        self.model = Sequential()
    
    def model_arch(self):
        # Model architecture
        self.model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        self.model.add(Dense(16))
        self.model.add(Activation('relu'))
        self.model.add(Dense(nb_actions))
        self.model.add(Activation('linear'))
        print(self.model.summary())
        
    def build_policy_and_compile(self):
        # Epsilon Greedy policy
        policy = EpsGreedyQPolicy()
        memory = SequentialMemory(limit=50000, window_length=1)
        self.dqn = DQNAgent(model=self.model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                       target_model_update=1e-2, policy=policy)
        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    def train_model(self):
        # Train the self.model
        self.dqn.fit(env, nb_steps=1000, visualize=True, verbose=2)

    def test_model(self):
        # Test the self.model.
        self.dqn.test(env, nb_episodes=5, visualize=True)

if __name__=='__main__':
    model_trainer = ModelTrainer()
    model_trainer.model_arch()
    model_trainer.build_policy_and_compile()
    model_trainer.train_model()
    model_trainer.test_model()