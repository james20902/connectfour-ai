import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Connect(gym.Env):
  def __init__(self, width = 6, height = 7):
    self.width = width
    self.height = height

    # observations, store using 0 for empty, 1 for red, -1 for yellow
    self.observation_space = spaces.Box(-1,1,(width,height))
    self.action_space = spaces.Discrete(width)
    self.grid = np.zeros((width,height))

  def copy(self):
    out = Connect(self.width, self.height)
    out.grid = self.grid.copy()
    return out

  def _get_info(self):
      return None

  # might need to swap to making a copy
  def _get_obs(self):
    return self.grid
  
  def manual_adjust_grid(self, coord, color):
    self.grid[coord[0], coord[1]] = color

  def reset(self, seed=None, options=None):
    super().reset(seed=seed, options=options)
    self.grid = np.zeros((self.width, self.height))

    observation = self._get_obs()
    info = self._get_info()

    return observation, info

  # action is index where we drop, color is -1 or 1
  def step(self, action, color):
    col = self.grid[action]
    # find some way to enforce that we have an empty spot at the end (top) of the column
    # maybe just instantly lose the game and give big negative reward
    if col[-1] != 0: # we lose?
      return self._get_obs(), -10*color, True, False, None

    # find the first non occupied spot in our column
    ind = np.argmin(np.abs(col))

    # set it to the new color
    self.grid[action,ind] = color

    placement = np.array([action, ind])
    obs = self._get_obs()
    reward = self.get_reward(obs, placement, color)

    terminated = self.is_full(obs) or self.determine_win(self.grid, placement) != 0

    return obs, reward, terminated, False, None


  # only get a reward for winning
  def get_reward(self, observation, last_placement, color):
    w = self.determine_win(observation, last_placement)
    if w == color:
      return color
    else:
      return 0

  def is_full(self, grid):
    return not np.any(grid == 0)

  # determine if the game is over or not
  # grid is the grid
  # last_placement is the index (x,y) of our last placement
  # return 0 if no win, 1 if 1 wins, -1 if -1 wins
  def determine_win(self, grid, last_placement):
    # the 4 possible directions we can connect 4 along
    dirs = np.array([[1,0], [0,1], [1,1], [1,-1]])
    color = grid[last_placement[0], last_placement[1]]
    for vec in dirs:
      # we follow vec until we hit either an edge or another color
      # then we trace in the other direction and see how many in a row we have
      num = -1
      loc = np.copy(last_placement)
        
      while 0 <= loc[0] < len(grid) and 0 <= loc[1] < len(grid[0]) and grid[loc[0],loc[1]] == color:
        loc += vec
        num += 1
        
      loc = np.copy(last_placement)

      while 0 <= loc[0] < len(grid) and 0 <= loc[1] < len(grid[0]) and grid[loc[0],loc[1]] == color:
        loc -= vec
        num += 1

      if num >= 4:
        return color
    
    return 0

class Bot:
  botID = 0
  def __init__(self, name=None):
    self.name = name
    if name == None:
      self.name = "Random_bot" + str(Bot.botID)
      Bot.botID += 1
    self.color = None
    self.actions = None

  def instantiate(self, color, actions, env):
    self.color = color
    self.actions = actions
    self.env = env

  def get_move(self, observation):
    return self.actions.sample()

  def train(obs, action, new_obs, color):
    pass

class MiniMaxBot(Bot):
  total_rollout_collisions = 0
  eval_dict = {}
  def __init__(self, name=None):
    super().__init__(name)

  def get_move(self, observation):
    scores = []
    for action in range(self.actions.n):
      e = self.env.copy()
      obs, reward, terminated, _, _ = e.step(action, self.color)
      if not terminated:
        scores.append(self.look_forward(-self.color, 0, 3, e))
      else:
        scores.append(reward)
    # print(scores)
    # print(observation)
    return np.argmax(np.array(scores) * self.color)

  def look_forward(self, color, depth, max_depth, env):
    if depth < max_depth:
      scores = []
      for action in range(self.actions.n):
        e = env.copy()
        obs, reward, terminated, _, _ = e.step(action, color)
        if terminated:
          return reward
        s = self.look_forward(color*-1, depth+1, max_depth, e)
        scores.append(s)
      return color*max(np.array(scores) * color)
    else:
      return self.mcts(color, 10, env.copy())

  def mcts(self, color, num_rollouts, env):
    obs = str(env._get_obs())
    # implement little lookup table to speed things up
    if obs in MiniMaxBot.eval_dict:
      MiniMaxBot.total_rollout_collisions += 1
      # if MiniMaxBot.total_rollout_collisions % 1 == 0:
      #   print("eval_dict size", len(MiniMaxBot.eval_dict.keys()))
      return MiniMaxBot.eval_dict[obs]

    s = []
    for n in range(num_rollouts):
      s.append(self.rollout(color, env.copy()))
    score = sum(s)/num_rollouts
    MiniMaxBot.eval_dict[obs] = score
    return score

  def rollout(self, color, env):
    terminated = False
    while not terminated:
      action = env.action_space.sample()
      obs, reward, terminated, _, _ = env.step(action, color)
      color *= -1
    return reward
