from time import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Connect(gym.Env):
  def __init__(self, width = 6, height = 7):
    self.width = width
    self.height = height
    self.terminated = False
    self.last_reward = 0
    # observations, store using 0 for empty, 1 for red, -1 for yellow
    self.observation_space = spaces.Box(-1,1,(width,height))
    self.action_space = spaces.Discrete(width)
    self.grid = np.zeros((width,height))

  def copy(self):
    out = Connect(self.width, self.height)
    out.grid = self.grid.copy()
    return out

  def manual_adjust_grid(self, coord, color):
    self.grid[coord[0], coord[1]] = color

  def _get_info(self):
      return None

  # might need to swap to making a copy
  def _get_obs(self):
    return self.grid

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self.last_reward = 0
    self.terminated = False
    self.grid = np.zeros((self.width, self.height))

    observation = self._get_obs()
    info = self._get_info()

    return observation, info

  def get_legal_moves(self):
    return np.where(self.grid[:,-1] == 0)[0]

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
    self.last_reward = reward
    terminated = self.is_full() or determine_win(self.grid, placement) != 0
    self.terminated = terminated
    return obs, reward, terminated, False, None


  # only get a reward for winning
  def get_reward(self, observation, last_placement, color):
    w = determine_win(observation, last_placement)
    if w == color:
      return color
    else:
      return 0

  def render_frame(self, screen, window_width, window_height):
    pygame.draw.rect(screen, pygame.Color(255, 255, 255), pygame.Rect(0, 0, window_width, window_height))

    square_width = window_width / self.width
    square_height = window_height / self.height

    for w in range(self.width):
        for h in range(self.height):
            if self.grid[w, h] == 1:
                pygame.draw.rect(screen, pygame.Color(209, 49, 31), pygame.Rect(square_width*w, square_height*h, square_width, square_height))
            elif self.grid[w, h] == -1:
                pygame.draw.rect(screen, pygame.Color(236, 242, 44), pygame.Rect(square_width*w, square_height*h, square_width, square_height))
            else:
                pygame.draw.rect(screen, pygame.Color(255, 255, 255), pygame.Rect(square_width*w, square_height*h, square_width, square_height))

  def hash(self):
    return

  def is_full(self):
    return not np.any(self.grid == 0)
  # def is_full(self,grid):
  #   return not np.any(grid == 0)
  # determine if the game is over or not
  # grid is the grid
  # last_placement is the index (x,y) of our last placement
  # return 0 if no win, 1 if 1 wins, -1 if -1 wins
  def determine_win(self, grid, last_placement):
    last_placement = np.array(last_placement)
    # the 4 possible directions we can connect 4 along
    dirs = np.array([[1,0], [0,1], [1,1], [1,-1]])
    color = grid[last_placement[0], last_placement[1]]
    for vec in dirs:

      # we follow vec until we hit either an edge or another color
      # then we trace in the other direction and see how many in a row we have
      num = -1
      loc = np.copy(last_placement)
      while 0 <= loc[0] < len(grid) and 0 <= loc[1] < len(grid[0])\
      and grid[loc[0],loc[1]] == color:

        loc += vec
        num+=1
      loc = np.copy(last_placement)
      while 0 <= loc[0] < len(grid) and 0 <= loc[1] < len(grid[0])\
      and grid[loc[0],loc[1]] == color:
        loc -= vec
        num += 1
      if num >= 4:
        return color
    return 0


def determine_win(grid, last_placement):
    last_placement = np.array(last_placement)
    # the 4 possible directions we can connect 4 along
    dirs = np.array([[1,0], [0,1], [1,1], [1,-1]])
    color = grid[last_placement[0], last_placement[1]]
    for vec in dirs:

      # we follow vec until we hit either an edge or another color
      # then we trace in the other direction and see how many in a row we have
      num = -1
      loc = np.copy(last_placement)
      while 0 <= loc[0] < len(grid) and 0 <= loc[1] < len(grid[0])\
      and grid[loc[0],loc[1]] == color:

        loc += vec
        num+=1
      loc = np.copy(last_placement)
      while 0 <= loc[0] < len(grid) and 0 <= loc[1] < len(grid[0])\
      and grid[loc[0],loc[1]] == color:
        loc -= vec
        num += 1
      if num >= 4:
        return color
    return 0


## Bots and running tournaments
# Default bot is random
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
  def __init__(self, name=None, depth = 2):
    super().__init__(name)
    self.discount = 0.95
    self.depth = depth

  def get_move(self, observation):
    scores = []
    actions = []
    for action in self.env.get_legal_moves():
      e = self.env.copy()
      actions.append(action)
      obs, reward, terminated, _, _ = e.step(action, self.color)
      if not terminated:
        scores.append(self.look_forward(-self.color, 0, self.depth, e))
      else:
        # print("Found terminal")
        scores.append(reward)
    # print(scores)
    # print(observation)
    return actions[np.argmax(np.array(scores) * self.color)]

  def look_forward(self, color, depth, max_depth, env):
    if depth < max_depth:
      scores = []
      for action in env.get_legal_moves():
        e = env.copy()
        obs, reward, terminated, _, _ = e.step(action, color)
        if terminated:
          return reward
        s = self.look_forward(color*-1, depth+1, max_depth, e)
        scores.append(s)
      return color*max(np.array(scores) * color) * self.discount
    else:
      return self.mcts(color, 6, env.copy()) * self.discount

  def mcts(self, color, num_rollouts, env):
    obs = str(env._get_obs())
    # implement little lookup table to speed things up
    # if obs in MiniMaxBot.eval_dict:
    #   MiniMaxBot.total_rollout_collisions += 1
    #   # if MiniMaxBot.total_rollout_collisions % 1 == 0:
    #   #   print("eval_dict size", len(MiniMaxBot.eval_dict.keys()))
    #   return MiniMaxBot.eval_dict[obs]

    s = []
    for n in range(num_rollouts):
      s.append(self.rollout(color, env.copy()))
    score = sum(s)/num_rollouts
    MiniMaxBot.eval_dict[obs] = score
    return score

  def rollout(self, color, env):
    terminated = False
    while not terminated:
      action = np.random.choice(env.get_legal_moves())
      obs, reward, terminated, _, _ = env.step(action, color)
      color *= -1
    return reward

class MCTSBot(Bot):
  # hashes position to tuple of (# wins for color 1, # of visits) only inside if fully expanded
  eval_dict = {}
  def __init__(self, name=None):
    super().__init__(name)
    self.update_list = []

  def get_move(self, observation):
    self.update_list = []
    now = time()
    while time() - now < 4:
      en = self.env.copy()
      en, color = self.traverse(en, self.color) # en
      rollout_result = self.rollout(en,color)
      #update weights
      for node_str in self.update_list:
        if node_str not in MCTSBot.eval_dict:
          MCTSBot.eval_dict[node_str] = [0,0]
        node = MCTSBot.eval_dict[node_str]
        if rollout_result > 0:
          node[0] += 1
        node[1] += 1

    vals = []
    for move in self.env.get_legal_moves():
      en = self.env.copy()
      en.step(move, color)
      try:
        node = MCTSBot.eval_dict[str(en._get_obs())]
        vals.append(node[0]/node[1])
      except:
        vals.append(-100 * self.color)
        # print(en.grid)

    return np.argmax((np.array(vals) * self.color))



  # traverse using uct to terminal or leaf node
  def traverse(self, en, color):
    e = en.copy()
    while s:=str(e._get_obs()) in MCTSBot.eval_dict and not e.is_full():
      self.update_list.append(s)
      self.step_best_UCT(e, color)
      color = -color

    self.update_list.append(s)
    return e, color

  # implement rollouts and backpropogate
  def rollout(self, env, color):
    if env.terminated:
      return env.last_reward

    action = np.random.choice(env.get_legal_moves())
    env.step(action, color)
    self.update_list.append(str(env._get_obs()))
    return self.rollout(env, -color)

    # may run into errors when run on terminal nodes that are unvisited?
    terminated = False
    while not terminated:
      action = env.action_space.sample()
      obs, reward, terminated, _, _ = env.step(action, color)
      self.update_list.append(str(env._get_obs()))
      color = -color
    return reward

  # step along based on UCT
  def step_best_UCT(self, e, color):
    best_move, best_val = -100,-100
    parent_vals = MCTSBot.eval_dict[str(e._get_obs())]
    moves = e.get_legal_moves()
    np.random.shuffle(moves)
    for move in moves:
      node = e.copy()
      node.step(move, color)
      if node_str:=str(node._get_obs()) not in MCTSBot.eval_dict:
        e.step(move, color) # we step and return, we've reached a leaf
        return
      else:
        node_vals = MCTSBot.eval_dict[node_str]
        wins1, visits = node_vals
        _, parent_visits = parent_vals

        wins = wins1 if color == 1 else visits - wins1
        UCT = wins/visits + np.sqrt(2 * np.log2(parent_visits)/visits)

        if UCT > best_val:
          best_val = UCT
          best_move = move
    e.step(best_move, color)





class Tournament:
  def __init__(self, bots: Bot, num_games, starting_position = None):
    self.bots = bots
    self.num_games = num_games
    self.env = Connect()
    self.tally = None

  def run_tournament(self):
    # head to head matrix of scores between bots
    # tally[i,j] is percent of times bot i beats bot j
    self.tally = np.zeros((len(self.bots), len(self.bots)))
    for n in tqdm(range(self.num_games)):
      print("\n",self.tally)
      for i in range(len(self.bots)):
        for j in range(len(self.bots)):
          if i != j:
            # play a game
            obs, info = self.env.reset()
            terminated = False
            color = 1

            # setup our bots
            self.bots[i].instantiate(1,self.env.action_space, self.env)
            self.bots[j].instantiate(-1,self.env.action_space, self.env)

            while not terminated:
              move = None
              if color == 1:
                move = self.bots[i].get_move(obs)
              else:
                move = self.bots[j].get_move(obs)
              obs, reward, terminated, _, _ = self.env.step(move, color)
              color *= -1
            if reward > 0:
              self.tally[i,j] += 1

    return (self.tally)