import numpy as np
import enum
import pickle
from PIL import Image

from wrappers import VideoWriter, Monitor, StochasticGolds, ActionGather

class Direction(enum.IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Action(enum.IntEnum):
    UP = Direction.UP
    RIGHT = Direction.RIGHT
    DOWN = Direction.DOWN
    LEFT = Direction.LEFT
    GATHER = ActionGather.ACTION_GATHER

class Cell(enum.IntEnum):
    EMPTY = 0
    WALL = 1

def direction_to_dxy(direction):
    """Return the corresponding dy,dx tuple. The y-axis points down, the x-axis points right (as the conventions in pyplot.imshow).

    Args:
        direction (Direction): a direction

    Returns:
        tuple: dy,dx
    
    >>> direction_to_dxy(Direction.UP)
    (0,-1)
    """
    if direction == Direction.UP:
        dx, dy = 0, -1
    elif direction == Direction.RIGHT:
        dx, dy = 1, 0
    elif direction == Direction.DOWN:
        dx, dy = 0, 1
    elif direction == Direction.LEFT:
        dx, dy = -1, 0
    else:
        raise ValueError("Wrong direction value '%d'" %(direction))
    return dy,dx

def dxy_to_direction(dy, dx):
    """Return the corresponding direction. See direction_to_dxy.

    Args:
        dy (float): delta on the y-axis
        dx (float): delta in the x-axis

    Returns:
        Direction: direction
    
    >>> dxy_to_direction(dy=-1, dx=0)
    Direction.UP
    """
    assert ((dy == 0) + (dx == 0)) == 1, (dy,dx)
    if dy == 0:
        if dx > 0:
            return Direction.RIGHT
        elif dx < 0:
            return Direction.LEFT
    elif dy > 0:
        return Direction.DOWN
    else:
        return Direction.UP

class Maze:
    """
    Holds a 2D maze where cells can be either empty or walls. Some empty cells may contain gold.
    
    For the exercise, this class sould not be directly instantiated by the user,
    but instead it should come wrapped, cf create_maze.
    Otherwise A basic usage is as follows:
    >>> obs = maze.reset()
    ... while True:
    ...   action = np.random.choice(Direction)
    ...   obs, gold, done, debug_info = maze.step(action)
    ...   if done:
    ...     break
    """
    
    NEIGHBOOR_ADD_COLOR = 50

    # for Maze.render, scalars must be at most (255 - NEIGHBOOR_ADD_COLOR)
    cell_to_color = {
        Cell.EMPTY: (190, 190, 190),
        Cell.WALL: (60, 60, 60),
    }

    @staticmethod
    def load_maze(filepath):
        """Loads a maze from a previously saved configuration, cf Maze.save_maze

        Args:
            filepath (str): Full path to a previously saved maze, eg "./maze_1.pkl"
        """
        return Maze(**pickle.load(open(filepath, "rb")))

    def __init__(self, layout, golds, dones, starts):
        """Instantiate a Maze

        Args:
            layout (np.ndarray of Cell): Describes the locations of walls
            golds (np.ndarray of float): Same shape as layout, this describes the locations and quantities of gold
            dones (np.ndarray of bool): Same shape as layout, this describes the possible ending locations
            starts (np.ndarray of bool): Same shape as layout, this describes the possible starting locations
        """
        shape = layout.shape
        assert shape == golds.shape, (shape, golds.shape)
        assert shape == dones.shape, (shape, dones.shape)
        assert shape == starts.shape, (shape, starts.shape)

        self.layout = layout.clip(min=0, max=len(Cell)).astype(int)
        self.layout[:,0] = Cell.WALL
        self.layout[:,-1] = Cell.WALL
        self.layout[0,:] = Cell.WALL
        self.layout[-1,:] = Cell.WALL

        self.golds = golds.astype(float)
        self.dones = dones.astype(bool)
        self.starts = starts.astype(bool)
        assert (self.dones + self.starts <= 1).all(), "A starting cell cannot also be a terminal cell"

        self.maze_frame = None
        self.must_be_reset = True
        self.render(just_init=True)

    def _get_observation(self, y, x):
        """Returns the observation of the current state.

        Args:
            y (int): y absolute position
            x (int): x absolute position

        Returns:
            tuple(int,int,Cell,Cell,Cell,Cell): (y,x,top,left,right,bottom)
        """
        neighbours = self.layout[y-1:y+2, x-1:x+2].flatten().astype(int)
        neighbours = [neighbours[1], neighbours[3], neighbours[5], neighbours[7]] # top, left, right, bottom
        obs = np.array([y, x] + neighbours, dtype=int) # x,y,top,left,right,bottom
        return obs
    
    def reset(self):
        """Resets the location of the player to one possible start. This should be called after self.__init__ and after everytime self.step returns done==True

        Returns:
            tuple: observation of the current state, see self._get_observation
        """
        self.must_be_reset = False
        starts = np.argwhere(self.starts)
        idx = np.random.randint(0,len(starts))
        self.y,self.x = starts[idx]
        self.obs = self._get_observation(self.y, self.x)
        return self.obs
    
    def step(self, direction, **kwargs):
        """Move one step in one direction, and compute the resulting state (eg if there is a wall in the given direction, the state is unchanged).
        This should be called after self.__init__ and before self.reset, see self.reset.

        Args:
            direction (Direction): direction to move to

        Returns:
            tuple(tuple, float, bool, dict): obs (observation of the current state, see self._get_observation), gold, done, info
        """
        assert 0 <= direction < len(Direction), direction
        if self.must_be_reset:
            raise RuntimeError("Maze must be .reset() after initialization and once done")
        dy, dx = direction_to_dxy(direction)
        y = self.y + dy
        x = self.x + dx
        if self.layout[y,x] == Cell.EMPTY: # indeed, we cannot move into walls
            self.y = y
            self.x = x
        gold = self.golds[self.y, self.x]
        done = self.dones[self.y, self.x]
        debug_info = {
            'y': self.y,
            'x': self.x,
        }
        if done:
            self.must_be_reset = True
        self.obs = self._get_observation(self.y, self.x)
        return self.obs, gold, done, debug_info
    
    def render(self, just_init=False, render_infos=None):
        """This is a logging function: this returns an image of the current state of the maze (with location of the player).

        Args:
            just_init (bool, optional): Initialize a view of the maze. This should be True only once (when called from self.__init__) and kept as False otherwise. Defaults to False.
            render_infos: list of tuples (pixel position, color) which can be passed by client code. These colors will overwrite the default pixel color. If set to None (default value), nothing is done.

        Returns:
            np.ndarray of np.uint8: image of the current state of the maze
        """
        if self.maze_frame is None: # draw the maze's layout only once
            offset1 = 150
            offset2 = 50
            self.maze_frame = np.array([[Maze.cell_to_color[cell] for cell in row] for row in self.layout]).astype(np.uint8)
            unique_golds = list(np.unique(self.golds))
            unique_golds.remove(0)
            minn = np.min(unique_golds)
            maxx = np.max(unique_golds)
            for y,x in np.argwhere(self.golds > 0):
                val = (self.golds[y,x] - minn)/(maxx - minn + 1e-6)
                self.maze_frame[y,x,:] = np.array([0,offset2,0]) + val*offset1*np.array([1,1,1])
            for y,x in np.argwhere(self.golds < 0):
                val = (self.golds[y,x] - minn)/(maxx - minn + 1e-6)
                self.maze_frame[y,x,:] = np.array([offset1+offset2,offset1,offset1]) - val*offset1*np.array([1,1,1])
        if just_init or self.must_be_reset:
            return self.maze_frame
        frame = self.maze_frame.copy()

        if render_infos is not None:
            h,w = self.layout.shape
            for render_info in render_infos:
                pixel_location, pixel_color = render_info
                #print(pixel_location, pixel_color)
                if 0 <= pixel_location[0] < h and 0 <= pixel_location[1] < w:
                    frame[pixel_location[0], pixel_location[1]] = list(pixel_color)

        # player's current fog of war
        frame[self.y-1:self.y+2, self.x-1:self.x+2] = self.maze_frame[self.y-1:self.y+2, self.x-1:self.x+2] + Maze.NEIGHBOOR_ADD_COLOR
        # player's current location
        frame[self.y,self.x,:] = [0,0,200]
        
        return frame
    
    def close(self):
        pass

    def save_maze(self, filename):
        """Save the configuration of this maze (layout, golds...). Cf Maze.load_maze

        Args:
            filename (str): A file path without extension eg "./maze_1"
        """
        data = {
            "layout": self.layout,
            "golds": self.golds,
            "dones": self.dones,
            "starts": self.starts,
        }
        pickle.dump(data, open(filename + ".pkl", "wb"))
        img = Image.fromarray(self.maze_frame)
        img.save(filename + ".png")

def procedural_maze(h,w,ngolds):
    def spawn_golds(maze, golds, ngolds, starts):
        h,w = golds.shape
        ck,cp = 1,15
        def dist2_to_gold(dist2):
            gold = dist2
            gold = np.sqrt(gold)
            gold = - ck*gold + cp
            return gold
        cy = np.random.randint(0,h)
        cx = np.random.randint(0,w)
        ntries = 500
        itry = 0
        igold = 0
        while (itry < ntries) and (igold < ngolds):
            y = np.random.randint(0,int((h-1)/2)) * 2 + 1
            x = np.random.randint(0,int((w-1)/2)) * 2 + 1
            if (golds[y,x] == 0) and (maze[y,x] == Cell.EMPTY) and (not starts[y,x]):
                dist2 = (cy-y)**2 + (cx-x)**2
                gold = dist2_to_gold(dist2)
                if gold > 0:
                    golds[y,x] = gold
                    igold += 1
            itry += 1

    assert (h % 2 == 1) and (w % 2 == 1), (h,w)
    layout = np.ones((h,w), dtype=int) * Cell.EMPTY
    layout[0,:] = Cell.WALL
    layout[-1,:] = Cell.WALL
    layout[:,0] = Cell.WALL
    layout[:,-1] = Cell.WALL
    layout[::2,::2] = Cell.WALL
    golds = np.zeros_like(layout).astype(float)
    dones = np.zeros(golds.shape)
    starts = np.zeros(golds.shape)
    starts[3,3] = 1
    spawn_golds(layout, golds, ngolds, starts)
    i_walls = 0
    n_walls = int(h*w/20)
    while i_walls < n_walls:
        y = np.random.randint(1,h-1)
        x = np.random.randint(0,int((w-1)/2)) * 2
        if y % 2 == 0:
            x += 1
        if layout[y,x] != Cell.WALL:
            layout[y,x] = Cell.WALL
            i_walls += 1
    env = Maze(layout, golds, dones, starts)
    return env

def create_maze(video_prefix="./vid", fps=4, overwrite_every_episode=False, save_video=True):
    """Generate a wrapped maze.

    Args:
        video_prefix (str): see documentation of VideoWriter
        fps (int): see documentation of VideoWriter
        overwrite_every_episode (bool): see documentation of VideoWriter

    Returns:
        A wrapped maze.
    """
    sizes = [13,21,29]
    size = np.random.choice(sizes)
    ngolds = 8
    env = maze = procedural_maze(size, size, ngolds)

    # wrap
    env = ActionGather(env, maze) # modifies several behaviors of the underlying env
    env = StochasticGolds(env, maze, std=0.7) # golds are made stochastic
    env = Monitor(env) # return additional debug information
    if save_video:
        env = VideoWriter(env, video_prefix, fps=fps, overwrite_every_episode=overwrite_every_episode) # record a video of the player's actions
    return env
