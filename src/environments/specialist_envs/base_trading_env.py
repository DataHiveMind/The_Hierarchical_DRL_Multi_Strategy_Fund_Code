import abc  # Abstract Base Class
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import scipy as sp

class BaseTradingEnv(gym.Env, abc.ABC):
    """
    An abstract base class for all specialist trading environments.

    This class provides the core structure for a trading environment, including
    data handling, step/reset logic, and portfolio value tracking.
    Specialist agents must inherit from this class and implement
    their own specific logic for:
    
    1. Defining `action_space` and `observation_space`
    2. `_get_obs()`
    3. `_take_action(action)`
    4. `_calculate_reward(action)`
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, 
                 df: pd.DataFrame, 
                 initial_balance: float = 1_000_000, 
                 transaction_cost_pct: float = 0.001):
        """
        Initializes the base trading environment.

        Args:
            df (pd.DataFrame): A DataFrame containing all market data (prices, features, etc.)
            initial_balance (float): The starting cash balance for the agent.
            transaction_cost_pct (float): The flat percentage cost for each trade (buy or sell).
        """
        super().__init__()
        
        # --- Core Data and State ---
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct

        # --- Spaces (MUST be defined in child class) ---
        # Example: self.action_space = spaces.Discrete(3) # Buy, Sell, Hold
        # Example: self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(1)  # Placeholder
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)  # Placeholder

        # --- Episode State Variables ---
        self._current_step = 0
        self._end_step = len(self.df) - 1
        
        self.balance = 0.0
        self.shares_held = 0.0  # Note: Can be generalized to a vector for multi-asset
        self.portfolio_value = 0.0
        self.total_pnl = 0.0
        
        self.render_mode = "human" # Default render mode

    @abc.abstractmethod
    def _get_obs(self):
        """
        Returns the current observation from the environment.
        This is what the agent "sees."
        
        MUST be implemented by the child class.
        
        Example:
        > return self.df.iloc[self._current_step][['price', 'volume', 'z_score']].values
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_info(self):
        """
        Returns a dictionary with auxiliary information.
        Not used for training, but useful for logging and debugging.

        MUST be implemented by the child class.

        Example:
        > return {
        >     "portfolio_value": self.portfolio_value,
        >     "pnl": self.total_pnl,
        >     "current_price": self.df.iloc[self._current_step]['price']
        > }
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _take_action(self, action):
        """
        Executes the given action (e.g., buy, sell, hold) and updates
        the environment state (balance, shares_held).
        This is where transaction costs are applied.

        MUST be implemented by the child class.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _calculate_reward(self, action):
        """
        Calculates the reward for the current step.
        This is the signal the agent learns from.

        MUST be implemented by the child class.
        
        Example:
        > # Simple P&L reward
        > prev_value = self.portfolio_value
        > self._update_portfolio_value() # Update based on new price
        > reward = self.portfolio_value - prev_value
        > return reward
        """
        raise NotImplementedError

    def _update_portfolio_value(self):
        """
        A helper function to update the portfolio value.
        This can be overridden by child classes (e.g., for multi-asset portfolios).
        """
        # This assumes a single asset for simplicity.
        # You'll need to get the "current price" from your self.df
        # current_price = self.df.iloc[self._current_step]['price']
        # self.portfolio_value = self.balance + (self.shares_held * current_price)
        pass # Implement this based on your data structure


    def step(self, action):
        """
        The main step function for the environment, following the gymnasium API.
        """
        # 1. Take the action (and apply transaction costs)
        self._take_action(action)
        
        # 2. Calculate the reward (e.g., P&L, Sharpe, etc.)
        reward = self._calculate_reward(action)
        
        # 3. Move to the next time step
        self._current_step += 1
        
        # 4. Update portfolio value based on new market data
        self._update_portfolio_value()
        self.total_pnl = self.portfolio_value - self.initial_balance

        # 5. Check for termination
        terminated = (self.portfolio_value <= 0) or (self._current_step >= self._end_step)
        truncated = False  # Not using time limits, termination is end of data
        
        # 6. Get new observation and info
        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        """
        super().reset(seed=seed, options=options)
        
        # Reset state variables
        self._current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.portfolio_value = self.initial_balance
        self.total_pnl = 0.0

        # Get initial observation and info
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info

    def render(self):
        """
        (Optional) Renders the environment.
        For trading, this usually means printing stats or plotting P&L.
        """
        if self.render_mode == "human":
            print(f"Step: {self._current_step}")
            print(f"Portfolio Value: {self.portfolio_value:,.2f}")
            print(f"Total P&L: {self.total_pnl:,.2f}")
            print(f"Shares Held: {self.shares_held}")
            print(f"Balance: {self.balance:,.2f}")

    def close(self):
        """
Section 2
(Optional) Perform any cleanup.
        """
        pass