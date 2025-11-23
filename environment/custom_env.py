_GYM_BACKEND = None
try:
    # Prefer the maintained fork
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_BACKEND = "gymnasium"
except Exception:
    try:
        import gym
        from gym import spaces
        _GYM_BACKEND = "gym"
    except Exception as e:
        # Provide a lightweight fallback so the environment can run without
        # requiring gym/gymnasium to be installed. This implements only the
        # minimal API used by this project: an `Env` base class and
        # `spaces.Box` / `spaces.Discrete` with a `sample()` method.
        from types import SimpleNamespace

        class _FallbackEnv:
            pass

        class _FallbackBox:
            def __init__(self, low, high, dtype=np.float32):
                self.low = np.array(low, dtype=dtype)
                self.high = np.array(high, dtype=dtype)
                self.dtype = dtype
                try:
                    self.shape = self.low.shape
                except Exception:
                    self.shape = None

            def sample(self):
                return np.random.uniform(self.low, self.high).astype(self.dtype)

        class _FallbackDiscrete:
            def __init__(self, n):
                self.n = int(n)

            def sample(self):
                return int(np.random.randint(0, self.n))

        spaces = SimpleNamespace(Box=_FallbackBox, Discrete=_FallbackDiscrete)
        gym = SimpleNamespace(Env=_FallbackEnv)
        _GYM_BACKEND = "fallback"

import numpy as np

class FraudEnv(gym.Env):
    """Custom environment for real-time financial transaction fraud detection.

    State: [transaction_amount, hour_of_day, location_code, previous_risk_score]
    Action: 0 = Legitimate, 1 = Fraud
    Reward: +1 for correct, -1 for wrong classification

    This environment follows the newer Gym API (reset -> (obs, info),
    step -> (obs, reward, terminated, truncated, info)).
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode=None):
        super(FraudEnv, self).__init__()
        # State: 4 features
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0], dtype=np.float32),
                                            high=np.array([10000, 23, 100, 1], dtype=np.float32),
                                            dtype=np.float32)
        # Action: fraud or legit
        self.action_space = spaces.Discrete(2)
        self.current_transaction = None
        self.render_mode = render_mode

    def step(self, action):
        if self.current_transaction is None:
            raise RuntimeError("Environment must be reset before calling step().")

        label = int(self.current_transaction[-1])  # last element = true label
        reward = 1 if int(action) == label else -1
        terminated = True  # each transaction is a 1-step episode
        truncated = False

        info = {"transaction": self.current_transaction.copy()}

        # Prepare next transaction for the following episode and return its observation
        next_transaction = self._get_transaction()
        self.current_transaction = next_transaction
        next_obs = next_transaction[:-1]

        return next_obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        # generate the first transaction
        self.current_transaction = self._get_transaction()
        obs = self.current_transaction[:-1]
        info = {}
        return obs, info

    def _get_transaction(self):
        """Random transaction generator:
        amount, hour, location, risk_score, label
        Fraud label = 1, Legit = 0
        """
        amount = np.random.uniform(1, 10000)
        hour = np.random.randint(0, 24)
        location = np.random.randint(0, 100)
        risk_score = np.random.rand()
        # Simple fraud labeling rule
        label = 1 if (amount > 3000 and risk_score > 0.5) else 0
        return np.array([amount, hour, location, risk_score, label], dtype=np.float32)

    def render(self, mode='human'):
        if self.current_transaction is None:
            print("No current transaction. Call reset() first.")
            return
        print(f"Transaction: {self.current_transaction[:-1]}, True Label: {int(self.current_transaction[-1])}")

    def close(self):
        # nothing to cleanup for this simple env, but implement for API compatibility
        return


# Backwards compatible name
FraudDetectionEnv = FraudEnv
