

class SimEnv:
    """Simulation environment"""

    def __init__(self, prices, weather, agent):
        """
        :param DataFrame prices: 15m electricity prices. DatetimeIndex.
        :param DataFrame weather: hourly weather. DatetimeIndex.
        :param Agent agent: RL agent.
        """

        self.prices = prices
        self.weather = weather
        self.agent = agent
