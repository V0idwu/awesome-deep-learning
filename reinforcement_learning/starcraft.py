# %%
import os

from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["SC2PATH"] = "C:\Program Files (x86)\Blizzard Game\StarCraft II"


class WorkerRushBot(BotAI):
    async def on_step(self, iteration: int):
        if iteration == 0:
            for worker in self.workers:
                worker.attack(self.enemy_start_locations[0])


run_game(maps.get("AbyssalReefLE"), [Bot(Race.Zerg, WorkerRushBot()), Computer(Race.Protoss, Difficulty.Medium)], realtime=True)
