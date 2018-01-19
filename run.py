from roomai.sevenking import *
import random
import roomai
import roomai.RoomAILogger

class KuhnPokerExamplePlayer(roomai.common.AbstractPlayer):
    # @override
    def receive_info(self, info):
        if info.person_state.available_actions is not None:
            self.available_actions = info.person_state.available_actions

    # @override
    def take_action(self):
        return self.available_actions.values()[int(random.random() * len(self.available_actions))]

    # @overide
    def reset(self):
        pass


if __name__ == "__main__":
    players = [KuhnPokerExamplePlayer() for i in range(2)]
    env = SevenKingEnv()
    scores = SevenKingEnv.compete(env, players)
    print(scores)