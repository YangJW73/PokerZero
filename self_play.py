"""
applying mcts for self play to generate training data
"""
import numpy as np
from mcts import MCTS
class SelfPlay(object):
    """
    class to implement Self Play part
    """
    def __init__(self, max_step):
        """
        initialize parameters for self play
        """
        self.T = max_step
        self.state_list = []
        self.action_list = []
        self.result = 0

    def run_selfplay(self, cur_state):
        """

        :return:
        """
        self.mcts = MCTS(cur_state)
        for t in range(self.T):
            self.mcts.tree_search()
            select_action = self.mcts.best_action()
            self.state_list.append(cur_state)
            self.action_list.append(select_action)
            self.mcts.cur_state = self.mcts.move(select_action)

    def save_data(self):
        """

        :return:
        """
        for t in range(len(self.state_list)):
            if t == 0:
                s = self.state_list[t]
                a = self.action_list[t]
                r = self.result
            else:
                s_ = self.state_list[t]
                a_ = self.action_list[t]
                r_ = self.result
                s = np.concatenate((s, s_))
                a = np.concatenate((a, a_))
                r = np.concatenate((r, r_))
        return s, a, r

    def gen_samples(self):
        pass
