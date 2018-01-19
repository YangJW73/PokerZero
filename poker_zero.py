from self_play import SelfPlay
from neural_network import NeuralNetwork

class PokerZero(object):
    """
    class implement poker zero
    """
    def __init__(self, max_step):
        # initialize neural network weights
        self.model = NeuralNetwork('resnet')

        # initialize iteration num i
        self.iteration_num = 1000

        # initialize termination step T
        self.T = max_step
        self.enhance_op = SelfPlay(max_step=max_step)

    def train(self):
        for i in range(self.iteration_num):
            # initialize state s0
            s0 = None # RoomAi generate one !!!
            # conduct self-play
            self.enhance_op.run_selfplay(cur_state=s0)
            """
            get the final score game r_T here
            """

            """
            label data z_t = +-r_T
            store data (s_t, \pi_t, z_t)
            """
            samples = self.enhance_op.save_data()
            """
            sample from all time step of last iteration of self-play
            """
            gen_samples = self.enhance_op.gen_samples()
            """
            train neural network using these samples
            """
            self.model.build_model(gen_samples)
