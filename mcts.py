#!/usr/bin/python3
# -*- coding: utf-8 -*-

from card_set import CardSet
from card_hand import CardHand
from card_hand import card_hand_compare

# from ddz_sample import VNSample
# from ddz_sample import PNSample

# from ddz_vn_eval import VNEval
from ddz_pn_eval import PNEval

from cheat import get_beat_hand
from cheat import get_simple_hand
from cheat import get_complete_hand

from functools import cmp_to_key

import sys
import math
import gflags

import numpy as np

FLAGS = gflags.FLAGS
gflags.DEFINE_integer('visit_count', 1000, 'visit count')
gflags.DEFINE_boolean('enable_vn', False, 'enable value network')
gflags.DEFINE_boolean('enable_pn', False, 'enable policy network')

# vn_eval = VNEval()
pn_eval = PNEval()

def select_action_by_rule(state):
    valid_hands = get_complete_hand(str(state.curr_cards)) if state.prev_hand.size() == 0 else get_beat_hand(str(state.prev_hand), str(state.curr_cards))
    valid_hands = valid_hands.split(" ")
    valid_hands = [CardSet.parse_from("") if valid_hand == "x" else CardSet.parse_from(valid_hand)
                   for valid_hand in valid_hands]

    # sorted by split card length
    left_cards = [CardSet.diff(state.curr_cards, valid_hand) for valid_hand in valid_hands]
    split_lens = [0 if left_card.size() == 0 else len(get_simple_hand(str(left_card)).split(" "))
                  for left_card in left_cards]

    hands_with_lens = list(zip(valid_hands, split_lens))
    min_split_len = min(hands_with_lens, key = lambda tpl: tpl[1])
    min_split_len_hands = [hand_with_len[0] for hand_with_len in hands_with_lens if hand_with_len[1] == min_split_len[1]]

    if len(min_split_len_hands) == 1:
        return min_split_len_hands[0]

    # sorted by left card length
    left_len = [(hand, 0 if CardSet.diff(state.curr_cards, hand).size() == 0 else len(str(CardSet.diff(state.curr_cards, hand))))
                for hand in min_split_len_hands]
    min_left_len = min(left_len, key = lambda tpl: tpl[1])
    min_left_len_hands = [hand for hand, length in left_len if length == min_left_len[1]]

    if len(min_left_len_hands) == 1:
        return min_left_len_hands[0]

    # sorted by hand rank
    card_hands = [CardHand.parse_from('' if str(hand) == 'x' else str(hand)) for hand in min_left_len_hands]
    sorted_by_rank = sorted(card_hands, key = cmp_to_key(card_hand_compare))
    return sorted_by_rank[0].card_set

def select_action_by_policy(state):
    valid_actions = state.valid_actions()
    action_probs = state.action_probs()

    candidates = sorted(zip(valid_actions, action_probs), key = lambda tpl: tpl[1], reverse = True)[0]
    return candidates[0]

class MCTSState(object):
    def __init__(self, player, curr_cards, next_cards, prev_hand, first_give):
        self.player = player
        self.curr_cards = curr_cards
        self.next_cards = next_cards
        self.prev_hand = prev_hand
        self.first_give = first_give

    def __str__(self):
        return ("dz" if self.player == 1 else "nm") + ": " + "-".join([
            str(self.curr_cards), str(self.next_cards),
            str(self.prev_hand),  str(self.first_give)
        ])

    def valid_actions(self):
        actions = get_complete_hand(str(self.curr_cards)) if self.prev_hand.size() == 0 else get_beat_hand(str(self.prev_hand), str(self.curr_cards))
        actions = actions.split(" ")
        actions = [CardSet.parse_from("") if action == "x" else CardSet.parse_from(action)
                   for action in actions]
        return actions

    def action_probs(self):
        return pn_eval.action_probs(self.curr_cards, self.next_cards, self.prev_hand)

    # def action_values(self):
    #     samples = [VNSample(
    #         self.curr_cards,
    #         self.next_cards,
    #         action,
    #         self.prev_hand,
    #         1 - self.player,
    #         self.first_give,
    #         0.0
    #     ) for action in self.valid_actions()]
    #
    #     matrices = [sample.to_matrix() for sample in samples]
    #     states = np.concatenate([matrix[0] for matrix in matrices])
    #     actions = np.concatenate([matrix[1] for matrix in matrices])
    #
    #     vn_scores = vn_eval.inference(states, actions)
    #     return vn_scores

C_PUCT = 30
LAMBDA = 0.5

class MCTSNode(object):
    def __init__(self, parent, state):
        self.parent = parent
        self.state = state
        self.children = []

        self.P = 0
        self.V = 0

        self.N_v = 0
        self.W_v = 0.0
        self.N_r = 0
        self.W_r = 0.0

    def Q(self):
        # value_part = 0.0 if self.N_v == 0 else self.W_v / self.N_v
        value_part = self.V
        rollout_part = 0.0 if self.N_r == 0 else self.W_r / self.N_r

        return (1 - LAMBDA) * value_part + LAMBDA * rollout_part

    def score(self):
        if self.parent is None:
            return 0.0

        Q_a = self.Q() if self.parent.state.player == 1 else -self.Q()
        score = Q_a + C_PUCT * self.P * math.sqrt(self.parent.N_r) / (1 + self.N_r)
        return score

    def __str__(self):
        return "|".join([
            str(self.state),
            "P: " + str(self.P),
            "V: " + str(self.V),
            "N_v: " + str(self.N_v),
            "W_v: " + str(self.W_v),
            "N_r: " + str(self.N_r),
            "W_r: " + str(self.W_r),
            "Q: " + str(self.Q()),
            "score: " + str(self.score())
        ])

    def select(self):
        child_scores = [child.score() for child in self.children]
        return np.argmax(child_scores)

    def _get_vn_scores(self):
        # if FLAGS.enable_vn:
        #     return self.state.action_values()
        # else:
            return [0.0 for child in self.children]

    def _get_pn_scores(self):
        # if FLAGS.enable_pn:
        #     return self.state.action_probs()
        # else:
            return [1.0 / len(self.children) for child in self.children]

    def has_children(self):
        return len(self.children) > 0

    def expand(self):
        if self.state.curr_cards.size() == 0 or self.state.next_cards.size() == 0:
            return False

        valid_hands = get_complete_hand(str(self.state.curr_cards)) if self.state.prev_hand.size() == 0 else get_beat_hand(str(self.state.prev_hand), str(self.state.curr_cards))
        valid_hands = [CardSet.parse_from("") if valid_hand == "x" else CardSet.parse_from(valid_hand)
                       for valid_hand in valid_hands.split(" ")]

        next_states = [MCTSState(
            1 - self.state.player,
            self.state.next_cards,
            CardSet.diff(self.state.curr_cards, valid_hand),
            valid_hand,
            False if valid_hand.nonempty() else True
        ) for valid_hand in valid_hands]

        self.children = [MCTSNode(self, next_state) for next_state in next_states]

        pn_scores = self._get_pn_scores()
        vn_scores = self._get_vn_scores()
        for child, pn_score, vn_score in zip(self.children, pn_scores, vn_scores):
            child.P = pn_score
            child.V = vn_score

        return True

    @staticmethod
    def rollout(state):
        # valid_hand = select_action_by_rule(state)
        valid_hand = select_action_by_policy(state)
        # print "%s: %s" % ("dz" if state.player == 1 else "nm", str(valid_hand))

        left_cards = CardSet.diff(state.curr_cards, valid_hand)
        if left_cards.size() == 0:
            # print "%s win" % ("dz" if state.player == 1 else "nm")
            return 1.0 if state.player == 1 else -1.0

        next_state = MCTSState(
            1 - state.player,
            state.next_cards,
            left_cards,
            valid_hand,
            False if state.prev_hand.nonempty() else True
        )

        return MCTSNode.rollout(next_state)

    def backup(self, V, Z):
        self.N_r += 1
        self.W_r += Z

        self.N_v += 1
        self.W_v += V

        if self.parent != None:
            MCTSNode.backup(self.parent, V, Z)

class MCTS(object):
    def __init__(self, root_node):
        self.root_node = root_node
        self.curr_node = self.root_node
        self.is_terminal = False

    def reset(self):
        self.curr_node = self.root_node

    def move(self, action):
        if not self.curr_node.has_children():
            self.curr_node.expand()

        for child in self.curr_node.children:
            if child.state.prev_hand == action:
                self.curr_node = child
                break

        return '-'.join([
            str(self.curr_node.state.player),
            str(self.curr_node.state.curr_cards),
            str(self.curr_node.state.next_cards),
            str(self.curr_node.state.prev_hand)
        ])

    def tree_search(self):
        node = self.curr_node
        while node.has_children():
            selected = node.select()
            node = node.children[selected]

        if node.expand():
            selected = node.select()
            node = node.children[selected]
            if node.state.curr_cards.size() == 0:
                Z = 1.0 if node.state.player == 1 else -1.0
            elif node.state.next_cards.size() == 0:
                Z = -1.0 if node.state.player == 1 else 1.0
            else:
                Z = MCTSNode.rollout(node.state)
        else:
            if node.state.curr_cards.size() == 0:
                Z = 1.0 if node.state.player == 1 else -1.0
            elif node.state.next_cards.size() == 0:
                Z = -1.0 if node.state.player == 1 else 1.0
            else:
                Z = MCTSNode.rollout(node.state)
        node.backup(node.V, Z)

    def best_action(self):
        for i in range(FLAGS.visit_count):
            self.tree_search()

        best = np.argmax([child.Q() for child in self.curr_node.children])
        return self.curr_node.children[best].state.prev_hand

    def info(self):
        result = ""

        result += "%s\n" % str(self.curr_node)
        for child in self.curr_node.children:
            result += "%s\n" % str(child)
        return result
