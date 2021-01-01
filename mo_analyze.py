# Copyright (c) 2019 Eric Steinberger

import pdb
import time
from os.path import dirname, abspath
import numpy as np

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR

# These two eval agents HAVE TO come from the same training run and iteration for this analysis to make sense.
path_to_dcfr_eval_agent = dirname(abspath(__file__)) + "/trained_agents/Example_FHP_AVRG_NET.pkl"

N_DECK = 52
N_HOLE = 169 # 13 * 12 + 13

def hand2rep(hand):
    card1_rank = hand[0][0]
    card1_suit = hand[0][1]
    card2_rank = hand[1][0]
    card2_suit = hand[1][1]
    suited = (card2_suit == card1_suit)
    high_rank = max(card1_rank, card2_rank)
    low_rank = min(card1_rank, card2_rank)
    return (high_rank, low_rank, suited)

#--------------- Generate p0 strat -------------------------

#Loading EvalAgents and checking if hey have same experiment name
eval_agent_dcfr = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=path_to_dcfr_eval_agent)

#get an env bldr from the agent and create an env
env_bldr = eval_agent_dcfr.env_bldr
env = env_bldr.get_new_env(is_evaluating=False)

start_time = time.time()
hands = {}
while len(hands) < N_HOLE:
    obs, rew, done, info = env.reset()
    eval_agent_dcfr.reset(deck_state_dict=env.cards_state_dict())
    hole_hand = hand2rep(env.seats[0].hand)
    if hole_hand not in hands:
        hands[hole_hand] = eval_agent_dcfr.get_a_probs()

print(f"Computed {N_HOLE} possible hands in {time.time()-start_time} sec")
for hand in hands.keys():
    print(f"for hand: {hand}, the probabilities are {hands[hand]}")

#----------------------------store data for p0
import pickle
f = open('p0_strat.pkl', 'ab')
pickle.dump(hands, f)
f.close()

#----------------------- Generate Image for p0
import plotting
plotting.np2img(hands)

pdb.set_trace()