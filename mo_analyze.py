# Copyright (c) 2019 Eric Steinberger

import pdb
import time
from os.path import dirname, abspath

import numpy as np

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR

# These two eval agents HAVE TO come from the same training run and iteration for this analysis to make sense.
path_to_dcfr_eval_agent = dirname(abspath(__file__)) + "/trained_agents/Example_FHP_AVRG_NET.pkl"
#path_to_sdcfr_eval_agent = dirname(abspath(__file__)) + "/trained_agents/Example_FHP_AVRG_NET.pkl"
#path_to_sdcfr_eval_agent = dirname(abspath(__file__)) + "/trained_agents/Example_FHP_SINGLE.pkl"

N_HOLE = 13 * 4 * 13 * 4 / 2

if __name__ == '__main__':
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
        for p in env.seats:
            if p.seat_id == 0:
                hole_hand = env.cards2str(p.hand)
                if hole_hand not in hands:
                    hands[hole_hand] = eval_agent_dcfr.get_a_probs()

print(f"Computed {len(N_HOLE)} in {time.time()-start_time} sec")
for hand in hands.keys():
    print(f"for hand: {hand}, the probabilities are {hands[hand]}")
