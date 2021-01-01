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

N_GAMES = 20

if __name__ == '__main__':
    #Loading EvalAgents and checking if hey have same experiment name
    eval_agent_dcfr = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=path_to_dcfr_eval_agent)

    #get an env bldr from the agent and create an env
    env_bldr = eval_agent_dcfr.env_bldr
    env = env_bldr.get_new_env(is_evaluating=False)

    start_time = time.time()

    for sample in range(1, N_GAMES + 1):
        #reset the environment
        obs, rew, done, info = env.reset()
        for p in env.seats:
            if p.seat_id == 0 :
                print(env.cards2str(p.hand))
        #get state from the env and reset the agents according to state
        eval_agent_dcfr.reset(deck_state_dict=env.cards_state_dict())

        legal_actions_list = env.get_legal_actions()
        a_dist_dcfr = eval_agent_dcfr.get_a_probs()
        #pdb.set_trace()