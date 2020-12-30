# Copyright (c) 2019 Eric Steinberger


import time
from os.path import dirname, abspath

import numpy as np

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR

# These two eval agents HAVE TO come from the same training run and iteration for this analysis to make sense.
path_to_dcfr_eval_agent = dirname(abspath(__file__)) + "/trained_agents/Example_FHP_AVRG_NET.pkl"
path_to_sdcfr_eval_agent = dirname(abspath(__file__)) + "/trained_agents/Example_FHP_SINGLE.pkl"

N_GAMES = 20
MAX_DEPTH = 6  # This is a constant for FHP

if __name__ == '__main__':
    eval_agent_dcfr = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=path_to_dcfr_eval_agent)
    eval_agent_sdcfr = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=path_to_sdcfr_eval_agent)
    assert eval_agent_dcfr.t_prof.name == eval_agent_sdcfr.t_prof.name

    env_bldr = eval_agent_dcfr.env_bldr
    env = env_bldr.get_new_env(is_evaluating=False)

    start_time = time.time()

    for sample in range(1, N_GAMES + 1):
        for p_eval in range(2):
            obs, rew, done, info = env.reset()


            eval_agent_sdcfr.reset(deck_state_dict=env.cards_state_dict())
            eval_agent_dcfr.reset(deck_state_dict=env.cards_state_dict())

            while not done:
                p_id_acting = env.current_player.seat_id
                legal_actions_list = env.get_legal_actions()

                # Step according to agent strategy
                if p_id_acting == p_eval:

                    # Compare Action-probability distribution
                    a_dist_sdcfr = eval_agent_sdcfr.get_a_probs()
                    a_dist_dcfr = eval_agent_dcfr.get_a_probs()
                    a = eval_agent_sdcfr.get_action(step_env=False)[0]

                # Handle env stepping along random trajectories
                else:
                    a = legal_actions_list[np.random.randint(len(legal_actions_list))]

                obs, rew, done, info = env.step(a)
                eval_agent_sdcfr.notify_of_action(p_id_acted=p_id_acting, action_he_did=a)
                eval_agent_dcfr.notify_of_action(p_id_acted=p_id_acting, action_he_did=a)
                breakpoint()

