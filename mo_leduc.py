from PokerRL.game.games import StandardLeduc
from PokerRL.eval.rl_br.RLBRArgs import RLBRArgs
from PokerRL.eval.lbr.LBRArgs import LBRArgs
from PokerRL.game.bet_sets import POT_ONLY
from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver
import pdb
if __name__ == '__main__':
    ctrl = Driver(t_prof=TrainingProfile(name="MO_LEDUC_EVAL",
                                         nn_type="feedforward",

                                         eval_agent_export_freq=3,
                                         checkpoint_freq=3,

                                         max_buffer_size_adv=1e6,
                                         n_traversals_per_iter=500,
                                         n_batches_adv_training=250,
                                         mini_batch_size_adv=2048,

                                         game_cls=StandardLeduc,

                                         n_units_final_adv=64,
                                         n_merge_and_table_layer_units_adv=64,
                                         init_adv_model="random",  # warm start neural weights with init from last iter
                                         use_pre_layers_adv=False,  # shallower nets
                                         use_pre_layers_avrg=False,  # shallower nets

                                         # You can specify one or both modes. Choosing both is useful to compare them.
                                         eval_modes_of_algo=(
                                             EvalAgentDeepCFR.EVAL_MODE_SINGLE,  # SD-CFR
                                         ),

                                         DISTRIBUTED=False,
                                         log_verbose=True,
                                         rl_br_args=RLBRArgs(rlbr_bet_set=None,
                                                             n_hands_each_seat=200,
                                                             n_workers=1,
                                                             # Training
                                                             DISTRIBUTED=False,
                                                             n_iterations=100,
                                                             play_n_games_per_iter=50,
                                                             # The DDQN
                                                             batch_size=512,
                                                             ),
                                         lbr_args=LBRArgs(n_lbr_hands_per_seat=30000,
                                                          n_parallel_lbr_workers=10,
                                                          DISTRIBUTED=False,
                                                          ),
                                         ),
                  eval_methods={'br': 1,
                                #'rlbr': 1,
                                'lbr': 1,
                  },
                  n_iterations=12)
    ctrl.run()
    pdb.set_trace()