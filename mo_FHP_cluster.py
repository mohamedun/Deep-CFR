from PokerRL.eval.head_to_head.H2HArgs import H2HArgs
from PokerRL.eval.rl_br.RLBRArgs import RLBRArgs
from PokerRL.game.games import Flop5Holdem
from PokerRL.game.bet_sets import POT_ONLY
from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

if __name__ == '__main__':
    """
    Runs FHP with the same parameters as original but only 20 workers
    """
    ctrl = Driver(t_prof=TrainingProfile(name="MO_FHP_cluster",
                                         redis_head_adr="auto",

                                         nn_type="feedforward",  # We also support RNNs, but the paper uses FF

                                         DISTRIBUTED=True,
                                         CLUSTER=True,
                                         n_learner_actor_workers=20,  # 20 workers

                                         # regulate exports
                                         export_each_net=False,
                                         checkpoint_freq=99999999,
                                         eval_agent_export_freq=10,  # produces around 15GB over 150 iterations!

                                         n_actions_traverser_samples=3,  # = external sampling in FHP
                                         n_traversals_per_iter=15000,
                                         n_batches_adv_training=4000,
                                         mini_batch_size_adv=512,  # *20=10240
                                         init_adv_model="random",

                                         use_pre_layers_adv=True,
                                         n_cards_state_units_adv=192,
                                         n_merge_and_table_layer_units_adv=64,
                                         n_units_final_adv=64,

                                         max_buffer_size_adv=2e6,  # *20 LAs = 40M
                                         lr_adv=0.001,
                                         lr_patience_adv=99999999,  # No lr decay

                                         n_batches_avrg_training=20000,
                                         mini_batch_size_avrg=1024,  # *20=20480
                                         init_avrg_model="random",

                                         use_pre_layers_avrg=True,
                                         n_cards_state_units_avrg=192,
                                         n_merge_and_table_layer_units_avrg=64,
                                         n_units_final_avrg=64,

                                         max_buffer_size_avrg=2e6,
                                         lr_avrg=0.001,
                                         lr_patience_avrg=99999999,  # No lr decay

                                         # With the H2H evaluator, these two are evaluated against eachother.
                                         eval_modes_of_algo=(
                                             EvalAgentDeepCFR.EVAL_MODE_AVRG_NET,
                                         ),

                                         log_verbose=True,
                                         game_cls=Flop5Holdem,

                                         # enables simplified obs. Default works also for 3+ players
                                         use_simplified_headsup_obs=True,

                                         rl_br_args=RLBRArgs(rlbr_bet_set=POT_ONLY),
                                         ),
                  # Evaluate Head-to-Head every 15 iterations of both players (= every 30 alternating iterations)
                  eval_methods={},

                  # 150 = 300 when 2 viewing alternating iterations as 2 (as usually done).
                  # This repo implements alternating iters as a single iter, which is why this says 150.
                  n_iterations=10,
                  )
    ctrl.run()
