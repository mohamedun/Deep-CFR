from PokerRL.eval.head_to_head.H2HArgs import H2HArgs
from PokerRL.game.games import Flop5Holdem
from PokerRL.game.games import LimitHoldem

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

if __name__ == '__main__':
    """
    Runs HULH training for the same parameters of FLH with only 5 workers.
    """
    ctrl = Driver(t_prof=TrainingProfile(name="MO_HULH_1",

                                         nn_type="feedforward",  # We also support RNNs, but the paper uses FF

                                         DISTRIBUTED=True,
                                         CLUSTER=False,
                                         n_learner_actor_workers=40,  # 20 workers

                                         # regulate exports
                                         export_each_net=False,
                                         checkpoint_freq=99999999,
                                         eval_agent_export_freq=5,

                                         n_actions_traverser_samples=3,  # = external sampling in FHP
                                         n_traversals_per_iter=500,
                                         n_batches_adv_training=2000,
                                         mini_batch_size_adv=512,  # *20=10240
                                         init_adv_model="random",

                                         use_pre_layers_adv=True,
                                         n_cards_state_units_adv=192,
                                         n_merge_and_table_layer_units_adv=64,
                                         n_units_final_adv=64,

                                         max_buffer_size_adv=2e6,  # *20 LAs = 40M
                                         lr_adv=0.001,
                                         lr_patience_adv=99999999,  # No lr decay

                                         # With the H2H evaluator, these two are evaluated against eachother.
                                         eval_modes_of_algo=(
                                            EvalAgentDeepCFR.EVAL_MODE_SINGLE,
                                         ),

                                         log_verbose=True,
                                         game_cls=LimitHoldem,

                                         # enables simplified obs. Default works also for 3+ players
                                         use_simplified_headsup_obs=True,
                                         ),
                  eval_methods={},


                  n_iterations=50,
                  )
    ctrl.run()
