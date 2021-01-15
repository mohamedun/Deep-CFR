from PokerRL.game.games import StandardLeduc

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

if __name__ == '__main__':
    ctrl = Driver(t_prof=TrainingProfile(name="MO_LEDUC_EXPLOITABILITY",
                                         nn_type="feedforward",

                                         DISTRIBUTED=True,
                                         log_verbose=True,
                                         n_learner_actor_workers=5,

                                         eval_agent_export_freq=3,
                                         checkpoint_freq=100,

                                         max_buffer_size_adv=1e6,
                                         n_traversals_per_iter=1500,
                                         n_batches_adv_training=750,
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

e,
                                         ),
                  eval_methods={
                  },
                  n_iterations=4)
    ctrl.run()
