from PokerRL.game.games import StandardLeduc

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

if __name__ == '__main__':
    ctrl = Driver(t_prof=TrainingProfile(name="MO_LEDUC_EXPLOITABILITY",
                                         nn_type="feedforward",
                                         max_buffer_size_adv=1e6,
                                         max_buffer_size_avrg=1e6,
                                         eval_agent_export_freq=3,
                                         checkpoint_freq=1,
                                         n_traversals_per_iter=1500,
                                         n_batches_adv_training=750,
                                         n_batches_avrg_training=5000,
                                         n_merge_and_table_layer_units_adv=64,
                                         n_merge_and_table_layer_units_avrg=64,
                                         n_units_final_adv=64,
                                         n_units_final_avrg=64,
                                         mini_batch_size_adv=2048,
                                         mini_batch_size_avrg=2048,
                                         init_adv_model="last",  # warm start neural weights with init from last iter
                                         init_avrg_model="random",
                                         use_pre_layers_adv=False,  # shallower nets
                                         use_pre_layers_avrg=False,  # shallower nets

                                         game_cls=StandardLeduc,

                                         # You can specify one or both modes. Choosing both is useful to compare them.
                                         eval_modes_of_algo=(
                                             EvalAgentDeepCFR.EVAL_MODE_SINGLE,  # SD-CFR
                                             EvalAgentDeepCFR.EVAL_MODE_AVRG_NET,  # Deep CFR
                                         ),

                                         DISTRIBUTED=False,
                                         log_verbose=False,
                                         ),
                  eval_methods={
                      "br": 10,
                  },
                  n_iterations=3)
    ctrl.run()
    ctlr.run()
