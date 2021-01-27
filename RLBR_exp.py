import logs_util
import pdb
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

n_iter = 3
driver_iterations = 12
hp_measure = {}
exp_name = "MO_LEDUC_EVAL"
TP = TrainingProfile(name=exp_name,
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
                                         )
                     #lbr_args = LBRArgs()
                     )


for i in range(n_iter):
    TP.rl_br_args.n_hands_each_seat = 5 * i
    ctrl = Driver(t_prof=TP, eval_methods={'br': 1, 'rlbr': 1}, n_iterations=driver_iterations)
    ctrl.run()
    dfs = logs_util.logs_to_dfs(exp_name=exp_name, iter_number=driver_iterations)
    # extract a measure for the hp
    RLBR_df = dfs['MO_LEDUC_EVAL SINGLE_stack_13: RL-BR Total']
    BR_df = dfs['MO_LEDUC_EVAL SINGLE_stack_13: BR Total']
    diff = RLBR_df - BR_df

    # append to a dict hp -> measure
    hp_measure[5 * i] = diff.mean()['Evaluation/MA_per_G']

pdb.set_trace()