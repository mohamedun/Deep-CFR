#usage: evaluateBR.py path_to_agent

import sys
import pdb
from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from PokerRL.util.file_util import do_pickle, load_pickle
method = sys.argv[1]
path_to_agent = sys.argv[2]
agent_to_eval = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=path_to_agent)
state_from_disk = load_pickle(path=path_to_agent)
agent_prof = agent_to_eval.t_prof

# #------- Dist BR
# from DeepCFR.workers.chief.local import Chief
# from PokerRL.eval.br.DistBRMaster import DistBRMaster as BRMaster
# import ray
# ray.init()
# eval_master = BRMaster.remote(t_prof=agent_prof,
#                        chief_handle=Chief(t_prof=agent_prof),
#                        eval_agent_cls=EvalAgentDeepCFR)
#
# import pdb
# #pdb.set_trace()
#
# print(ray.get(eval_master.set_agent.remote(2)))
# #eval_master.evaluate.remote(iter_nr=0)
#
# #-----------Local LBR
# from PokerRL.eval.lbr.LocalLBRMaster import LocalLBRMaster as LBRMaster
# from PokerRL.eval.lbr.LocalLBRWorker import LocalLBRWorker as LBRWorker
# from PokerRL.eval.lbr.LBRArgs import LBRArgs
# agent_prof.module_args['lbr'] = LBRArgs()
# lbr_chief = Chief(t_prof=agent_prof)
# eval_master = LBRMaster(t_prof=agent_to_eval.t_prof,
#                                chief_handle=lbr_chief)
# num_workers = 3
# LBR_workers = [LBRWorker(t_prof=agent_prof, chief_handle=lbr_chief, eval_agent_cls=EvalAgentDeepCFR) for _ in range(num_workers)]
# eval_master.set_worker_handles(*LBR_workers)
# #
#
# #eval_master.evaluate(0)

#-------- Driver Approach
from DeepCFR.workers.driver.Driver import Driver
from PokerRL.eval.lbr.LBRArgs import LBRArgs
agent_prof.module_args['lbr'] = LBRArgs()
ctrl = Driver(agent_prof, eval_methods={'lbr': 1})
ctrl.chief_handle._strategy_buffers = agent_to_eval._strategy_buffers
w, info = ctrl.chief_handle._pull_current_eval_strategy()

ctrl.eval_masters['lbr'][0]._eval_agent = agent_to_eval
ctrl.eval_masters['lbr'][0].weights_for_eval_agent = w
ctrl.eval_masters['lbr'][0].evaluate(0)


# ctrl.eval_masters['br'][0]._eval_agent = agent_to_eval
# ctrl.eval_masters['br'][0].evaluate(0)


# agent_prof.module_args['lbr'] = LBRArgs()
# ctrl = Driver(agent_prof, eval_methods={'lbr': 1})
#
# ctrl.eval_masters['lbr'][0]._eval_agent = agent_to_eval
# ctrl.eval_masters['lbr'][0].evaluate(0)
'''
locallbrmaster calls update weights on each worker with the weights it has gotten from chief
each worker calls update weights on its agent wrapper
the agent wrapper contains an EvalAgent, at the end the EvalAgent.update_weights gets called
with the weights that the master got from chief using _pull*strat 
want: put weights in lbrmaster in the same format that it gets from pull_current_strat_from_chief...
which calls chief.pull_current_eval_strategy([n,n])

so compare current_eval_strategy with store to disk 
for single, storing has everything in chief._strategy_buffers

strategy for SINGLE: update _strategy_buffers in chief from eval_agent._strategy_buffers
then pull it using the chief method, then update the LBR master with these update_weights


LBRMaster_inst.weights_for_eval_agents = self.pull_current_strat_from_chief()

EvaluatorMasterBase_inst.pull_current_strat_from_chief()
w, info = chief_handle.pull_current_eval_strategy
return w
'''

