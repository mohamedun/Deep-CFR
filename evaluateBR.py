#usage: evaluateBR.py path_to_agent

import sys
from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
method = sys.argv[1]
path_to_agent = sys.argv[2]
agent_to_eval = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=path_to_agent)

from DeepCFR.workers.driver.Driver import Driver
from DeepCFR.workers.chief.dist import Chief
from PokerRL.eval.br.DistBRMaster import DistBRMaster as BRMaster

eval_master = BRMaster(t_prof=agent_to_eval.t_prof,
                       chief_handle=Chief(t_prof=agent_to_eval.t_prof),
                       eval_agent_cls= EvalAgentDeepCFR)

eval_master._eval_agent = agent_to_eval
eval_master.evaluate()


