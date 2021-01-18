#usage: evaluateBR.py path_to_agent

import sys
from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
method = sys.argv[1]
path_to_agent = sys.argv[2]
agent_to_eval = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=path_to_agent)

#------- Chnage here to dist/LBR etc
from DeepCFR.workers.chief.local import Chief
from PokerRL.eval.br.DistBRMaster import DistBRMaster as BRMaster
import ray
ray.init()
eval_master = BRMaster.remote(t_prof=agent_to_eval.t_prof,
                       chief_handle=Chief(t_prof=agent_to_eval.t_prof),
                       eval_agent_cls=EvalAgentDeepCFR)

import pdb
#pdb.set_trace()

print(ray.get(eval_master.set_agent.remote(2)))
#eval_master.evaluate.remote(iter_nr=0)


