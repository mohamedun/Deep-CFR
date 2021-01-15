#usage: evaluateBR.py path_to_agent

import sys
from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
path_to_agent = sys.argv[1]
agent_to_evaluate = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=path_to_agent)

from DeepCFR.workers.driver.Driver import Driver

ctrl = Driver(t_prof=agent_to_eval.t_prof, eval_methods={'br': 1}, n_iterations=1)
ctrl.run()
