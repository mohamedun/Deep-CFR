#usage: evaluateBR.py path_to_agent

import sys
from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
method = sys.argv[1]
path_to_agent = sys.argv[2]
agent_to_eval = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=path_to_agent)

from DeepCFR.workers.driver.Driver import Driver

ctrl = Driver(t_prof=agent_to_eval.t_prof, eval_methods={method: 1}, n_iterations=1)
ctrl.evaluate()

