#usage: evaluateBR.py path_to_agent

import sys
from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
path_to_agent = sys.argv[1]
agent_to_evaluate = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=path_to_agent)

