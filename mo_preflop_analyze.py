import pdb
import time
from os.path import dirname, abspath
import numpy as np
import sys
from PokerRL.game.Poker import Poker


from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR


path_to_eval_agent = sys.argv[1]
#History is represented by a string r for raise, c for call/check, f for fold.
str_to_action = {'r': Poker.BET_RAISE,
                 'c': Poker.CHECK_CALL,
                 'f': Poker.FOLD}
history = sys.argv[2]

N_DECK = 52
N_HOLE = 169 # Number of possible hole cards 13 * 12 + 13

#A function that takes a hole hand and produces (high rank, low rank, is_suited) representation
def hand2rep(hand):
    card1_rank = hand[0][0]
    card1_suit = hand[0][1]
    card2_rank = hand[1][0]
    card2_suit = hand[1][1]
    suited = (card2_suit == card1_suit)
    high_rank = max(card1_rank, card2_rank)
    low_rank = min(card1_rank, card2_rank)
    return (high_rank, low_rank, suited)

#Load EvalAgent from file
curr_eval_agent = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=path_to_eval_agent)

#get an env bldr from the agent and create an env
env_bldr = curr_eval_agent.env_bldr
env = env_bldr.get_new_env(is_evaluating=False)

start_time = time.time()
hands = {}
while len(hands) < N_HOLE:
    #Reset env and EvalAgent
    env.reset()
    curr_eval_agent.reset(deck_state_dict=env.cards_state_dict())
    #Act
    for c in history:
        current_seat = env.current_player.seat_id
        env.step(str_to_action[c])
        curr_eval_agent.notify_of_action(p_id_acted=current_seat, action_he_did=str_to_action[c])

    hole_hand = hand2rep(env.seats[env.current_player.seat_id].hand)
    if hole_hand not in hands:
        hands[hole_hand] = curr_eval_agent.get_a_probs()


print(f"Computed {N_HOLE} possible hands in {time.time()-start_time} sec")

#----------------------------Store Data
import pickle
f = open(history + '_strat.pkl', 'ab')
pickle.dump(hands, f)
f.close()

#----------------------- Generate and Store Image
import plot_strat
plot_strat.np2img(hands, history + path_to_eval_agent + '_strat_img.png')