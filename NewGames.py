from PokerRL.game.Poker import Poker
from PokerRL.game._.rl_env.game_rules import HoldemRules, LeducRules, FlopHoldemRules, BigLeducRules
from PokerRL.game._.rl_env.poker_types.DiscretizedPokerEnv import DiscretizedPokerEnv
from PokerRL.game._.rl_env.poker_types.LimitPokerEnv import LimitPokerEnv
from PokerRL.game._.rl_env.poker_types.NoLimitPokerEnv import NoLimitPokerEnv

class ModFlop5Holdem(FlopHoldemRules, LimitPokerEnv):
    RULES = FlopHoldemRules
    IS_FIXED_LIMIT_GAME = True
    IS_POT_LIMIT_GAME = False

    SMALL_BLIND = 50
    BIG_BLIND = 100
    ANTE = 0
    DEFAULT_STACK_SIZE = 20000

    EV_NORMALIZER = 1000.0 / BIG_BLIND  # Milli BB
    WIN_METRIC = Poker.MeasureBB

    MAX_N_RAISES_PER_ROUND = {
        Poker.PREFLOP: 3,  # is actually 1, but BB counts as a raise in this codebase
        Poker.FLOP: 2,
    }
    ROUND_WHERE_BIG_BET_STARTS = Poker.TURN

    UNITS_SMALL_BET = None
    UNITS_BIG_BET = None

    FIRST_ACTION_NO_CALL = True

    def __init__(self, env_args, lut_holder, is_evaluating):
        FlopHoldemRules.__init__(self)
        LimitPokerEnv.__init__(self,
                               env_args=env_args,
                               lut_holder=lut_holder,
                               is_evaluating=is_evaluating)

    def _adjust_raise(self, raise_total_amount_in_chips):
        return self.get_fraction_of_pot_raise(fraction=1.0, player_that_bets=self.current_player)

class ModLimitHoldem(HoldemRules, LimitPokerEnv):
    """
    Fixed-Limit Texas Hold'em is a long-standing benchmark game that has been essentially solved by Bowling et al
    (http://science.sciencemag.org/content/347/6218/145) using an efficient distributed implementation of CFR+, an
    optimized version of regular CFR.
    """

    RULES = HoldemRules
    IS_FIXED_LIMIT_GAME = True
    IS_POT_LIMIT_GAME = False
    MAX_N_RAISES_PER_ROUND = {
        Poker.PREFLOP: 4,
        Poker.FLOP: 3,
        Poker.TURN: 3,
        Poker.RIVER: 3,
    }
    ROUND_WHERE_BIG_BET_STARTS = Poker.TURN

    SMALL_BLIND = 1
    BIG_BLIND = 2
    ANTE = 0
    SMALL_BET = 2
    BIG_BET = 4
    DEFAULT_STACK_SIZE = 48

    EV_NORMALIZER = 1000.0 / BIG_BLIND  # Milli BB
    WIN_METRIC = Poker.MeasureBB

    def __init__(self, env_args, lut_holder, is_evaluating):
        HoldemRules.__init__(self)
        LimitPokerEnv.__init__(self,
                               env_args=env_args,
                               lut_holder=lut_holder,
                               is_evaluating=is_evaluating)