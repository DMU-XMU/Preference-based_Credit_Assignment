from .basis_alg.sac import SAC
basis_algorithm_collection = {
    'sac': SAC
}

from .ircr import IRCR
from .rrd import RandomizedReturnDecomposition as RRD
from .ours import RewardPredictor

advanced_algorithm_collection = {
    'ircr': IRCR,
    'rrd': RRD,
    'ours': RewardPredictor
}

algorithm_collection = {
    **basis_algorithm_collection,
    **advanced_algorithm_collection
}

def create_agent(args):
    return basis_algorithm_collection[args.alg](args)

def create_predictor(args):
    if args.pre=='rrd':
        return RRD(args).to(args.device)
    if args.pre=='ircr':
        return IRCR(args)
    return RewardPredictor(args).to(args.device)
