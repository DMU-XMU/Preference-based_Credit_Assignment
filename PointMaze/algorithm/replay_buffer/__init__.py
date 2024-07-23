from .fetch_buffer import ReplayBuffer_IRCR, ReplayBuffer_RRD, ReplayBuffer_Transition, ReplayBuffer_Predictor

def create_buffer(args):
    if args.pre =='ircr':
        return ReplayBuffer_IRCR(args)
    if args.pre =='rrd':
        return ReplayBuffer_RRD(args)
    if args.alg=='sac':
        return ReplayBuffer_Predictor(args)
    return ReplayBuffer_Transition(args)
