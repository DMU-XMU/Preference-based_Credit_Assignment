from .fetch import FetchLearner
def create_learner(args):
    return {
        'fetch': FetchLearner,
    }[args.env_category](args)
