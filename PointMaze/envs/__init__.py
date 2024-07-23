from .normal_fetch import FetchNormalEnv
from .ep_rews import create_EpisodicRewardsEnv

fetch_list = [
    'PointMaze_UMaze-v3','PointMaze_Medium-v3','PointMaze_Large-v3'
]


envs_collection = {
    # Fetch envs
    **{
        fetch_name : 'fetch'
        for fetch_name in fetch_list
    },
}

def make_env(args):
    normal_env = {
        'fetch': FetchNormalEnv
    }[envs_collection[args.env]]

    return {
        'normal': normal_env,
        'ep_rews': create_EpisodicRewardsEnv(normal_env)
    }[args.env_type](args)
