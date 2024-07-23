import numpy as np
from test import Tester
from envs import make_env, envs_collection
from algorithm import create_agent, algorithm_collection, create_predictor
from algorithm.replay_buffer import create_buffer
from learner import create_learner
from utils.os_utils import get_arg_parser, get_logger, str2bool

def get_args():
    parser = get_arg_parser()

    # basic arguments
    parser.add_argument('--tag', help='the terminal tag in logger', type=str, default='')
    parser.add_argument('--env', help='gym env id', type=str, default='Swimmer-v2')
    parser.add_argument('--env_type', help='the type of environment', type=str, default='ep_rews')
    parser.add_argument('--alg', help='backend algorithm', type=str, default='sac', choices=algorithm_collection.keys())
    parser.add_argument('--pre', help='predictor', type=str, default='none')
    parser.add_argument('--delta1',  type=np.float32, default=0.3)
    parser.add_argument('--delta2', type=np.float32, default=0.8)

    # training arguments
    parser.add_argument('--epochs', help='the number of epochs', type=np.int32, default=3)
    parser.add_argument('--cycles', help='the number of cycles per epoch', type=np.int32, default=400)
    parser.add_argument('--iterations', help='the number of iterations per cycle', type=np.int32, default=25)
    parser.add_argument('--timesteps', help='the number of timesteps per iteration', type=np.int32, default=100)

    # testing arguments
    parser.add_argument('--test_rollouts', help='the number of rollouts to test per cycle', type=np.int32, default=50)
    parser.add_argument('--test_timesteps', help='the number of timesteps per rollout', type=np.int32, default=100)
    parser.add_argument('--save_rews', help='whether to save cumulative rewards', type=str2bool, default=True)

    # buffer arguments
    parser.add_argument('--buffer_size', help='the number of transitions in replay buffer', type=np.int32, default=1000000)
    parser.add_argument('--batch_size', help='the size of sample batch', type=np.int32, default=256)
    parser.add_argument('--warmup', help='the number of timesteps for buffer warmup', type=np.int32, default=10000)

    args, _ = parser.parse_known_args()

    # env arguments
    parser.add_argument('--gamma', help='discount factor', type=np.float32, default=0.99)
       
    def fetch_args():
        pass

    env_args_collection = {
        'fetch': fetch_args,
    }
    env_category = envs_collection[args.env]
    env_args_collection[env_category]()
    
    def sac_args():

        parser.add_argument('--policy', default="Gaussian",
                            help='Policy Type: Gaussian | Deterministic (default: Gaussian)')

        parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                            help='target smoothing coefficient(τ) (default: 0.005)')
        parser.add_argument('--lr', type=float, default=0.001, metavar='G',
                            help='learning rate (default: 0.0003)')
        parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                            help='Temperature parameter α determines the relative importance of the entropy\
                                    term against the reward (default: 0.2)')
        parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                            help='Automaically adjust α (default: False)')
        parser.add_argument('--hidden_size', type=int, default=64, metavar='N',
                            help='hidden size (default: 256)')
        parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                            help='Value target update per no. of updates per step (default: 1)')
        parser.add_argument('--cuda', action="store_true", default='True',
                            help='run on CUDA (default: False)')


    default_batch_size, default_sample_size = { 'fetch': (256, 64)}[env_category]
    parser.add_argument('--rrd_batch_size', help='the size of sample batch for reward regression', type=np.int32, default=default_batch_size)
    parser.add_argument('--rrd_sample_size', help='the size of sample for reward regression', type=np.int32, default=default_sample_size)
    parser.add_argument('--rrd_bias_correction', help='whether to use bias-correction loss', type=str2bool, default=True)

    global algorithm_args_collection
    algorithm_args_collection = {
        'sac': sac_args,
    }
    algorithm_args_collection[args.alg]()

    args = parser.parse_args()
    args.env_category = envs_collection[args.env]

    logger_name = args.alg+'-'+args.env
    if args.tag!='': logger_name = args.tag+'-'+logger_name
    args.logger = get_logger(logger_name)

    for key, value in args.__dict__.items():
        if key!='logger':
            args.logger.info('{}: {}'.format(key,value))

    return args

def experiment_setup(args):
    env = make_env(args)
    args.act_type = 'continuous'
    args.acts_dims = env.acts_dims
    args.obs_dims = env.obs_dims
    args.act_space = env.action_space
    args.device = 'cuda:0'
    args.env_instance = env
    args.buffer = buffer = create_buffer(args)
    args.agent = agent = create_agent(args)

    args.learner = learner = create_learner(args)
    if args.pre != 'none':
        args.predictor = predictor = create_predictor(args)
    else:
        args.predictor= predictor  = None
    args.logger.info('*** network initialization complete ***')
    args.tester = tester = Tester(args)
    args.logger.info('*** tester initialization complete ***')

    return env, agent, buffer, learner, tester, predictor
