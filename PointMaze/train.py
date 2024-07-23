import numpy as np
import time
from common import get_args,experiment_setup
from torch.utils.tensorboard import SummaryWriter
import datetime


if __name__=='__main__':
    args = get_args()
    filename = 'log/{}_SAC_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.pre, args.env,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else "")
    writter = SummaryWriter(filename)
    args.writter = writter
    args.filename = filename
    env, agent, buffer, learner, tester, predictor = experiment_setup(args)
    
    

    episodes_cnt = 0
    tester.cycle_summary()
    for epoch in range(args.epochs):
        
        for cycle in range(args.cycles):

            learner.learn(args, env, agent, buffer, predictor)

            tester.cycle_summary()
        
        tester.epoch_summary()

    tester.final_summary()
