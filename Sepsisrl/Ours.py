import tensorflow as tf
import numpy as np
import math
import os
from tqdm import *
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans
from joblib import dump, load
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
import copy
os.environ['CUDA_VISIBLE_DEVICES']='0'
noi = 10

REWARD_THRESHOLD = 20
reg_lambda = 5
per_alpha = 0.6 # PER hyperparameter
per_epsilon = 0.01 # PER hyperparameter
batch_size = 32
gamma = 0.99 # discount factor
num_steps = 70000 # How many steps to train for
load_model = False #Whether to load a saved model.
save_dir = "./estimate/mc5_alpha=0.3_prd_{}/".format(str(noi))
save_path = "./estimate/mc5_alpha=0.3_prd_{}/ckpt".format(str(noi))#The path to save our model to.
tau = 0.001 #Rate to update target network toward primary network
save_results = False
# SET THIS TO FALSE
clip_reward = False
with open('../data_o/state_features.txt') as f:
    state_features = f.read().split()
print (state_features)

df = pd.read_csv('../data_prd/prd_train_data_10000_5_9_noi_{}.csv'.format(str(noi)))
val_df = pd.read_csv('../data_prd/prd_val_data_10000_5_9_noi_{}.csv'.format(str(noi)))
test_df = pd.read_csv('../data_prd/prd_test_data_10000_5_9_noi_{}.csv'.format(str(noi)))



# to search module
df_unscaled = pd.read_csv('../data_o/rl_train_set_unscaled.csv')  #未缩放时的数据
kmeans_train = load('../data_o/kmeans_750.joblib')
df_state_ids= pd.read_csv('../data_o/rl_train_data_discrete.csv')

# to numpy ,speed up
df_np = np.array(df)
df_state_ids_np = np.array(df_state_ids)
print(df_state_ids_np.shape)
#df_unscaled_np = np.array(df_unscaled)

###############################

##### to DR estimator ########
# load in the policies for the physician on val and test sets
#phys_policy_test = pickle.load(open("test_policy.p", "rb" ))
#Pi_b = phys_policy_test[:,5:30]
phys_policy_test_KNN = pickle.load(open("../data_o/test_policy_KNN.p", "rb" ))
Pi_b = phys_policy_test_KNN
###############################

# PER important weights and params
per_flag = True  #是否使用权重采样
beta_start = 0.9
df['prob'] = abs(df['reward'])
temp = 1.0/df['prob']
df['imp_weight'] = pow((1.0/len(df) * temp), beta_start)

action_map = {}
count = 0
for iv in range(5):
    for vaso in range(5):
        action_map[(iv,vaso)] = count
        count += 1

hidden_1_size = 128
hidden_2_size = 128
#  Q-network uses Leaky ReLU activation
class Qnetwork():
    def __init__(self):
        self.phase = tf.placeholder(tf.bool)

        self.num_actions = 25
        self.alpha = 0.3
        self.input_size = len(state_features)

        self.state = tf.placeholder(tf.float32, shape=[None, self.input_size], name="input_state")

        self.fc_1 = tf.contrib.layers.fully_connected(self.state, hidden_1_size, activation_fn=None)
        self.fc_1_bn = tf.contrib.layers.batch_norm(self.fc_1, center=True, scale=True, is_training=self.phase)
        self.fc_1_ac = tf.maximum(self.fc_1_bn, self.fc_1_bn * 0.01)
        self.fc_2 = tf.contrib.layers.fully_connected(self.fc_1_ac, hidden_2_size, activation_fn=None)
        self.fc_2_bn = tf.contrib.layers.batch_norm(self.fc_2, center=True, scale=True, is_training=self.phase)
        self.fc_2_ac = tf.maximum(self.fc_2_bn, self.fc_2_bn * 0.01)

        # advantage and value streams
        self.streamA, self.streamV = tf.split(self.fc_2_ac, 2, axis=1)
        self.AW = tf.Variable(tf.random_normal([hidden_2_size // 2, self.num_actions]))
        self.VW = tf.Variable(tf.random_normal([hidden_2_size // 2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        # Then combine them together to get our final Q-values.
        self.q_output = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))

        self.predict = tf.argmax(self.q_output, 1, name='predict')  # vector of length batch size

        # Below we obtain the loss by taking the sum of squares difference between the target and predicted Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.MC = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, self.num_actions, dtype=tf.float32)

        # Importance sampling weights for PER, used in network update
        self.imp_weights = tf.placeholder(shape=[None], dtype=tf.float32)

        # select the Q values for the actions that would be selected
        self.Q = tf.reduce_sum(tf.multiply(self.q_output, self.actions_onehot),
                               reduction_indices=1)  # batch size x 1 vector

        # regularisation penalises the network when it produces rewards that are above the
        # reward threshold, to ensure reasonable Q-value predictions
        self.reg_vector = tf.maximum(tf.abs(self.Q) - REWARD_THRESHOLD, 0)
        self.reg_term = tf.reduce_sum(self.reg_vector)

        self.abs_error = tf.abs(self.targetQ - self.Q)

        self.td_error = tf.square(self.targetQ - self.Q)  + self.alpha * tf.square(self.MC - self.Q)

        # below is the loss when we are not using PER
        self.old_loss = tf.reduce_mean(self.td_error)

        # as in the paper, to get PER loss we weight the squared error by the importance weights
        self.per_error = tf.multiply(self.td_error, self.imp_weights)

        # total loss is a sum of PER loss and the regularisation term
        if per_flag:
            self.loss = tf.reduce_mean(self.per_error) + reg_lambda * self.reg_term
        else:
            self.loss = self.old_loss + reg_lambda * self.reg_term

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            # Ensures that we execute the update_ops before performing the model update, so batchnorm works
            self.update_model = self.trainer.minimize(self.loss)
def update_target_graph(tf_vars,tau):
    total_vars = len(tf_vars)
    op_holder = []
    for idx,var in enumerate(tf_vars[0:int(total_vars/2)]):
        op_holder.append(tf_vars[idx+int(total_vars/2)].assign((var.value()*tau) + ((1-tau)*tf_vars[idx+int(total_vars/2)].value())))
    return op_holder
def update_target(op_holder,sess):
    for op in op_holder:
        sess.run(op)


def process_train_batch(size):
    if per_flag:
        # uses prioritised exp replay
        a = df.sample(n=size, weights=df['prob'])
    else:
        a = df.sample(n=size)
    states = None
    actions = None
    rewards = None
    next_states = None
    done_flags = None
    for i in a.index:
        cur_state = a.loc[i, state_features]
        iv = int(a.loc[i, 'iv_input'])
        vaso = int(a.loc[i, 'vaso_input'])
        action = action_map[iv, vaso]
        reward = a.loc[i, 'reward']

        if clip_reward:
            if reward > 1: reward = 1
            if reward < -1: reward = -1

        if i != df.index[-1]:
            # if not terminal step in trajectory
            if df.loc[i, 'icustayid'] == df.loc[i + 1, 'icustayid']:
                next_state = df.loc[i + 1, state_features]
                done = 0
            else:
                # trajectory is finished
                next_state = np.zeros(len(cur_state))
                done = 1
        else:
            # last entry in df is the final state of that trajectory
            next_state = np.zeros(len(cur_state))
            done = 1

        if states is None:
            states = copy.deepcopy(cur_state)
        else:
            states = np.vstack((states, cur_state))

        if actions is None:
            actions = [action]
        else:
            actions = np.vstack((actions, action))

        if rewards is None:
            rewards = [reward]
        else:
            rewards = np.vstack((rewards, reward))

        if next_states is None:
            next_states = copy.deepcopy(next_state)
        else:
            next_states = np.vstack((next_states, next_state))

        if done_flags is None:
            done_flags = [done]
        else:
            done_flags = np.vstack((done_flags, done))

    return (states, np.squeeze(actions), np.squeeze(rewards), next_states, np.squeeze(done_flags), a)
# extract chunks of length size from the relevant dataframe, and yield these to the caller
def process_eval_batch(size, eval_type=None):
    if eval_type is None:
        raise Exception('Provide eval_type to process_eval_batch')
    elif eval_type == 'train':
        a = df.copy()
    elif eval_type == 'val':
        a = val_df.copy()
    elif eval_type == 'test':
        a = test_df.copy()
    else:
        raise Exception('Unknown eval_type')
    count = 0
    while count < len(a.index):
        states = None
        actions = None
        rewards = None
        next_states = None
        done_flags = None

        start_idx = count
        end_idx = min(len(a.index), count + size)
        segment = a.index[start_idx:end_idx]

        for i in segment:
            cur_state = a.loc[i, state_features]
            iv = int(a.loc[i, 'iv_input'])
            vaso = int(a.loc[i, 'vaso_input'])
            action = action_map[iv, vaso]
            reward = a.loc[i, 'reward']

            if clip_reward:
                if reward > 1: reward = 1
                if reward < -1: reward = -1

            if i != a.index[-1]:
                # if not terminal step in trajectory
                if a.loc[i, 'icustayid'] == a.loc[i + 1, 'icustayid']:
                    next_state = a.loc[i + 1, state_features]
                    done = 0
                else:
                    # trajectory is finished
                    next_state = np.zeros(len(cur_state))
                    done = 1
            else:
                # last entry in df is the final state of that trajectory
                next_state = np.zeros(len(cur_state))
                done = 1

            if states is None:
                states = copy.deepcopy(cur_state)
            else:
                states = np.vstack((states, cur_state))

            if actions is None:
                actions = [action]
            else:
                actions = np.vstack((actions, action))

            if rewards is None:
                rewards = [reward]
            else:
                rewards = np.vstack((rewards, reward))

            if next_states is None:
                next_states = copy.deepcopy(next_state)
            else:
                next_states = np.vstack((next_states, next_state))

            if done_flags is None:
                done_flags = [done]
            else:
                done_flags = np.vstack((done_flags, done))

        yield (states, np.squeeze(actions), np.squeeze(rewards), next_states, np.squeeze(done_flags), a)

        count += size
def do_eval(eval_type):
    gen = process_eval_batch(size=1000, eval_type=eval_type)
    phys_q_ret = []
    actions_ret = []
    agent_q_ret = []
    actions_taken_ret = []
    agent_qsa_ret = []
    error_ret = 0

    for b in gen:
        states, actions, rewards, next_states, done_flags, _ = b

        # firstly get the chosen actions at the next timestep
        actions_from_q1 = sess.run(mainQN.predict, feed_dict={mainQN.state: next_states, mainQN.phase: 0})

        # Q values for the next timestep from target network, as part of the Double DQN update
        Q2 = sess.run(targetQN.q_output, feed_dict={targetQN.state: next_states, targetQN.phase: 0})

        # handles the case when a trajectory is finished
        end_multiplier = 1 - done_flags

        double_q_value = Q2[range(len(Q2)), actions_from_q1]

        targetQ = rewards + (gamma * double_q_value * end_multiplier)

        q_output, actions_taken, abs_error = sess.run([mainQN.q_output, mainQN.predict, mainQN.abs_error], \
                                                      feed_dict={mainQN.state: states,
                                                                 mainQN.targetQ: targetQ,
                                                                 mainQN.actions: actions,
                                                                 mainQN.phase: False})
        # return the relevant q values and actions
        phys_q = q_output[range(len(q_output)), actions]
        agent_q = q_output[range(len(q_output)), actions_taken]
        error = np.mean(abs_error)

        #       update the return vals
        phys_q_ret.extend(phys_q)
        actions_ret.extend(actions)
        agent_qsa_ret.extend(q_output)  # qsa
        agent_q_ret.extend(agent_q)  # q
        actions_taken_ret.extend(actions_taken)  # a
        error_ret += error

    return agent_qsa_ret, phys_q_ret, actions_ret, agent_q_ret, actions_taken_ret, error_ret
def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)
def DR_estimator(Pi_e,Pi_b,Q_agent,df_test):
    unique_ids = df_test['icustayid'].unique()
    DR = []
    rho_all = []
    ind = 0
    for uid in unique_ids:
        rho = []
        traj = df_test.loc[df_test['icustayid'] == uid]
        for t in range(len(traj)):
            iv = df_test.loc[ind,'iv_input']
            vaso = df_test.loc[ind, 'vaso_input']
            phys_action = action_map[(iv,vaso)]    #行为策略 a
            #agent_a = df_test['agent_action'][ind]
            rho_t = Pi_e[ind][phys_action] / Pi_b[ind][phys_action]  #df_test['phys_prob'][ind]   #使用的是数据行为动作
            rho.append(rho_t)
            ind +=1
        rho_all.append(np.array(rho))
    max_H = max(len(rho) for rho in rho_all)
    rho_cum = np.zeros((len(unique_ids), max_H))
    for i, rho in enumerate(rho_all):
        rho_tmp = np.ones(max_H)
        rho_tmp[:len(rho)] = rho
        rho_cum[i] = np.cumprod(rho_tmp)
    ind = 0
    n_traj = 0
    for uid in unique_ids:
        trajectory = df_test.loc[df_test['icustayid'] == uid]
        rho_cumulative = rho_cum[n_traj]
        V_WDR = 0
        for t in range(len(trajectory)):
            iv = df_test.loc[ind,'iv_input']
            vaso = df_test.loc[ind, 'vaso_input']
            phys_action = action_map[(iv,vaso)]   #行为策略 a
            Q_hat =  Q_agent[ind][phys_action]   #   test_set <s,a,r>
            V_hat = np.nansum(Q_agent[ind] * Pi_e[ind])
            r_t = df_test['reward'][ind]
            rho_1t = rho_cumulative[t]
            if t == 0:
                rho_1t_1 = 1.0
            else:
                rho_1t_1 = rho_cumulative[t-1]
            V_WDR = V_WDR + np.power(gamma, t) * (rho_1t * r_t - (rho_1t * Q_hat - rho_1t_1 * V_hat))
            #print(V_WDR)
            ind+=1
        DR.append(V_WDR)
        n_traj +=1
    return DR
def WDR_estimator(Pi_e, Pi_b, Q_agent, df_test):
    unique_ids = df_test['icustayid'].unique()
    rho_all = []
    DR = []
    V_WDR = 0
    ind = 0
    for uid in unique_ids:
        rho = []
        traj = df_test.loc[df_test['icustayid'] == uid]
        for t in range(len(traj)):
            iv = df_test.loc[ind, 'iv_input']
            vaso = df_test.loc[ind, 'vaso_input']
            phys_action = action_map[(iv, vaso)]  # 行为策略 a
            if np.isclose(Pi_b[ind][phys_action], 0.0):
                rho_t = Pi_e[ind][phys_action] / 0.001
            else:
                rho_t = Pi_e[ind][phys_action] / Pi_b[ind][phys_action]  # df_test['phys_prob'][ind]   #使用的是数据行为动作
            rho.append(rho_t)
            ind += 1
        rho_all.append(np.array(rho))
    max_H = max(len(rho) for rho in rho_all)
    rho_cum = np.zeros((len(unique_ids), max_H))
    for i, rho in enumerate(rho_all):
        rho_tmp = np.ones(max_H)
        rho_tmp[:len(rho)] = rho
        rho_cum[i] = np.cumprod(rho_tmp)  # 累乘（uids，H）
    weights = np.mean(rho_cum, axis=0)
    ind = 0
    n_traj = 0

    for uid in unique_ids:
        trajectory = df_test.loc[df_test['icustayid'] == uid]
        rho_cumulative = rho_cum[n_traj]
        V_WDR = 0
        for t in range(len(trajectory)):
            iv = df_test.loc[ind, 'iv_input']
            vaso = df_test.loc[ind, 'vaso_input']
            phys_action = action_map[(iv, vaso)]  # 行为策略 a
            Q_hat = Q_agent[ind][phys_action]  # test_set <s,a,r>
            V_hat = np.nansum(Q_agent[ind] * Pi_e[ind])
            # V_hat = Q_agent[ind].max()
            r_t = df_test['reward'][ind]
            rho_1t = rho_cumulative[t] / weights[t]
            if t == 0:
                rho_1t_1 = 1.0
            else:
                rho_1t_1 = rho_cumulative[t - 1] / weights[t - 1]
            V_WDR = V_WDR + np.power(gamma, t) * (rho_1t * r_t - (rho_1t * Q_hat - rho_1t_1 * V_hat))
            ind += 1
        DR.append(V_WDR)
        n_traj += 1
    return DR
config = tf.ConfigProto()
config.gpu_options.allow_growth = False  # Don't use all GPUs
config.allow_soft_placement = True  # Enable manual control
def do_save_results(type=None):
    # get the chosen actions for the train, val, and test set when training is complete.
    if type == None:
        agent_qsa_train, _, _, agent_q_train, agent_actions_train, _ = do_eval(eval_type='train')
        agent_qsa_val, _, _, agent_q_val, agent_actions_val, _ = do_eval(eval_type='val')
        agent_qsa_test, _, _, agent_q_test, agent_actions_test, _ = do_eval(eval_type='test')
        print("length IS ", len(agent_actions_train))

        # save everything for later - they're used in policy evaluation and when generating plots
        with open(save_dir + 'dqn_prd_actions_train.p', 'wb') as f:
            pickle.dump(agent_actions_train, f)
        with open(save_dir + 'dqn_prd_actions_val.p', 'wb') as f:
            pickle.dump(agent_actions_val, f)
        with open(save_dir + 'dqn_prd_actions_test.p', 'wb') as f:
            pickle.dump(agent_actions_test, f)

        with open(save_dir + 'dqn_prd_q_train.p', 'wb') as f:
            pickle.dump(agent_q_train, f)
        with open(save_dir + 'dqn_prd_q_val.p', 'wb') as f:
            pickle.dump(agent_q_val, f)
        with open(save_dir + 'dqn_prd_q_test.p', 'wb') as f:
            pickle.dump(agent_q_test, f)

        with open(save_dir + 'dqn_prd_qsa_train.p', 'wb') as f:
            pickle.dump(agent_qsa_train, f)
        with open(save_dir + 'dqn_prd_qsa_val.p', 'wb') as f:
            pickle.dump(agent_qsa_val, f)
        with open(save_dir + 'dqn_prd_qsa_test.p', 'wb') as f:
            pickle.dump(agent_qsa_test, f)
    elif  type=='test':
        agent_qsa_test, _, _, agent_q_test, agent_actions_test, _ = do_eval(eval_type='test')
        with open(save_dir + 'dqn_prd_qsa_test.p', 'wb') as f:
            pickle.dump(agent_qsa_test, f)
        with open(save_dir + 'dqn_prd_q_test.p', 'wb') as f:
            pickle.dump(agent_q_test, f)
        with open(save_dir + 'dqn_prd_actions_test.p', 'wb') as f:
            pickle.dump(agent_actions_test, f)
    return

def future_info_search(index,actions,rewards):
    states_IDs = df_state_ids_np[index,2]
    gamma = 0.99
    Mc_return = []
    for n,ind in enumerate(index): #32
        a_target = actions[n]
        s_target = states_IDs[n]
        for i in range(0,195120,8):
            a_i = df_np[i,61]                  #(190000, 63)
            s_i = df_state_ids_np[i,2]         #(190000, 5)
            if  s_target==s_i and a_target==a_i : #and abs(r_target-r_i)<0.1  #r_i = df.loc[i,'reward'] #r_target = rewards[n]
                if df_state_ids_np[i,1] == df_state_ids_np[i+3,1]:
                    Mc_return.append(df_state_ids_np[i,3]+gamma**1 *df_state_ids_np[i+1,3]+gamma**2 *df_state_ids_np[i+2,3]+gamma**3 *df_state_ids_np[i+3,3])
                    break
            if i>195110:
                Mc_return.append(0.0)
    return np.array(Mc_return)

# The main training loop is here

tf.reset_default_graph()
mainQN = Qnetwork()
targetQN = Qnetwork()
av_q_list = []
DR_estimator_list = []
saver = tf.train.Saver(tf.global_variables())
init = tf.global_variables_initializer()
trainables = tf.trainable_variables()
target_ops = update_target_graph(trainables, tau)

# Make a path for our model to be saved in.
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with tf.Session(config=config) as sess:
    if load_model == True:
        print('Trying to load model...')
        try:
            restorer = tf.train.import_meta_graph(save_path + '.meta')
            restorer.restore(sess, tf.train.latest_checkpoint(save_dir))
            print("Model restored")
        except IOError:
            print("No previous model found, running default init")
            sess.run(init)
        try:
            per_weights = pickle.load(open(save_dir + "per_weights.p", "rb"))
            imp_weights = pickle.load(open(save_dir + "imp_weights.p", "rb"))

            # the PER weights, governing probability of sampling, and importance sampling
            # weights for use in the gradient descent updates
            df['prob'] = per_weights
            df['imp_weight'] = imp_weights
            print("PER and Importance weights restored")
        except IOError:
            print("No PER weights found - default being used for PER and importance sampling")
    else:
        print("Running default init")
        sess.run(init)
    print("Init done")

    net_loss = 0.0
    for i in tqdm(range(num_steps)):
        if save_results:
            print("Calling do save results")
            do_save_results()
            break

        states, actions, rewards, next_states, done_flags, sampled_df = process_train_batch(batch_size)  #当前抽取的 <s,a,r,d,s_> batch=32

        mc_r = future_info_search(sampled_df.index,actions,rewards)
        #print(states[0],sampled_df.index[0])

        actions_from_q1 = sess.run(mainQN.predict, feed_dict={mainQN.state: next_states, mainQN.phase: 1})

        cur_act = sess.run(mainQN.predict, feed_dict={mainQN.state: states, mainQN.phase: 1})

        Q2 = sess.run(targetQN.q_output, feed_dict={targetQN.state: next_states, targetQN.phase: 1})

        end_multiplier = 1 - done_flags

        double_q_value = Q2[range(len(Q2)), actions_from_q1]

        double_q_value[double_q_value > REWARD_THRESHOLD] = REWARD_THRESHOLD
        double_q_value[double_q_value < -REWARD_THRESHOLD] = -REWARD_THRESHOLD

        targetQ = rewards + (gamma * double_q_value * end_multiplier)

        # Calculate the importance sampling weights for PER
        imp_sampling_weights = np.array(sampled_df['imp_weight'] / float(max(df['imp_weight'])))
        imp_sampling_weights[np.isnan(imp_sampling_weights)] = 1
        imp_sampling_weights[imp_sampling_weights <= 0.001] = 0.001

        # Train with the batch
        _, loss, error = sess.run([mainQN.update_model ,mainQN.loss, mainQN.abs_error], \
                                  feed_dict={mainQN.state: states,
                                             mainQN.targetQ: targetQ,
                                             mainQN.MC:mc_r,
                                             mainQN.actions: actions,
                                             mainQN.phase: True,
                                             mainQN.imp_weights: imp_sampling_weights})

        update_target(target_ops, sess)

        net_loss += sum(error)

        # Set the selection weight/prob to the abs prediction error and update the importance sampling weight
        new_weights = pow((error + per_epsilon), per_alpha)
        df.loc[df.index.isin(sampled_df.index), 'prob'] = new_weights
        temp = 1.0 / new_weights
        df.loc[df.index.isin(sampled_df.index), 'imp_weight'] = pow(((1.0 / len(df)) * temp), beta_start)

        if i % 10000 == 0 and i > 0:
            saver.save(sess, save_path)
            print("Saved Model, step is " + str(i))

            av_loss = net_loss / (10000.0 * batch_size)
            print("Average loss is ", av_loss)
            net_loss = 0.0

            print("Saving PER and importance weights")
            with open(save_dir + 'per_weights.p', 'wb') as f:
                pickle.dump(df['prob'], f)
            with open(save_dir + 'imp_weights.p', 'wb') as f:
                pickle.dump(df['imp_weight'], f)

        if i % 1000 == 0  :             # DR \ WDR 评估
            do_save_results(type= 'test')
            agent_qsa = np.array(pickle.load(open(save_dir+'/dqn_prd_qsa_test.p', "rb")))
            Pi_e = softmax(agent_qsa, axis=1)
            #DR = DR_estimator(Pi_e, Pi_b, agent_qsa, test_df)
            WDR = WDR_estimator(Pi_e, Pi_b, agent_qsa, test_df)
            print('step:',i, 'WDR:',np.nanmean(np.clip(WDR, -15, 15)))
            DR_estimator_list.append(np.nanmean(np.clip(WDR, -15, 15)))
    with open(save_dir + 'WDR_list.p', 'wb') as f:
        pickle.dump(DR_estimator_list, f)
    do_save_results()
