#python 05_train_controller.py car_racing -e 1 -n 4 -t 1 --max_length 1000
#xvfb-run -a -s "-screen 0 1400x900x24" python 05_train_controller.py car_racing -n 4 -t 1 -e 1 --max_length 1000
#python 05_train_controller.py car_racing -e 4 -n 8 -t 2 --max_length 1000
#xvfb-run -a -s "-screen 0 1400x900x24" python 05_train_controller.py car_racing -n 16 -t 1 -e 4 --max_length 1000

from mpi4py import MPI
import numpy as np
import json
import os
import subprocess
import sys

import pickle
import random

from pympler.tracker import SummaryTracker

from model import make_model, simulate
from es import CMAES, SimpleGA, OpenES, PEPG
import argparse
import time

import config

### ES related code - parameters are just dummy values so do not edit here. Instead, set in the args to the script.
num_episode = 1
eval_steps = 25 # evaluate every N_eval steps
retrain_mode = True
dream_mode = 0
cap_time_mode = True

num_worker = 8
num_worker_trial = 16

population = num_worker * num_worker_trial

env_name = 'invalid_env_name'
optimizer = 'cma'
antithetic = True
batch_mode = 'mean'

max_length = -1

# seed for reproducibility
seed_start = 0

### name of the file (can override):
filebase = None

model = None
num_params = -1

es = None

### saved models

init_opt = ''

### MPI related code
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

PRECISION = 10000
SOLUTION_PACKET_SIZE = None
RESULT_PACKET_SIZE = None
###

def initialize_settings(sigma_init=0.1, sigma_decay=0.9999, init_opt = ''):
  global population, filebase, controller_filebase, model, num_params, es, PRECISION, SOLUTION_PACKET_SIZE, RESULT_PACKET_SIZE
  population = num_worker * num_worker_trial
  filebase = './log/'+env_name+'.'+optimizer+'.'+str(num_episode)+'.'+str(population)
  controller_filebase = './controller/'+env_name+'.'+optimizer+'.'+str(num_episode)+'.'+str(population)

  model = make_model()

  num_params = model.param_count
  #print("size of model", num_params)

  if len(init_opt) > 0:
    es = pickle.load(open(init_opt, 'rb'))  
  else:
    if optimizer == 'ses':
      ses = PEPG(num_params,
        sigma_init=sigma_init,
        sigma_decay=sigma_decay,
        sigma_alpha=0.2,
        sigma_limit=0.02,
        elite_ratio=0.1,
        weight_decay=0.005,
        popsize=population)
      es = ses
    elif optimizer == 'ga':
      ga = SimpleGA(num_params,
        sigma_init=sigma_init,
        sigma_decay=sigma_decay,
        sigma_limit=0.02,
        elite_ratio=0.1,
        weight_decay=0.005,
        popsize=population)
      es = ga
    elif optimizer == 'cma':
      cma = CMAES(num_params,
        sigma_init=sigma_init,
        popsize=population)
      es = cma
    elif optimizer == 'pepg':
      pepg = PEPG(num_params,
        sigma_init=sigma_init,
        sigma_decay=sigma_decay,
        sigma_alpha=0.20,
        sigma_limit=0.02,
        learning_rate=0.01,
        learning_rate_decay=1.0,
        learning_rate_limit=0.01,
        weight_decay=0.005,
        popsize=population)
      es = pepg
    else:
      oes = OpenES(num_params,
        sigma_init=sigma_init,
        sigma_decay=sigma_decay,
        sigma_limit=0.02,
        learning_rate=0.01,
        learning_rate_decay=1.0,
        learning_rate_limit=0.01,
        antithetic=antithetic,
        weight_decay=0.005,
        popsize=population)
      es = oes

  PRECISION = 10000
  SOLUTION_PACKET_SIZE = (4+num_params)*num_worker_trial
  RESULT_PACKET_SIZE = 4*num_worker_trial
###

def sprint(*args):
  print(args) # if python3, can do print(*args)
  sys.stdout.flush()

class Seeder:
  def __init__(self, init_seed=0):
    np.random.seed(init_seed)
    self.limit = np.int32(2**31-1)
  def next_seed(self):
    result = np.random.randint(self.limit)
    return result
  def next_batch(self, batch_size):
    result = np.random.randint(self.limit, size=batch_size).tolist()
    return result

def encode_solution_packets(seeds, solutions, max_len=-1):

  n = len(seeds)
  result = []
  worker_num = 0
  for i in range(n):
    worker_num = int(i / num_worker_trial) + 1
    result.append([worker_num, i, seeds[i], max_len])
    result.append(np.round(np.array(solutions[i])*PRECISION,0))

  result = np.concatenate(result).astype(np.int32)
  result = np.split(result, num_worker)


  
  return result

def decode_solution_packet(packet):
  packets = np.split(packet, num_worker_trial)
  result = []
  for p in packets:
    result.append([p[0], p[1], p[2], p[3], p[4:].astype(np.float)/PRECISION])
  return result

def encode_result_packet(results):
  r = np.array(results)
  r[:, 2:4] *= PRECISION
  return r.flatten().astype(np.int32)

def decode_result_packet(packet):
  r = packet.reshape(num_worker_trial, 4)
  workers = r[:, 0].tolist()
  jobs = r[:, 1].tolist()
  fits = r[:, 2].astype(np.float)/PRECISION
  fits = fits.tolist()
  times = r[:, 3].astype(np.float)/PRECISION
  times = times.tolist()
  result = []
  n = len(jobs)
  for i in range(n):
    result.append([workers[i], jobs[i], fits[i], times[i]])
  return result

def worker(weights, seed, max_len, new_model):

  #print('WORKER working on environment {}'.format(new_model.env_name))

  new_model.set_model_params(weights)

  reward_list, t_list = simulate(
    new_model
    , num_episode=num_episode
    , seed=seed
    , max_len=max_len
    )

  if batch_mode == 'min':
    reward = np.min(reward_list)
  else:
    reward = np.mean(reward_list)
  t = np.mean(t_list)
  return reward, t

def follower():

  new_model = make_model()
  dream_model = make_model()
  
  while 1:
    #print('waiting for packet')
    packet = comm.recv(source=0)
    #comm.Recv(packet, source=0)
    current_env_name = packet['current_env_name']
    dream_mode = packet['dream_mode']

    packet = packet['result']

    
    assert(len(packet) == SOLUTION_PACKET_SIZE), (len(packet), SOLUTION_PACKET_SIZE)
    solutions = decode_solution_packet(packet)

    results = []
    
    if dream_mode:
      new_model.make_env(current_env_name + '_dream', model = dream_model)
    else:
      new_model.make_env(current_env_name)


    for solution in solutions:
      worker_id, jobidx, seed, max_len, weights = solution
      
      worker_id = int(worker_id)
      possible_error = "work_id = " + str(worker_id) + " rank = " + str(rank)
      assert worker_id == rank, possible_error
      jobidx = int(jobidx)
      seed = int(seed)
    
      fitness, timesteps = worker(weights, seed, max_len, new_model)
     
      results.append([worker_id, jobidx, fitness, timesteps])

    
    new_model.env.close()

    result_packet = encode_result_packet(results)
    assert len(result_packet) == RESULT_PACKET_SIZE
    comm.Send(result_packet, dest=0)


def send_packets_to_followers(packet_list, current_env_name, dream_mode):
  num_worker = comm.Get_size()
  assert len(packet_list) == num_worker-1
  for i in range(1, num_worker):
    packet = packet_list[i-1]
    assert(len(packet) == SOLUTION_PACKET_SIZE), (len(packet), SOLUTION_PACKET_SIZE)
    packet = {'result': packet, 'current_env_name': current_env_name, 'dream_mode': dream_mode}
    comm.send(packet, dest=i)

def receive_packets_from_followers():
  result_packet = np.empty(RESULT_PACKET_SIZE, dtype=np.int32)

  reward_list_total = np.zeros((population, 2))

  check_results = np.ones(population, dtype=np.int)
  for i in range(1, num_worker+1):
    comm.Recv(result_packet, source=i)
    results = decode_result_packet(result_packet)
    for result in results:
      worker_id = int(result[0])
      possible_error = "work_id = " + str(worker_id) + " source = " + str(i)
      assert worker_id == i, possible_error
      idx = int(result[1])
      reward_list_total[idx, 0] = result[2]
      reward_list_total[idx, 1] = result[3]
      check_results[idx] = 0

  check_sum = check_results.sum()
  assert check_sum == 0, check_sum
  return reward_list_total

def evaluate_batch(model_params, max_len):
  # duplicate model_params
  solutions = []
  for i in range(es.popsize):
    solutions.append(np.copy(model_params))

  seeds = np.arange(es.popsize)

  packet_list = encode_solution_packets(seeds, solutions, max_len=max_len)

  overall_rewards = []
  reward_list = np.zeros(population)

  for current_env_name in config.train_envs:
    send_packets_to_followers(packet_list, current_env_name, dream_mode = 0)
    packets_from_followers = receive_packets_from_followers()
    reward_list = packets_from_followers[:, 0] # get rewards
    overall_rewards.append(np.mean(reward_list))
    print(reward_list)
    print(overall_rewards)

  return np.mean(overall_rewards)


def leader():

  start_time = int(time.time())
  sprint("training", env_name)
  sprint("population", es.popsize)
  sprint("num_worker", num_worker)
  sprint("num_worker_trial", num_worker_trial)
  sprint("num_episode", num_episode)
  sprint("max_length", max_length)

  sys.stdout.flush()

  seeder = Seeder(seed_start)

  filename = filebase+'.json'
  filename_log = filebase+'.log.json'
  filename_hist = filebase+'.hist.json'
  filename_best = controller_filebase+'.best.json'
  filename_es = controller_filebase+'.es.pk'

  t = 0

  #if len(config.train_envs) == 1:
  current_env_name = config.train_envs[0]
  # model.make_env(current_env_name)

  history = []
  eval_log = []
  best_reward_eval = 0
  best_model_params_eval = None

  

  while True:
    
    t += 1

    solutions = es.ask()

    if antithetic:
      seeds = seeder.next_batch(int(es.popsize/2))
      seeds = seeds+seeds
    else:
      seeds = seeder.next_batch(es.popsize)

    packet_list = encode_solution_packets(seeds, solutions, max_len=max_length)

    reward_list = np.zeros(population)
    time_list = np.zeros(population)
    e_num = 1
    
    for current_env_name in config.train_envs:
      # print('before send packets')
      # tracker1 = SummaryTracker()
      send_packets_to_followers(packet_list, current_env_name, dream_mode)
      # print('between send and receive')
      # tracker1.print_diff()
      packets_from_followers = receive_packets_from_followers()
      # print('after receive')
      # tracker1.print_diff()
      reward_list = reward_list  + packets_from_followers[:, 0]
      time_list = time_list  + packets_from_followers[:, 1]
      if len(config.train_envs) > 1:
        print('completed environment {} of {}'.format(e_num, len(config.train_envs)))
      e_num += 1
      
    reward_list = reward_list / len(config.train_envs)
    time_list = time_list / len(config.train_envs)

    mean_time_step = int(np.mean(time_list)*100)/100. # get average time step
    max_time_step = int(np.max(time_list)*100)/100. # get max time step
    avg_reward = int(np.mean(reward_list)*100)/100. # get average reward
    std_reward = int(np.std(reward_list)*100)/100. # get std reward

    es.tell(reward_list)

    es_solution = es.result()
    model_params = es_solution[0] # best historical solution
    reward = es_solution[1] # best reward
    curr_reward = es_solution[2] # best of the current batch
    # model.set_model_params(np.array(model_params).round(4))

    r_max = int(np.max(reward_list)*100)/100.
    r_min = int(np.min(reward_list)*100)/100.

    curr_time = int(time.time()) - start_time

    h = (t, curr_time, avg_reward, r_min, r_max, std_reward, int(es.rms_stdev()*100000)/100000., mean_time_step+1., int(max_time_step)+1)

    if cap_time_mode:
      max_len = 2*int(mean_time_step+1.0)

    history.append(h)

    with open(filename, 'wt') as out:
      res = json.dump([np.array(es.current_param()).round(4).tolist()], out, sort_keys=True, indent=2, separators=(',', ': '))

    with open(filename_hist, 'wt') as out:
      res = json.dump(history, out, sort_keys=False, indent=0, separators=(',', ':'))

    pickle.dump(es, open(filename_es, 'wb'))

    sprint(env_name, h)
    # sprint(np.array(es.current_param()).round(4))
    # sprint(np.array(es.current_param()).round(4).sum())

    

    if (t == 1):
      best_reward_eval = avg_reward
    if (t % eval_steps == 0): # evaluate on actual task at hand

      prev_best_reward_eval = best_reward_eval
      model_params_quantized = np.array(es.current_param()).round(4)
      reward_eval = evaluate_batch(model_params_quantized, max_len=-1)
      model_params_quantized = model_params_quantized.tolist()
      improvement = reward_eval - best_reward_eval
      eval_log.append([t, reward_eval, model_params_quantized])
      with open(filename_log, 'wt') as out:
        res = json.dump(eval_log, out)
      if (len(eval_log) == 1 or reward_eval > best_reward_eval):
        best_reward_eval = reward_eval
        best_model_params_eval = model_params_quantized
      else:
        if retrain_mode:
          sprint("reset to previous best params, where best_reward_eval =", best_reward_eval)
          es.set_mu(best_model_params_eval)

      with open(filename_best, 'wt') as out:
        res = json.dump([best_model_params_eval, best_reward_eval], out, sort_keys=True, indent=0, separators=(',', ': '))
      
      sprint("improvement", t, improvement, "curr", reward_eval, "prev", prev_best_reward_eval, "best", best_reward_eval)


def main(args):
  global env_name, optimizer, init_opt, num_episode, eval_steps, max_length, num_worker, num_worker_trial, antithetic, seed_start, retrain_mode, dream_mode, cap_time_mode #, vae_version, rnn_version,
  env_name = args.env_name
  optimizer = args.optimizer
  init_opt = args.init_opt
  #vae_version = args.vae_version
  #rnn_version = args.rnn_version
  num_episode = args.num_episode
  eval_steps = args.eval_steps
  max_length = args.max_length

  num_worker = args.num_worker
  num_worker_trial = args.num_worker_trial
  antithetic = (args.antithetic == 1)
  retrain_mode = (args.retrain == 1)
  dream_mode = (args.dream_mode == 1)
  cap_time_mode= (args.cap_time == 1)
  seed_start = args.seed_start

  initialize_settings(args.sigma_init, args.sigma_decay, init_opt)

  sprint("process", rank, "out of total ", comm.Get_size(), "started")

  if (rank == 0):
    leader()
  else:
    follower()

def mpi_fork(n):
  """Re-launches the current script with workers
  Returns "parent" for original parent, "child" for MPI children
  (from https://github.com/garymcintire/mpi_util/)
  """
  if n<=1:
    return "child"
  if os.getenv("IN_MPI") is None:
    env = os.environ.copy()
    env.update(
      MKL_NUM_THREADS="1",
      OMP_NUM_THREADS="1",
      IN_MPI="1"
    )
    print( ["mpirun", "-np", str(n), sys.executable] + sys.argv)
    subprocess.check_call(["mpirun", "-np", str(n), sys.executable] +['-u']+ sys.argv, env=env)
    return "parent"
  else:
    global nworkers, rank
    nworkers = comm.Get_size()
    rank = comm.Get_rank()
    print('assigning the rank and nworkers', nworkers, rank)
    return "child"

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                'using pepg, ses, openes, ga, cma'))
  parser.add_argument('env_name', type=str, help='car_racing etc - this is only used for labelling files etc, the actual environments are defined in train_envs in config.py')
  parser.add_argument('-o', '--optimizer', type=str, help='ses, pepg, openes, ga, cma.', default='cma')
  parser.add_argument('--init_opt', type=str, default = '', help='which optimiser pickle file to initialise with')
  parser.add_argument('-e', '--num_episode', type=int, default=1, help='num episodes per trial (controller)')
  parser.add_argument('-n', '--num_worker', type=int, default=4)
  parser.add_argument('-t', '--num_worker_trial', type=int, help='trials per worker', default=1)
  parser.add_argument('--eval_steps', type=int, default=25, help='evaluate every eval_steps step')

  parser.add_argument('--max_length', type=int, help='maximum length of episode', default=-1)

  parser.add_argument('--antithetic', type=int, default=1, help='set to 0 to disable antithetic sampling')
  parser.add_argument('--cap_time', type=int, default=0, help='set to 0 to disable capping timesteps to 2x of average.')
  parser.add_argument('--retrain', type=int, default=0, help='set to 0 to disable retraining every eval_steps if results suck.\n only works w/ ses, openes, pepg.')
  parser.add_argument('-s', '--seed_start', type=int, default=111, help='initial seed')
  parser.add_argument('--sigma_init', type=float, default=0.1, help='sigma_init')
  parser.add_argument('--sigma_decay', type=float, default=0.999, help='sigma_decay')

  parser.add_argument('--dream_mode', type=int, help='train the agent in its dreams?', default=0)


  args = parser.parse_args()
  if "parent" == mpi_fork(args.num_worker+1): os.exit()
  main(args)

