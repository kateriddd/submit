Short explanation of the code organization:
The code libero_env_fast.py was used to set up the environment for all the experiments. 
**Note**: I had problems getting the dataset from hugging face, so I downloaded the Libero spatial 9 tasks
localy.
The code for Part 1 dense PPO are : train_dense_rl.py and the config file dense_ppo.yaml.
The code for Part 2: train_transformer_rl.py for all sub parts, for training and grpo fucntions. Uses the config transformer_rl.py
  2.1) Uses the PPO update function in train_dense_rl.py, the file grp_model.py for the pretrained HW1 
  model's functions and structure.
  2.2) Uses the grpo update and collection functions in train_dense_rl.py, the file grp_model.py for the pretrained HW1 
  model's functions and structure.
  2.3) Uses the grpo world model update and collection functions in train_dense_rl.py, the file grp_model.py for the pretrained HW1 
  model's functions and structure, and DreamerV3.py code from HW2 for the world model code.
  

**Note**: The dagger part was not implemented for bonus points.

**Commands to run the experiments reported in the report.**
**Part 1 experiments:**
PPO with seed 0:
python hw3/train_dense_rl.py \
	experiment.name=hw3_dense_ppo_seed0 \
	r_seed=0 \
	sim.task_set=libero_spatial \
	"sim.eval_tasks=[9]" \
	training.total_env_steps=100000 \
	training.rollout_length=128 \
	training.ppo_epochs=10 \
	training.minibatch_size=64

PPO with seed 1:
python hw3/train_dense_rl.py \
	experiment.name=hw3_dense_ppo_seed1 \
	r_seed=1 \
	sim.task_set=libero_spatial \
	"sim.eval_tasks=[9]" \
	training.total_env_steps=100000 \
	training.rollout_length=128 \
	training.ppo_epochs=10 \
	training.minibatch_size=64

PPO with seed 1500:
python hw3/train_dense_rl.py \
	experiment.name=hw3_dense_ppo_seed1500 \
	r_seed=1500 \
	sim.task_set=libero_spatial \
	"sim.eval_tasks=[9]" \
	training.total_env_steps=100000 \
	training.rollout_length=128 \
	training.ppo_epochs=10 \
	training.minibatch_size=64

**Part 2 experiments**
I was runnning on LightningAI, and the it was difficult to find any files if not using the complete path, 
so I am showing here exactly how I passed it. I uploaded my HW1 and Dreamer checkpoints in the same directory 
as where I was running the code.

Part 2 a):
PPO with seed 0 task 0
python hw3/train_transformer_rl.py\
  experiment.name=hw3_transformer_ppo_task0_seed0\
  r_seed=0 \
  sim.task_set=libero_spatial \
  "sim.eval_tasks=[0]" \
  training.total_env_steps=15000 \
  training.rollout_length=128 \
  training.ppo_epochs=4 \
  training.minibatch_size=64\
  init_checkpoint="/teamspace/studios/this_studio/robot_learning_2026/hw3/miniGRP.pth" \
  rl.algorithm=ppo


PPO with seed 0 task 9
python hw3/train_transformer_rl.py\
  experiment.name=hw3_transformer_ppo_task9_seed0\
  r_seed=0 \
  sim.task_set=libero_spatial \
  "sim.eval_tasks=[0]" \
  training.total_env_steps=15000 \
  training.rollout_length=128 \
  training.ppo_epochs=4 \
  training.minibatch_size=64\
  init_checkpoint="/teamspace/studios/this_studio/robot_learning_2026/hw3/miniGRP.pth" \
  rl.algorithm=ppo

PPO with seed 1500 task 9
python hw3/train_transformer_rl.py\
  experiment.name=hw3_transformer_ppo_seed1500_task9\
  r_seed=1500 \
  sim.task_set=libero_spatial \
  "sim.eval_tasks=[9]" \
  training.total_env_steps=15000 \
  training.rollout_length=128 \
  training.ppo_epochs=4 \
  training.minibatch_size=64\
  init_checkpoint="/teamspace/studios/this_studio/robot_learning_2026/hw3/miniGRP.pth" \
  rl.algorithm=ppo

	
PPO with seed 0 frozen
python hw3/train_transformer_rl.py\
  experiment.name=hw3_transformer_ppo_seed0_frozen\
  r_seed=0 \
  sim.task_set=libero_spatial \
  "sim.eval_tasks=[9]" \
  training.total_env_steps=15000 \
  training.rollout_length=128 \
  training.ppo_epochs=4 \
  training.minibatch_size=64\
  init_checkpoint="/teamspace/studios/this_studio/robot_learning_2026/hw3/miniGRP.pth" \
  rl.algorithm=ppo \
  model.policy.freeze=true


Part 2 b):

Grpo with seed 0 task 9
python hw3/train_transformer_rl.py\
  experiment.name=hw3_transformer_grpo_seed0\
  r_seed=0 \
  sim.task_set=libero_spatial \
  "sim.eval_tasks=[9]" \
  training.total_env_steps=15000 \
  training.rollout_length=128 \
  training.grpo_epochs=6 \
  training.minibatch_size=64\
  grpo.num_groups=2 \
  training.value.kl_coeff=0.01 \
  init_checkpoint="/teamspace/studios/this_studio/robot_learning_2026/hw3/miniGRP.pth" \
  rl.algorithm=grpo

Grpo with seed 1500 task 9
python hw3/train_transformer_rl.py\
  experiment.name=hw3_transformer_grpo_seed0\
  r_seed=0 \
  sim.task_set=libero_spatial \
  "sim.eval_tasks=[9]" \
  training.total_env_steps=15000 \
  training.rollout_length=128 \
  training.grpo_epochs=4 \
  training.minibatch_size=64\
  grpo.num_groups=4 \
  training.value.kl_coeff=0 \
  init_checkpoint="/teamspace/studios/this_studio/robot_learning_2026/hw3/miniGRP.pth" \
  rl.algorithm=grpo


Part 2 c):

Grpo with DreamerV3, seed 0 on task 9, episode length of 150
python hw3/train_transformer_rl.py\
  experiment.name=hw3_transformer_grpo_wm_task9_seed0\
  r_seed=0 \
  sim.task_set=libero_spatial \
  "sim.eval_tasks=[9]" \
  training.total_env_steps=15000 \
  training.rollout_length=128 \
  training.grpo_epochs=4 \
  training.minibatch_size=64 \
  sim.value.episode_length=150 \
  init_checkpoint="/teamspace/studios/this_studio/robot_learning_2026/hw3/miniGRP.pth" \
  wm_checkpoint="/teamspace/studios/this_studio/robot_learning_2026/hw3/dreamer_model.pth" \
  rl.algorithm=grpo \
  grpo.wm=true 

Grpo with DreamerV3, seed 0 on task 0, episode length of 150
python hw3/train_transformer_rl.py\
  experiment.name=hw3_transformer_grpo_wm_seed0_ep150\
  r_seed=0 \
  sim.task_set=libero_spatial \
  "sim.eval_tasks=[0]" \
  training.total_env_steps=15000 \
  training.rollout_length=128 \
  training.grpo_epochs=4 \
  training.minibatch_size=64 \
  sim.value.episode_length=150 \
  init_checkpoint="/teamspace/studios/this_studio/robot_learning_2026/hw3/miniGRP.pth" \
  wm_checkpoint="/teamspace/studios/this_studio/robot_learning_2026/hw3/dreamer_model.pth" \
  rl.algorithm=grpo \
  grpo.wm=true 

Grpo with DreamerV3, seed 0 on task 0, episode length of 300
python hw3/train_transformer_rl.py\
  experiment.name=hw3_transformer_grpo_wm_seed0_ep300\
  r_seed=0 \
  sim.task_set=libero_spatial \
  "sim.eval_tasks=[0]" \
  training.total_env_steps=15000 \
  training.rollout_length=128 \
  training.grpo_epochs=4 \
  training.minibatch_size=64 \
  sim.value.episode_length=300 \
  init_checkpoint="/teamspace/studios/this_studio/robot_learning_2026/hw3/miniGRP.pth" \
  wm_checkpoint="/teamspace/studios/this_studio/robot_learning_2026/hw3/dreamer_model.pth" \
  rl.algorithm=grpo \
  grpo.wm=true 

Grpo with DreamerV3, seed 0 on task 0, episode length of 300
python hw3/train_transformer_rl.py\
  experiment.name=hw3_transformer_grpo_wm_seed1337_ep100\
  r_seed=1337 \
  sim.task_set=libero_spatial \
  "sim.eval_tasks=[0]" \
  training.total_env_steps=15000 \
  training.rollout_length=128 \
  training.grpo_epochs=4 \
  training.minibatch_size=64 \
  sim.value.episode_length=100 \
  init_checkpoint="/teamspace/studios/this_studio/robot_learning_2026/hw3/miniGRP.pth" \
  wm_checkpoint="/teamspace/studios/this_studio/robot_learning_2026/hw3/dreamer_model.pth" \
  rl.algorithm=grpo \
  grpo.wm=true 
        
        
        
        
        
