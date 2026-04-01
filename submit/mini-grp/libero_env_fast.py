"""
Fast LIBERO environment - bypasses Robosuite wrapper for ~6000 SPS
"""
import hydra
import numpy as np
import gymnasium as gym
import os
import sys
from typing import Any, Mapping, cast
import h5py
import cv2


# Add LIBERO path
# 1. Point to your downloaded datasets
os.environ['LIBERO_DATASET_PATH'] = '/teamspace/studios/this_studio/robot_learning_2026/LIBERO/libero/datasets'

# 2. Point to the LIBERO code repository
sys.path.insert(0, '/teamspace/studios/this_studio/robot_learning_2026/LIBERO')
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import DenseRewardEnv, OffScreenRenderEnv

class FastLIBEROEnv:
    """
    Fast LIBERO using raw MuJoCo for stepping.
    Observation: robot proprio + simple object positions.
    Reward: dense distance-based reward.
    """
    
    def __init__(self, benchmark_name='libero_spatial', task_id=0, max_episode_steps=300,
                 num_sim_steps=1, action_repeat=1, render_mode=None, cfg=None,
                 output_image_obs=False, image_size=64, image_camera='agentview'):
        self.benchmark_name = benchmark_name
        self.task_id = task_id
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.prev_bowl_pos = None
        self.prev_bowl_rel = None
        self.render_mode = render_mode
        self._num_settle_steps = 10

        sim_cfg = getattr(cfg, "sim", None)
        cfg_output_image = bool(getattr(sim_cfg, "fast_env_output_image", False)) if sim_cfg is not None else False
        cfg_image_size = int(getattr(sim_cfg, "fast_env_image_size", image_size)) if sim_cfg is not None else image_size
        cfg_image_camera = str(getattr(sim_cfg, "fast_env_image_camera", image_camera)) if sim_cfg is not None else image_camera
        self.output_image_obs = bool(output_image_obs or cfg_output_image)
        self.image_size = int(cfg_image_size)
        self.image_camera = cfg_image_camera
        
        # Get task
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite_name = cfg.sim.task_set if cfg is not None else benchmark_name
        self.benchmark_name = task_suite_name
        task_suite = benchmark_dict[task_suite_name]()
        task = task_suite.get_task(task_id)
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        
        # Create base env
        enable_offscreen = (render_mode == 'rgb_array') or self.output_image_obs
        camera_size = self.image_size if self.output_image_obs else 256
        self.env = DenseRewardEnv(
            bddl_file_name=task_bddl_file,
            use_camera_obs=enable_offscreen,
            has_offscreen_renderer=enable_offscreen,
            has_renderer=False,
            camera_heights=camera_size,
            camera_widths=camera_size,
            # control_freq=5,  # Control frequency (Hz)
            # controller="OSC_POSE",
        )
        
        # Action space: 7DOF arm
        self._action_dim = 7
        self._num_sim_steps = num_sim_steps
        self._action_repeat = action_repeat
        
        # Get robot and object info
        self._setup()
        
        # Observation space: either proprio/object state vector or RGB image.
        self.obs_dim = 7 + 3 * len(self.target_objects)  # 7 + 3*2 = 13
        if self.output_image_obs:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(3, self.image_size, self.image_size), dtype=np.uint8
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-10, high=10, shape=(self.obs_dim,), dtype=np.float32
            )
        
        # Action space: 7DOF arm + 1 gripper (use 7 for now)
        self._action_dim = 7
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self._action_dim,), dtype=np.float32
        )
        
    def set_init_state(self, num_states):
        self.env.set_init_state(num_states)
        
    def _setup(self):
        """Extract object names and control range"""
        # Get inner env
        inner = self.env.env if hasattr(self.env, 'env') else self.env
        
        # Get robot
        self.robot = inner.robots[0]
        
        # Get target objects (first 2)
        self.target_objects = []
        # for name in inner.objects_dict.keys():
        #     if len(self.target_objects) >= 2:
        #         break
        self.target_objects.append("akita_black_bowl_1_main")
        self.target_objects.append("plate_1_main")
        
        # Get goal position (from task)
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.benchmark_name]()
        task = task_suite.get_task(self.task_id)
        self.goal_description = task.language
        self.goal_image = get_libero_goal_image(task_suite, self.task_id)        # Control range
        self.ctrlrange = self.env.sim.model.actuator_ctrlrange[:self._action_dim]
        
    def _get_state_obs(self, obs_dict=None):
        """State observation: [eef_pos(3), eef_quat_xyz(3), gripper_qpos(1), rel_obj_offsets(6)]."""
        if obs_dict is None:
            obs_getter = getattr(self.env, "_get_observations", None)
            if callable(obs_getter):
                obs_dict = obs_getter()
            else:
                raise RuntimeError("FastLIBEROEnv requires an observation dict from env.reset/env.step")

        obs_dict = cast(Mapping[str, Any], obs_dict)

        eef_pos = np.asarray(obs_dict["robot0_eef_pos"], dtype=np.float32)
        eef_quat_xyz = np.asarray(obs_dict["robot0_eef_quat"], dtype=np.float32)[:3]
        gripper_qpos = np.asarray([obs_dict["robot0_gripper_qpos"][0]], dtype=np.float32)

        sim = self.env.sim
        rel_obj_pos = []
        for name in self.target_objects:
            body_id = sim.model.body_name2id(name)
            obj_pos = sim.data.body_xpos[body_id]
            rel_obj_pos.append((obj_pos - eef_pos).astype(np.float32))

        state = np.concatenate([eef_pos, eef_quat_xyz, gripper_qpos] + rel_obj_pos).astype(np.float32)
        return state

    def _get_image_obs(self):
        """RGB image observation from simulator camera."""
        frame = self.render(camera_name=self.image_camera, width=self.image_size, height=self.image_size)

        if frame is None:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        if frame.shape[-1] > 3:
            frame = frame[:, :, :3]
            
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        return frame

    def _get_obs(self, obs_dict=None):
        if self.output_image_obs:
            return self._get_image_obs()
        return self._get_state_obs(obs_dict)
    
    def _reward(self, state, action):
        """Reusable reward function from action + observation vector.

        Expects the same layout as _get_obs:
            [qpos(7), bowl_rel_to_gripper(3), plate_rel_to_gripper(3), ...padding]
        """
        #Intuition: reward all the subgoals on the way to completing the task
        sim = self.env.sim
        #1. gripper to bowl 0.0 (far) to 1.0 (touching)
        eef_pos = state[:3]
        bowl_pos = sim.data.body_xpos[sim.model.body_name2id(self.target_objects[0])]
        dist_gripper_bowl = np.linalg.norm(eef_pos - bowl_pos)
        r_reach = 1.0 - np.tanh(dist_gripper_bowl * 3.0) 
        #2. grip bowl
        gripper_state = state[6]
        grip = (dist_gripper_bowl < 0.10) and (gripper_state < 0.05)
         # 2. Grasp Reward: Big to encourage success
        r_grasp = 2.0 if grip else 0.0
         #3. take bowl to plate
        plate_pos = sim.data.body_xpos[sim.model.body_name2id(self.target_objects[1])]
        dist_bowl_plate = np.linalg.norm(bowl_pos[:2] - plate_pos[:2])
        r_place = (1.0 - np.tanh(dist_bowl_plate * 3.0)) if grip else 0.0  
        #4. put bowl in plate, gripper let go
        height_close = abs(bowl_pos[2] - plate_pos[2]) < 0.05
        is_placed = (dist_bowl_plate < 0.06 and height_close)
        r_success = 10.0 if is_placed else 0.0
    
        reward = r_reach + r_grasp + r_place + r_success 
        return reward, {
            'r_reach': r_reach, 'r_grasp': r_grasp, 'r_place': r_place,
            'r_success': r_success, 'dist_gripper_bowl': dist_gripper_bowl,
            'is_grasping': float(grip), 'is_placed': float(is_placed)
        }

         #OLD NEGATIVE REWARD FUNCTION USED FOR PART 1
        # #Intuition: reward all the subgoals on the way to completing the task.
        # sim = self.env.sim
        # #get the position of the plate and the bowl,
        # plate_pos = sim.data.body_xpos[sim.model.body_name2id(self.target_objects[1])]
    
        #     #1. gripper to bowl
        # eef_pos = state[:3]     #center coordinates of gripper state
        # bowl_pos = sim.data.body_xpos[sim.model.body_name2id(self.target_objects[0])]   #bowl pos
        # dist_gripper_bowl = np.linalg.norm(eef_pos - bowl_pos)
        # r_reach = -0.1 * np.tanh(dist_gripper_bowl * 5.0)                               #Penalize the distance between gripper to bowl (lead the gripper to the bowl)
        #     #2. grip bowl
        # gripper_state = state[6]                                                        #robot fingers, closed or open for grip verif
        # grip = (dist_gripper_bowl < 0.10) and (gripper_state < 0.05)                    #close to bowl and griped
        # r_grasp = 1.0 if grip else 0.0  #reward if grip
        #     #3. take bowl to plate
        # dist_bowl_plate = np.linalg.norm(bowl_pos[:2] - plate_pos[:2])
        # r_place = -0.1 * np.tanh(dist_bowl_plate * 5.0) if grip else 0.0                # panalize the distance between bowl and plate
        #     #4. put bowl in plate, gripper let go
        # height_close = abs(bowl_pos[2] - plate_pos[2]) < 0.05                           #bowl and plate y coordinate is almost same == placed
        # is_placed = (dist_bowl_plate < 0.06 and height_close)                           #did it let go?
        # r_success = 5.0 if is_placed else 0.0                                           #success bonus for placing it                                    
        
        # reward = r_reach + r_grasp + r_place + r_success
        
        # return reward, {
        #     'r_reach': r_reach, 'r_grasp': r_grasp, 'r_place': r_place,
        #     'r_success': r_success, 'dist_gripper_bowl': dist_gripper_bowl,
        #     'is_grasping': float(grip), 'is_placed': float(is_placed)
        # }

    def _compute_reward(self, action, state=None):
        """Dense reward computed from the same vector returned by _get_obs."""
        if state is None:
            raise RuntimeError("_compute_reward requires an explicit state when using FastLIBEROEnv")
        reward, reward_info = self._reward(state, action)

        return reward, reward_info

    def _compute_init_distance(self):
        """Compute initial distance between bowl and plate"""
        sim = self.env.sim
        try:
            init_bowl = sim.data.body_xpos[sim.model.body_name2id(self.target_objects[0])]
            init_plate = sim.data.body_xpos[sim.model.body_name2id(self.target_objects[1])]
            self.init_bowl_plate_dist = np.linalg.norm(init_bowl - init_plate)
        except:
            self.init_bowl_plate_dist = 0.1  # default
    
    def reset(self, seed=None, options=None):
        self.current_step = 0
        obs_dict = self.env.reset()

        init_state = None
        if options is not None:
            init_state = options.get("init_state", None)
        if init_state is not None:
            obs_dict = self.env.set_init_state(init_state)

        # Compute initial distance
        self._compute_init_distance()

        # Warmup: apply no-op actions to let objects settle
        for _ in range(self._num_settle_steps):
            obs_dict, _, done, _ = self.env.step([0, 0, 0, 0, 0, 0, -1])
            if done:
                break

        state = self._get_state_obs(obs_dict)
        # Update previous bowl-relative position for next step
        self.prev_bowl_rel = state[7:10].copy()

        obs = self._get_image_obs() if self.output_image_obs else state
        info = {"state_obs": state.copy()}
        return obs, info
    
    def step(self, action):
        action = np.clip(action, -1, 1)        
        
        obs_dict = None
        done = False
        info_env = {}
        for _ in range(max(1, self._num_sim_steps)):
            obs_dict, _, done, info_env = self.env.step(action.tolist())
            if done:
                break
        
        # Compute reward
        state = self._get_state_obs(obs_dict)
        reward, reward_info = self._reward(state, action)
        
        # Check success metrics (regardless of reward function)
        sim = self.env.sim
        gripper_pos = state[:3]
        
        try:
            bowl_pos = sim.data.body_xpos[sim.model.body_name2id(self.target_objects[0])]
        except:
            bowl_pos = np.zeros(3)
        
        try:
            plate_pos = sim.data.body_xpos[sim.model.body_name2id(self.target_objects[1])]
        except:
            plate_pos = np.zeros(3)
        
        # Success 1: Gripper close to bowl (grasping) - threshold 4cm
        dist_gripper_bowl = np.linalg.norm(gripper_pos - bowl_pos)
        success_grasping = 1.0 if dist_gripper_bowl < 0.04 else 0.0
        
        # Success 2: Bowl on plate - threshold 5cm horizontal, similar height
        dist_bowl_plate = np.linalg.norm(bowl_pos[:2] - plate_pos[:2])  # x,y only
        height_close = abs(bowl_pos[2] - plate_pos[2]) < 0.05
        success_placed = 1.0 if (dist_bowl_plate < 0.05 and height_close and bowl_pos[2] > plate_pos[2]) else 0.0
        
        # Add to info
        reward_info['success_grasping'] = float(success_grasping)
        reward_info['success_placed'] = float(success_placed)
        
        # Check termination
        self.current_step += 1
        done = False
        truncated = self.current_step >= self.max_episode_steps
        if truncated:
            done = True
            
        info = dict(info_env)
        info.update(reward_info)
        info['state_obs'] = state.copy()

        obs = self._get_image_obs() if self.output_image_obs else state
        
        return obs, reward, done, truncated, info
    
    def close(self):
        self.env.close()

    def render(self, camera_name='agentview', width=256, height=256):
        if not (self.render_mode == 'rgb_array' or self.output_image_obs):
            return None

        try:
            frame = self.env.sim.render(camera_name=camera_name, width=width, height=height)
            if frame is None:
                return None
            frame = np.asarray(frame)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            return frame[::-1, :, :]
        except Exception:
            return None
    
    @property
    def unwrapped(self):
        return self

import datasets


def get_libero_goal_image(task_suite, task_id, target_size=(64, 64)):
    """
    Gets the goal image (last frame in the sample) for the HW1 model that uses it.
    """
    try:
        # If the HF stream fails, need a fallback that isn't just zeros.
        # Try to get the first frame of the current environment as a 'temp' goal to avoid crash
        ds = datasets.load_dataset("gberseth/libero_spatial_sequence", split='train', streaming=True)
        for sample in ds:
            raw_frame = np.array(sample['img'][-1])
            if raw_frame.size > 0:
                # Ensure the frame is valid before resizing
                return cv2.resize(raw_frame, target_size).transpose(2, 0, 1) # Match C(), H, W)
    except Exception as e:
        print(f"Goal Image Error: {e}. Using dummy (3, 64, 64).")
        return np.zeros((3, target_size[0], target_size[1]), dtype=np.float32)

if __name__ == '__main__':
    import time
    
    env = FastLIBEROEnv()
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}")
    print(f"Action shape: {env.action_space.shape}")
    
    # Test speed
    start = time.time()
    episodes = 0
    steps = 0
    for i in range(500):
        action = env.action_space.sample()
        obs, reward, done, truncated, _ = env.step(action)
        steps += 1
        if done:
            env.reset()
            episodes += 1
    elapsed = time.time() - start
    
    print(f"FastLIBERO: {steps/elapsed:.0f} SPS, {episodes} episodes")
    env.close()
