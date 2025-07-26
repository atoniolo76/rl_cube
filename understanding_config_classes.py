### below code is the configuation class


## to be more specific, let's call the environment class that has an instance of this config class as a attribute the "RL environment", and the training environments that the config class will spin up in parrallel as "training environments'"
class TutorialRLEnvConfig(DirectRLConfig):
    def __init__(self):
        super().__init__()
    
    sim_cfg: SimulationConfig = SimulationConfig(dt=1 / 120, render_interval=2)

    robot_cfg: ArticulationCfg = CartpoleConfig.replace(prim_path="/World/envs/env_.*/robot")

    # if replicate_physics is True, then each environment will have isolated physics, otherwise they will share the same physics and potentially collide with each other unless you specify adequate environment spacingn
    scene_cfg: InteractiveSceneCfg = InteractiveSceneCfg.replace(num_envs=4096, env_spacing = 4.0, env_start_position = [0, 0, 0.5], replicate_physics = True)


### below is the actual environment class
from .isaac_lab_tutorial_env_cfg import IsaacLabTutorialEnvCfg


class TutorialRLEnv(DirectRLEnv):
    # this is a type annotation
    cfg: TutorialRLEnvConfig

    def _init_(self, cfg: TutorialRLEnvConfig, sim_device: str, graphics_device_id: int, headless: bool):
        super().__init__(cfg, sim_device, graphics_device_id, headless)
    
    def setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        self.scene.articulations["robot"] = self.robot

    def apply_action(self, actions: torch.Tensor):
        self.robot.set_joint_efforts(actions, indices=self.robot.joint_indices)

    def get_observations(self):
        return {
            "robot": self.robot.get_joint_positions(),
        }

    def _get_dones(self):
        return {
            "episode_timeout": self.episode_length_buf >= self.max_episode_length - 1,
            "task_success": torch.abs(self.robot.get_joint_positions()[:, 0]) > 0.8,
        }
    
    def reset_idx(self, env_ids: torch.Tensor):
        super().reset_idx(env_ids)
        self.robot.set_joint_positions(torch.zeros_like(self.robot.get_joint_positions()), env_ids=env_ids)
        self.robot.set_joint_velocities(torch.zeros_like(self.robot.get_joint_velocities()), env_ids=env_ids)
        self.robot.set_joint_efforts(torch.zeros_like(self.robot.get_joint_efforts()), env_ids=env_ids)
        self.robot.set_joint_positions(torch.zeros_like(self.robot.get_joint_positions()), env_ids=env_ids)
    

@torch.jit.script
def compute_rewards(...):
    . . .
    return total_reward



# note the difference between continuous and discrete RL tasks: continuous tasks are default alive: a cartpole (balancing) is alive, a lunar lander (navigation) is dead.

# in the cartpole example, the environment negatively rewards the agent when deviating from the center of the pole. if the pole hits the ground, the episode is terminated.