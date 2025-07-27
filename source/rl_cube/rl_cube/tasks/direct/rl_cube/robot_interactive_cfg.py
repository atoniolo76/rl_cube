import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from .so_100_asset_cfg import SO_100_ASSET_CFG


@configclass
class RobotInteractiveCfg(InteractiveSceneCfg):
    # simulation
    sim = sim_utils.SimulationCfg(dt=1 / 60, render_interval=2)
    
    # scene
    num_envs = 1
    env_spacing = 2.0
    
    # robot
    robot = SO_100_ASSET_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")