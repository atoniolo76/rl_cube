"""Script to run the robot interactive scene."""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Robot Interactive Scene")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch the simulation app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows AFTER AppLauncher initialization."""

import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from rl_cube.tasks.direct.rl_cube.so_100_asset_cfg import SO_100_ASSET_CFG


def main():
    """Main function."""
    
    # Create a basic scene configuration
    scene_cfg = InteractiveSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    
    # Create scene
    scene = InteractiveScene(scene_cfg)
    
    # Manually add the robot
    robot_cfg = SO_100_ASSET_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot = Articulation(robot_cfg)
    scene.articulations["robot"] = robot
    
    # Setup ground plane
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    
    # Add lights
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    
    # Clone and replicate
    scene.clone_environments(copy_from_source=False)
    scene.filter_collisions(global_prim_paths=[])
    
    # Play simulation
    sim_cfg = sim_utils.SimulationCfg(dt=1/60, render_interval=2)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.reset()
    
    print("[INFO]: Starting interactive scene...")
    
    # Simulation variables
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            count = 0
            # reset the scene entities
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_state_to_sim(root_state)
            
            joint_pos = scene["robot"].data.default_joint_pos.clone()
            joint_vel = scene["robot"].data.default_joint_vel.clone()
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")

        # wave action - move first 4 joints
        joint_pos_target = scene["robot"].data.default_joint_pos.clone()
        joint_pos_target[:, 0:4] = 0.25 * np.sin(2 * np.pi * 0.5 * sim_time)
        scene["robot"].set_joint_position_target(joint_pos_target)

        # write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()