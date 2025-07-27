import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

SO_100_ASSET_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="./so_100.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "Rotation": 0.0,
            "Pitch": 0.0,
            "Elbow": 0.0,
            "Wrist_Pitch": 0.0,
            "Wrist_Roll": 0.0,
            "Jaw": 0.0,
        },
        pos=(0.25, -0.25, 0.0),
    ),
    actuators={
        "Rotation": ImplicitActuatorCfg(
            joint_names_expr=["Rotation"],
            stiffness=4000.0,  # Base rotation - needs higher stiffness
            damping=175.0,     # 2 * 0.7 * sqrt(4000 * 1.0) ≈ 175
        ),
        "Pitch": ImplicitActuatorCfg(
            joint_names_expr=["Pitch"],
            stiffness=3000.0,  # Shoulder pitch
            damping=152.0,     # 2 * 0.7 * sqrt(3000 * 1.0) ≈ 152
        ),
        "Elbow": ImplicitActuatorCfg(
            joint_names_expr=["Elbow"],
            stiffness=2000.0,  # Elbow - lighter load
            damping=125.0,     # 2 * 0.7 * sqrt(2000 * 1.0) ≈ 125
        ),
        "Wrist_Pitch": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Pitch"],
            stiffness=1000.0,  # Wrist joints - lighter
            damping=88.0,      # 2 * 0.7 * sqrt(1000 * 1.0) ≈ 88
        ),
        "Wrist_Roll": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Roll"],
            stiffness=1000.0,
            damping=88.0,
        ),
        "Jaw": ImplicitActuatorCfg(
            joint_names_expr=["Jaw"],
            stiffness=500.0,   # Gripper - very light
            damping=62.0,      # 2 * 0.7 * sqrt(500 * 1.0) ≈ 62
        ),
    },
)