import argparse
import time
import numpy as np
import rtde_control
import rtde_receive
import onnxruntime as ort
from pathlib import Path
from scipy.spatial.transform import Rotation

class UR10e:
    """Controller for running RL policies on UR10e robot"""
    
    def __init__(
        self,
        robot_ip: str,
        model_path: str,
        control_freq: float = 125.0,
        acceleration: float = 0.5,
        home_joint_positions: np.ndarray = None,
        target_position: np.ndarray = None,
        target_orientation: np.ndarray = None,
        joint_limits_soft_factor: float = 1.0,
        clip_observations: float = 100.0,
        clip_actions: float = 1.0,
        debug: bool = False,
    ):
        """
        Initialize UR10e controller.
        
        Args:
            robot_ip: IP address of the robot
            model_path: Path to ONNX model file
            control_freq: Control loop frequency in Hz
            acceleration: Joint acceleration limit in rad/s^2
            target_position: Target TCP position [x, y, z] in meters
            target_orientation: Target TCP orientation as quaternion [w, x, y, z]
        """
        self.clip_observations = clip_observations
        self.clip_actions = clip_actions
        
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        self.acceleration = acceleration
        
        # Target pose
        self.target_position = target_position if target_position is not None else np.array([0.76177, 0.14285, 0.065  ])
        self.target_orientation = target_orientation if target_orientation is not None else np.array([ 0.0000003 ,  0.6292    , -0.77724   , -0.00000048])
        
        # UR10e specifications
        self.joint_limits = np.array([
            [-6.283185307179586, 6.283185307179586],
            [-6.283185307179586, 6.283185307179586],
            [-3.141592653589793, 3.141592653589793],
            [-6.283185307179586, 6.283185307179586],
            [-6.283185307179586, 6.283185307179586],
            [-6.283185307179586, 6.283185307179586],
        ])
        self.joint_limits_soft_factor = joint_limits_soft_factor
        self.joint_limits_soft = self.joint_limits * self.joint_limits_soft_factor
        
        self.velocity_limits = np.array([
            2.0944,  # shoulder: 120 deg/s
            2.0944,  # shoulder: 120 deg/s
            3.1416,  # elbow: 180 deg/s
            3.1416,  # wrist: 180 deg/s
            3.1416,  # wrist: 180 deg/s
            3.1416,  # wrist: 180 deg/s
        ])

        # Home position (default: all joints at 0)
        self.home_joint_positions = (
            home_joint_positions if home_joint_positions is not None 
            else np.array([0.0, -1.5707963267948966, 1.5707963267948966, -1.5707963267948966, -1.5707963267948966, 0.0])  # Common "ready" pose
        )
        
        self.debug = debug

        if self.debug:
            print(f"Controller initialized:")
            print(f"  - Robot IP: {robot_ip}")
            print(f"  - Control frequency: {control_freq} Hz")
            print(f"  - Acceleration: {acceleration} rad/s^2")
            print(f"  - Target position: {target_position}")
            print(f"  - Target orientation (quat): {target_orientation}")
            print(f"  - Debug mode: {debug}")

        # Load Model
        self.__load_model(model_path)
        self.__connect(robot_ip)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure cleanup even on crashes"""
        self.disconnect()
        return False
    
    def __connect(self, robot_ip: str):
        """Connect to the robot"""
        self.robot_ip = robot_ip
        print(f"\nConnecting to robot at {self.robot_ip}...")
        
        try:
            self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)
            print(f"✓ Connected to robot")
            
            # Get current position
            if self.debug:
                current_q = self.rtde_r.getActualQ()
                print(f"\nCurrent joint positions: {np.array(current_q)}")
            
        except Exception as e:
            if hasattr(self, 'rtde_c') and self.rtde_c:
                self.rtde_c.disconnect()
            if hasattr(self, 'rtde_r') and self.rtde_r:
                self.rtde_r.disconnect()
            raise RuntimeError(f"Failed to connect to robot: {e}")
        
    def reset(self):
        """Move the robot to the home position."""
        if self.rtde_c:
            success = self.rtde_c.moveJ(self.home_joint_positions.tolist(), speed=1.0, acceleration=1.0, asynchronous=False)
            if success:
                print("✓ Reached home position")
                if self.debug:
                    final_q = self.rtde_r.getActualQ()
                    print(f"  Current joint positions: {np.array(final_q)}")
            else:
                print("⚠ Warning: moveJ returned False - movement may not have completed")
        return self.get_observations()

    def disconnect(self):
        """Disconnect from the robot"""
        if self.rtde_c:
            self.rtde_c.speedStop()
            time.sleep(0.5)
            self.rtde_c.stopScript()
            self.rtde_c.disconnect()
        if self.rtde_r:
            self.rtde_r.disconnect()
        print("✓ Disconnected from robot")
    
    def __load_model(self, model_path: str):
        """Load ONNX model"""
        self.model_path = model_path
        try:
            self.model = ort.InferenceSession(self.model_path)
            self.input_name = self.model.get_inputs()[0].name
            self.output_name = self.model.get_outputs()[0].name
            input_shape = self.model.get_inputs()[0].shape
            output_shape = self.model.get_outputs()[0].shape
            self.obs_dim = input_shape[1]
            self.action_dim = output_shape[1] if len(output_shape) > 1 else 1
            self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
            print(f"✓ Model {self.model_path} loaded successfully!")
            if self.debug:
                print(f"  - {self.input_name} dim: {self.obs_dim}")
                print(f"  - {self.output_name} dim: {self.action_dim}")
        except:
            print(f"Failed to load model")
    
    def __quat_mul(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1,x1,y1,z1 = q1
        w2,x2,y2,z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    def __rotvec_to_quat(self, rvec: np.ndarray) -> np.ndarray:
        theta = np.linalg.norm(rvec)
        if theta < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis = rvec / theta
        half = 0.5 * theta
        w = np.cos(half)
        xyz = np.sin(half) * axis
        return np.concatenate([[w], xyz])
    
    def __normalize_joint_positions(self, joint_positions: np.ndarray) -> np.ndarray:
        """Normalize joint positions to [-1, 1] based on soft limits."""
        lower = self.joint_limits_soft[:, 0]
        upper = self.joint_limits_soft[:, 1]
        offset = (lower + upper) * 0.5
        normalized = 2 * (joint_positions - offset) / (upper - lower)
        return normalized
    
    def __quat_conjugate(self, q):
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def __dh_to_urdf(self, tcp_pose_real):
        """
        Convert real robot TCP (DH convention) to URDF convention.
        Real robot uses: getActualTCPPose() with standard base, TCP active.
        Returns: [x, y, z, qw, qx, qy, qz]
        """
        pos = np.array(tcp_pose_real[:3])
        rot_vec = np.array(tcp_pose_real[3:])
        
        # Position: flip X and Y signs
        urdf_pos = np.array([-pos[0], -pos[1], pos[2]])
        
        # Orientation: apply Rz(180°) transformation
        R_real = Rotation.from_rotvec(rot_vec)
        R_transform = Rotation.from_euler('z', 180, degrees=True)
        R_urdf = R_transform * R_real
        
        # Convert to quaternion
        urdf_quat = Rotation.as_quat(R_urdf, scalar_first=True)  # [w, x, y, z]
        
        return np.concatenate([urdf_pos, urdf_quat])


    def __urdf_to_dh(self, urdf_pose):
        """
        Convert URDF convention to real robot TCP (DH convention).
        Input: [x, y, z, qw, qx, qy, qz]
        Returns: [x, y, z, rx, ry, rz]
        """
        pos = np.array(urdf_pose[:3])
        quat = np.array(urdf_pose[3:])  # [w, x, y, z]
        
        # Position: flip X and Y signs
        real_pos = np.array([-pos[0], -pos[1], pos[2]])
        
        # Orientation: apply Rz(-180°) transformation
        R_urdf = Rotation.from_quat(quat, scalar_first=True)
        R_transform = Rotation.from_euler('z', -180, degrees=True)
        R_real = R_transform * R_urdf
        
        real_rot_vec = R_real.as_rotvec()
        
        return np.concatenate([real_pos, real_rot_vec])

    def get_observations(self, debug: bool = False) -> np.ndarray:
        """
        Get observations from the robot.
        
        Returns:
            observation vector as numpy array with shape (obs_dim,)
        """
        # Propioception
        # - Joint Positions Normalized
        joint_pos = np.array(self.rtde_r.getActualQ())
        joint_pos_normalized = self.__normalize_joint_positions(joint_pos)
        
        # - Joint Velocities
        joint_vel = np.array(self.rtde_r.getActualQd())
        
        # - TCP pose (w.r.t robot base) XYZ + quaternion (w, x, y, z)
        tcp_pose_dh = np.array(self.rtde_r.getActualTCPPose())  # [x, y, z, rx, ry, rz] in DH frame
        if self.debug and debug:
            print(f"TCP pose (DH/UR frame): {tcp_pose_dh}")
        # Convert DH to URDF (returns [x, y, z, qw, qx, qy, qz])
        tcp_pose_urdf = self.__dh_to_urdf(tcp_pose_dh)
        tcp_position_base = tcp_pose_urdf[:3]
        tcp_rotation_base = tcp_pose_urdf[3:]  # [w, x, y, z]

        if self.debug and debug:
            print(f"TCP position (URDF frame): {tcp_position_base}")
            print(f"TCP orientation (quat, URDF frame): {tcp_rotation_base}")
        
        # - Target TCP pose XYZ + quaternion (w, x, y, z)
        target_position = self.target_position
        target_orientation = self.target_orientation
        
        # - Object pose XYZ + quaternion (w, x, y, z)
        object_position_base = self.target_position
        object_rotation_base = self.target_orientation
        
        # - Previous actions
        prev_actions = self.prev_action  # 6
        
        # Stack all observations
        observation = np.concatenate([
            joint_pos_normalized,    # 6
            joint_vel,              # 6
            tcp_position_base,      # 3
            tcp_rotation_base,      # 4 (w, x, y, z)
            target_position,        # 3
            target_orientation,     # 4 (w, x, y, z)
            # object_position_base,   # 3
            # object_rotation_base,   # 4 (w, x, y, z)
            prev_actions,           # 6
        ], dtype=np.float32)  # Total: 39 values
        
        if self.debug and debug:
            print(f"Observations:")
            print(f"  Joint pos (norm): {joint_pos_normalized}")
            print(f"  Joint vel: {joint_vel}")
            print(f"  TCP pos: {tcp_position_base}")
            print(f"  TCP rot (quat): {tcp_rotation_base}")
            print(f"  Target pos: {target_position}")
            print(f"  Target rot (quat): {target_orientation}")
            print(f"  Prev actions: {prev_actions}")
            # Quaternion difference
            q_error = self.__quat_mul(self.__quat_conjugate(tcp_rotation_base), target_orientation)
            angle_error = 2 * np.arccos(np.clip(q_error[0], -1, 1))
            print(f"  Orientation error: {np.degrees(angle_error):.1f}°")

        return observation

    def get_action(self, observation: np.ndarray, debug: bool = False) -> np.ndarray:
        """
        Run inference to get action from policy.
        
        Args:
            observation: Current observation vector (obs_dim,)
            
        Returns:
            action vector (action_dim,)
        """
        # Add batch dimension for ONNX
        if observation.ndim == 1:
            observation = observation[np.newaxis, :]
        
        # Clip observations
        observation_clipped = np.clip(observation, -self.clip_observations, self.clip_observations)
        
        # Run ONNX inference
        action = self.model.run([self.output_name], {self.input_name: observation_clipped})[0]
        
        # Clip actions and remove batch dimension
        action_clipped = np.clip(action, -self.clip_actions, self.clip_actions)
        action_output = action_clipped.squeeze(0)
        
        # Store this action as prev_action for next step
        self.prev_action = action_output.copy()

        # return np.zeros_like(action_output)
        if self.debug and debug:
            print(f"Action (raw): {action}")
            print(f"Action (clipped): {action_output}")
        
        return action_output
    
    def send_action(self, action: np.ndarray, debug: bool = False):
        """
        Send action (joint velocities) to the robot.
        Action comes from policy in [-1, 1] range.
        We need to scale it like JointVelocityActionCfg does.
        """
        if action.ndim != 1 or len(action) != 6:
            raise ValueError(f"Action must be (6,), got shape {action.shape}")

        # Scale actions to velocity targets (same as Isaac)
        action_scales = np.array([
            2.0,  # shoulder_pan_joint: [-2.0, 2.0] rad/s
            2.0,  # shoulder_lift_joint: [-2.0, 2.0] rad/s
            3.0,  # elbow_joint: [-3.0, 3.0] rad/s
            3.0,  # wrist_1_joint: [-3.0, 3.0] rad/s
            3.0,  # wrist_2_joint: [-3.0, 3.0] rad/s
            3.0,  # wrist_3_joint: [-3.0, 3.0] rad/s
        ])
        velocity_targets = action * action_scales
        
        # Get current joint positions
        current_q = np.array(self.rtde_r.getActualQ())
        
        # Integrate velocity to get position target (Euler integration)
        # This is what PhysX does internally at each timestep
        position_target = current_q + velocity_targets * self.dt

        # Clip to joint limits for safety
        position_target = np.clip(
            position_target,
            self.joint_limits[:, 0],
            self.joint_limits[:, 1]
        )
        
        if self.debug and debug:
            print(f"Sending servoJ:")
            print(f"  Velocity targets: {velocity_targets}")
            print(f"  Current q: {current_q}")
            print(f"  Target q: {position_target}")
            current_vel = np.array(self.rtde_r.getActualQd())
            vel_error = velocity_targets - current_vel
            print(f"Velocity error: {vel_error}")

        # servoJ acts like PhysX's implicit controller
        # It computes torques internally to reach position_target
        self.rtde_c.servoJ(
            position_target.tolist(),  # q: target joint positions
            0.5,                       # speed: NOT used (set to any value)
            self.acceleration,         # acceleration: NOT used (but set it anyway)
            self.dt,                   # time: blocking time in seconds (0.008 for e-series)
            0.1,                      # lookahead_time: smoothing [0.03-0.2]
            150                        # gain: P-gain [100-2000]
        )

