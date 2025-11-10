#!/usr/bin/python3
import argparse
import time
from tqdm import tqdm
from controller import UR10e

parser = argparse.ArgumentParser(description="UR10e Sim2Real Controller")
# Robot connection
parser.add_argument(
    "--robot-ip",
    type=str,
    required=True,
    help="IP address of the UR10e robot"
)

# Model
parser.add_argument(
    "--model-path",
    type=str,
    required=True,
    help="Path to ONNX model file"
)

# Control parameters
parser.add_argument(
    "--control-freq",
    type=float,
    default=60.0,
    help="Control loop frequency in Hz (default: 60.0)"
)

parser.add_argument(
    "--acceleration",
    type=float,
    default=0.5,
    help="Joint acceleration limit in rad/s^2 (default: 0.5)"
)

# Target pose
parser.add_argument(
    "--target-position",
    type=float,
    nargs=3,
    default=[0.6, 0.0, 0.065],
    metavar=("X", "Y", "Z"),
    help="Target TCP position in meters (default: 0.7 0.0 0.0)"
)

parser.add_argument(
    "--target-orientation",
    type=float,
    nargs=4,
    default=[0.0, 1.0, 0.0, 0.0],
    # default=[-0, -0.70710678, 0.70710678, 0],
    metavar=("W", "X", "Y", "Z"),
    help="Target TCP orientation as quaternion (default: 1 0 0 0)"
)

parser.add_argument(
    "--joint-limits-soft-factor",
    type=float,
    default=0.9,
    help="Soft limit factor for joint limits (default: 0.9)"
)

# Execution parameters
parser.add_argument(
    "--max-steps",
    type=int,
    default=10000,
    help="Maximum number of control steps (default: 10000)"
)

parser.add_argument(
    "--max-time",
    type=float,
    default=60.0,
    help="Maximum execution time in seconds (default: 60.0)"
)

parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Enable debug mode (default: False)"
)

args = parser.parse_args()

def main():
    # Configuration
    robot_ip = args.robot_ip
    model_path = args.model_path
    control_freq = args.control_freq
    acceleration = args.acceleration
    target_position = args.target_position
    target_orientation = args.target_orientation
    joint_limits_soft_factor = args.joint_limits_soft_factor
    max_steps = args.max_steps
    max_time = args.max_time
    debug = args.debug
    dt = 1.0 / control_freq
    debug_interval = 10

    with UR10e(robot_ip=robot_ip, model_path=model_path, control_freq=control_freq, acceleration=acceleration, target_position=target_position, target_orientation=target_orientation, joint_limits_soft_factor=joint_limits_soft_factor, debug=debug) as robot:
        print("Resetting robot...")
        obs = robot.reset()

        if debug:
            print("Initial observations:", obs)

        print("Press any key to start the control loop...")
        input()  # Wait for user input

        print(f"Starting control loop at {control_freq} Hz")

        try:
            t_init = time.time()
            step = 0
            
            with tqdm(total=max_steps, desc="Control Loop", ncols=80) as pbar:
                while step < max_steps and (time.time() - t_init) < max_time:
                    t_loop_start = time.time()
                    t_curr = t_loop_start - t_init
                    debug_step = debug and (step % debug_interval == 0)
                    
                    # Read robot state
                    obs = robot.get_observations(debug=debug_step)
                    # Infer action from model
                    action = robot.get_action(obs, debug=debug_step)
                    # Send action to robot
                    robot.send_action(action, debug=debug_step)

                    if debug_step:
                        tqdm.write(f"t={t_curr:.2f}s | step={step} | action={action}")
                    
                    step += 1
                    pbar.update(1)
                    
                    # Sleep to maintain control frequency
                    t_elapsed = time.time() - t_loop_start
                    if t_elapsed < dt:
                        time.sleep(dt - t_elapsed)
                    else:
                        tqdm.write(f"⚠️ Loop time {t_elapsed:.4f}s exceeded target {dt:.4f}s")
                        
        except KeyboardInterrupt:
            print("\nStopping control loop...")

    print("Control loop stopped")

if __name__ == "__main__":
    main()
