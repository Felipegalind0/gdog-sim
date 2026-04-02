import genesis as gs
import argparse
import numpy as np
from jinja2 import Template
import tempfile
import os

def generate_random_robot_urdf():
    """Reads the Jinja template, randomizes parameters, and returns a path to a temporary URDF."""
    with open('gdog.urdf.jinja', 'r') as f:
        template = Template(f.read())

    # Generate Random Parameters (Domain Randomization)
    robot_params = {
        'body_length': np.random.uniform(0.3, 0.5),
        'body_width': np.random.uniform(0.2, 0.3),
        'body_height': np.random.uniform(0.08, 0.15),
        'body_mass': np.random.uniform(4.0, 7.0),
        'leg_thickness': np.random.uniform(0.04, 0.06),
        'thigh_length': np.random.uniform(0.18, 0.25),
        'calf_thickness': np.random.uniform(0.03, 0.05),
        'calf_length': np.random.uniform(0.18, 0.25),
        'wheel_radius': np.random.uniform(0.04, 0.08),
        'wheel_width': np.random.uniform(0.03, 0.06)
    }

    # Render URDF XML string
    urdf_string = template.render(**robot_params)

    # Write to a temporary file
    temp_urdf = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf", mode='w')
    temp_urdf.write(urdf_string)
    temp_urdf.close()

    print("Generated random robot with parameters:")
    for k, v in robot_params.items():
        print(f"  {k}: {v:.3f}")

    return temp_urdf.name

def main():
    parser = argparse.ArgumentParser(description="MVP Wheeled Robot Dog Simulator")
    parser.add_argument("--render", action="store_true", help="Enable interactive 3D viewer")
    parser.add_argument("--video", action="store_true", help="Record and save a video (mp4)")
    args = parser.parse_args()

    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer=args.render,
        sim_options=gs.options.SimOptions(dt=0.01),
        viewer_options=gs.options.ViewerOptions(
            res=(1920, 1080),
            camera_pos=(1.5, -1.5, 1.0),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=40,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            show_link_frame=False,
            show_cameras=False,
            plane_reflection=False, 
            ambient_light=(1, 1, 1),
        )
    )

    plane = scene.add_entity(gs.morphs.Plane())

    # --- Randomization Step ---
    urdf_path = generate_random_robot_urdf()
    
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
            pos=(0, 0, 0.5) # Raised higher to ensure wheels don't clip on larger random spawns
        )
    )

    scene.build()

    # Clean up the temporary file now that Genesis has parsed it
    os.remove(urdf_path)

    # Separate Leg DoFs from Wheel DoFs
    leg_dofs = []
    wheel_dofs = []
    standing_dof_pos = np.zeros(robot.n_dofs)
    
    for joint in robot.joints:
        dofs_idx = np.atleast_1d(joint.dofs_idx_local)
        if len(dofs_idx) == 1:
            idx = dofs_idx[0]
            
            if "wheel" in joint.name:
                wheel_dofs.append(idx)
            else:
                leg_dofs.append(idx)
                # Apply standing pose
                if "hip" in joint.name:
                    standing_dof_pos[idx] = 0.785
                elif "knee" in joint.name:
                    standing_dof_pos[idx] = -1.57

    leg_dofs = np.array(leg_dofs)
    wheel_dofs = np.array(wheel_dofs)
    target_leg_pos = standing_dof_pos[leg_dofs]

    # Initialize quaternion states for QPos
    qs = np.zeros(robot.n_qs)
    qs[0:3] = [0, 0, 0.5] 
    qs[3:7] = [1, 0, 0, 0] 
    
    for joint in robot.joints:
        dofs_idx = np.atleast_1d(joint.dofs_idx_local)
        qs_idx = np.atleast_1d(joint.qs_idx_local)
        if len(dofs_idx) == 1 and len(qs_idx) == 1:
            qs[qs_idx[0]] = standing_dof_pos[dofs_idx[0]]
            
    robot.set_qpos(qs)

    cam = None
    if args.video:
        cam = scene.add_camera(res=(1920, 1080), pos=(1.5, -1.5, 1.0), lookat=(0, 0, 0.3), fov=40, GUI=False)
        cam.start_recording()

    print("Starting simulation loop...")
    steps = 500 if args.video else 10000
    
    # Target velocity for wheels (Let's make it drive forward slowly)
    target_wheel_vel = np.array([5.0, 5.0, 5.0, 5.0]) 

    for i in range(steps):
        # 1. Maintain the standing position with the legs
        robot.control_dofs_position(target_leg_pos, dofs_idx_local=leg_dofs)
        
        # 2. Drive the wheels with velocity control
        robot.control_dofs_velocity(target_wheel_vel, dofs_idx_local=wheel_dofs)
        
        scene.step()
        
        if cam and i % 2 == 0: 
            cam.render()

    if args.video and cam:
        cam.stop_recording(save_to_filename='wheeled_go2.mp4', fps=50)
        print("Video saved to wheeled_go2.mp4")

if __name__ == "__main__":
    main()