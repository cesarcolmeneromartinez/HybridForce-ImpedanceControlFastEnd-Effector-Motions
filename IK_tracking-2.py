import pinocchio as pin
import numpy as np
from quadprog import solve_qp  # Install with `pip install quadprog`

# Load the robot model
urdf_path = "path/to/your/robot.urdf"  # Replace with the path to your URDF file
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

# Define a time-based trajectory as a list of SE(3) poses
time_steps = np.linspace(0, 5, 50)  # 5 seconds, 50 steps
trajectory = [
    pin.SE3(np.eye(3), np.array([0.5 + 0.1 * np.sin(t), 0.2, 0.3]))  # Example trajectory
    for t in time_steps
]

# Joint limits
q_min = model.lowerPositionLimit  # Replace with your robot's joint limits
q_max = model.upperPositionLimit   # Replace with your robot's joint limits

# End-effector frame index (modify based on your URDF)
end_effector_frame = model.getFrameId("ee_name_here")  # Replace with your end-effector frame name

# Initialize joint configuration
q = pin.neutral(model)  # Starting at the neutral configuration
max_iterations = 100
tolerance = 1e-4
damping = 1e-6  # Regularization factor for stability

# Initialize storage for joint configurations and errors
q_trajectory = []
pose_errors = []

# Loop through each desired pose in the trajectory
for desired_pose in trajectory:
    for i in range(max_iterations):
        # Compute forward kinematics and Jacobian
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pin.computeJointJacobians(model, data, q)

        # Current end-effector pose
        current_pose = data.oMf[end_effector_frame]

        # Error in SE(3) as a 6D vector (translation + rotation)
        error = pin.log6(desired_pose.inverse() * current_pose).vector
        if np.linalg.norm(error) < tolerance:
            print(f"Converged to pose in {i+1} iterations")
            break

        # Compute Jacobian in local frame since the error is in the end-effector frame
        J = pin.computeFrameJacobian(model, data, q, end_effector_frame, pin.ReferenceFrame.LOCAL)

        # Quadratic program matrices
        H = J.T @ J + damping * np.eye(model.nq)  # Regularized Hessian
        g = -J.T @ error  # Gradient term

        # Inequality constraints for joint limits
        # Ensuring q_min <= q + Δq <= q_max
        C = np.vstack([np.eye(model.nq), -np.eye(model.nq)])
        b = np.hstack([q_min - q, -(q_max - q)])

        # Solve QP
        delta_q = solve_qp(H, g, C.T, b)[0]  # Solving the QP with constraints

        # Update joint configuration
        q = pin.integrate(model, q, delta_q)

    # Store the joint configuration and error for this step
    q_trajectory.append(q)
    pose_errors.append(np.linalg.norm(error))

# Print the resulting trajectory of joint configurations
for i, q in enumerate(q_trajectory):
    print(f"Time Step {i+1}, Joint Configuration: {q}")

# Optional: Plot the error over time
import matplotlib.pyplot as plt
plt.plot(time_steps, pose_errors)
plt.xlabel('Time (s)')
plt.ylabel('Pose Error')
plt.title('Task-Space Pose Error Over Time')
plt.grid()
plt.show()
