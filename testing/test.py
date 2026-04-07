from pathlib import Path

import warp as wp
from pxr import Usd

import newton

# these dependencies are needed to load example assets and ingest meshes from USD
import newton.examples
import newton.usd

# Create a new model builder
builder = newton.ModelBuilder()

# Add a ground plane (infinite static plane at z=0)
builder.add_ground_plane()

# Height from which to drop shapes
drop_z = 2.0

# SPHERE
sphere_pos = wp.vec3(0.0, -4.0, drop_z)
body_sphere = builder.add_body(
    xform=wp.transform(p=sphere_pos, q=wp.quat_identity()),
    label="sphere",  # Optional: human-readable identifier
)
builder.add_shape_sphere(body_sphere, radius=0.5)

# CAPSULE
capsule_pos = wp.vec3(0.0, -2.0, drop_z)
body_capsule = builder.add_body(xform=wp.transform(p=capsule_pos, q=wp.quat_identity()), label="capsule")
builder.add_shape_capsule(body_capsule, radius=0.3, half_height=0.7)

# CYLINDER
cylinder_pos = wp.vec3(0.0, 0.0, drop_z)
body_cylinder = builder.add_body(xform=wp.transform(p=cylinder_pos, q=wp.quat_identity()), label="cylinder")
builder.add_shape_cylinder(body_cylinder, radius=0.4, half_height=0.6)

# Multi-Shape Collider
multi_shape_pos = wp.vec3(0.0, 2.0, drop_z)
body_multi_shape = builder.add_body(xform=wp.transform(p=multi_shape_pos, q=wp.quat_identity()), label="multi_shape")

# Now attach both a sphere and a box to the multi-shape body
# body-local shape offsets, offset sphere in x so the body will topple over
sphere_offset = wp.vec3(0.1, 0.0, -0.3)
box_offset = wp.vec3(0.0, 0.0, 0.3)
builder.add_shape_sphere(body_multi_shape, wp.transform(p=sphere_offset, q=wp.quat_identity()), radius=0.25)
builder.add_shape_box(body_multi_shape, wp.transform(p=box_offset, q=wp.quat_identity()), hx=0.25, hy=0.25, hz=0.25)

print(f"Added {builder.body_count} bodies with collision shapes")

# Load a mesh from a USD file
usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
demo_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/bunny"))

# Add the mesh as a rigid body
mesh_pos = wp.vec3(0.0, 4.0, drop_z - 0.5)
body_mesh = builder.add_body(xform=wp.transform(p=mesh_pos, q=wp.quat(0.5, 0.5, 0.5, 0.5)), label="bunny")
builder.add_shape_mesh(body_mesh, mesh=demo_mesh)

print(f"Added mesh body with {demo_mesh.vertices.shape[0]} vertices")

# Optional: Run the simulation on CPU
use_cpu = False
if use_cpu:
    wp.set_device("cpu")  # alternatively, pass device="cpu" to the finalize method

# Finalize the model - this creates the simulation-ready Model object
model = builder.finalize()

print(f"Model finalized for device {model.device}:")
print(f"  Bodies: {model.body_count}")
print(f"  Shapes: {model.shape_count}")
print(f"  Joints: {model.joint_count}")

# Create two state objects for time integration
state_0 = model.state()  # Current state
state_1 = model.state()  # Next state

# The control object is not used in this example, but we create it for completeness
control = model.control()

# Allocate Contacts buffer
contacts = model.contacts()

print("State, Contacts and Control objects created")

# Create the XPBD solver with 10 constraint iterations
solver = newton.solvers.SolverXPBD(model, iterations=10)

print(f"Solver created: {type(solver).__name__}")

# Simulation parameters
fps = 60  # Frames per second for visualization
frame_dt = 1.0 / fps  # Time step per frame
sim_substeps = 10  # Number of physics substeps per frame
sim_dt = frame_dt / sim_substeps  # Physics time step

print("Simulation configured:")
print(f"  Frame rate: {fps} Hz")
print(f"  Frame dt: {frame_dt:.4f} s")
print(f"  Physics substeps: {sim_substeps}")
print(f"  Physics dt: {sim_dt:.4f} s")

viewer = newton.viewer.ViewerGL()
viewer.set_model(model)

def simulate():
    """Run multiple physics substeps for one frame."""
    global state_0, state_1

    for _ in range(sim_substeps):
        # 1. Clear forces in input state
        state_0.clear_forces()

        # 2. Apply control targets/forces, and viewer picking forces if using the OpenGL viewer
        # update_control(state_0, control)
        viewer.apply_forces(state_0)

        # 3. Detect collisions
        model.collide(state_0, contacts)

        # 4. Step the simulation by one physics timestep
        solver.step(state_in=state_0, state_out=state_1, control=control, contacts=contacts, dt=sim_dt)

        # 5. Swap states (next becomes current)
        state_0, state_1 = state_1, state_0

# Capture the simulation as a CUDA graph (if running on GPU)
if wp.get_device().is_cuda:
    with wp.ScopedCapture() as capture:
        simulate()
    graph = capture.graph
    print("CUDA graph captured for optimized execution")
else:
    graph = None
    print("Running on CPU (no CUDA graph)")

from imgui_bundle import imgui

# 1. Create a flag to track if a reset was requested via UI
reset_requested = False

def gui(ui):
    global reset_requested  # Tell Python we want to modify the variable outside this function
    
    ui.text("Simulation Controls")
    ui.text("HELLO FROM PYTHON!!!!!")
    
    # ui.button returns True ONLY on the frame it is clicked
    if ui.button("Reset Simulation"):
        reset_requested = True
        print("Reset requested via GUI")

# Register the callback
viewer.register_ui_callback(gui, position="side")

# --- Main Simulation Loop ---
sim_time = 0.0

while viewer.is_running():
    # 2. Check for Reset (Keyboard 'R' OR the GUI button flag)
    if viewer.is_key_down('r') or reset_requested:
        # Reset the physics states
        state_0 = model.state()
        state_1 = model.state()
        sim_time = 0.0
        
        # IMPORTANT: Clear the flag so we don't reset forever!
        reset_requested = False
        
        # 3. Recapture the CUDA graph for the new memory addresses
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                simulate()
            graph = capture.graph
        
        print("Simulation Reset Complete")

    # 4. Standard Step Logic
    elif not viewer.is_paused():
        if graph:
            wp.capture_launch(graph)
        else:
            simulate()
        sim_time += frame_dt

    # 5. Render
    viewer.begin_frame(sim_time)
    viewer.log_state(state_0)
    viewer.log_contacts(contacts, state_0)
    viewer.end_frame()
print(f"\nSimulation complete! Total time: {sim_time:.2f} seconds")

