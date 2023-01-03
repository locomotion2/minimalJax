import os
import time

import links_and_nodes as ln

SERVICE_NAME = "bert.distance_reward_service"
SERVICE_INTERFACE = "distance_reward_service"

# Connect to ln manager
ln_client = ln.client("distance_distance_service", os.environ["LN_MANAGER"])
# Retrieve service
distance_service = ln_client.get_service(SERVICE_NAME, SERVICE_INTERFACE)
energy_service = ln_client.get_service("bert.energy_cost_service", "energy_cost_service")

# Toggle recording
distance_service.call()
energy_service.call()

duration = 4.0  # in s
time.sleep(duration)

# Disable recording
distance_service.call()
energy_service.call()

# Retrieve the response
dx = distance_service.resp.dx
dy = distance_service.resp.dy
# total_distance = distance_service.resp.total_distance
mean_speed = distance_service.resp.mean_speed
mean_distance = distance_service.resp.mean_distance
n_steps = distance_service.resp.n_steps

print(f"dx= {dx * 100:.2f} cm dy= {dy * 100:.2f} cm mean_speed= {mean_speed:.2f} m/s")
approx_speed = dx / duration
print(f"Approx mean speed = {approx_speed:.2f} m/s  fps = {n_steps / duration:.2f} {n_steps} steps")
print(f"Heading deviation = {distance_service.resp.heading_deviation:.2f} deg")

# Retrieve the response
mean_cost_mecha = energy_service.resp.mean_cost_mecha
mean_cost_elec = energy_service.resp.mean_cost_elec
n_steps = energy_service.resp.n_steps

print(f"Mean cost mecha = {mean_cost_mecha:.4f} - Mean cost elec: {mean_cost_elec:.4f} - {n_steps} steps")

# release the service
ln_client.release_service(distance_service)
ln_client.release_service(energy_service)
