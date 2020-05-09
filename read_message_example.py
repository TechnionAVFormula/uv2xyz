from pyFormulaClientNoNvidia import messages
from PIL import Image
from perception_functions import get_cones_from_camera, draw_results_on_image
from geometry import trasform_img_cones_to_xyz
from google.protobuf import json_format

# Parameters
type_map = [' ']*3
type_map[messages.perception.Yellow-1] ='yellow'
type_map[messages.perception.Blue-1] ='blue'
type_map[messages.perception.Orange-1] ='orange'

# Open message file: {msg frame}_camera_msg_{msg id}.bin
with open('Frames_from_01_05_20/1_camera_msg_4.bin', 'rb') as f:
    buffer = f.read()
    camera_msg = messages.common.Message()
    camera_msg.ParseFromString(buffer)
with open('Frames_from_01_05_20/1_depth_camera_msg_3.bin', 'rb') as f:
    buffer = f.read()
    depth_camera_msg = messages.common.Message()
    depth_camera_msg.ParseFromString(buffer)
with open('Frames_from_01_05_20/1_ground_truth_5.bin', 'rb') as f:
    buffer = f.read()
    ground_truth_msg = messages.common.Message()
    ground_truth_msg.ParseFromString(buffer)

# Extract data from camera
camera_data = messages.sensors.CameraSensor()
camera_msg.data.Unpack(camera_data)
depth_camera_data = messages.sensors.DepthCameraSensor()
depth_camera_msg.data.Unpack(depth_camera_data)
ground_truth_data = messages.ground_truth.GroundTruth()
ground_truth_msg.data.Unpack(ground_truth_data)

camera_pos = camera_data.config.sensor_position
camera_pos.z = 1.2  # an error occured which sets z=0 in these sample messages
camera_pos_depth = depth_camera_data.config.sensor_position

# Process camera data
img_cones = get_cones_from_camera(camera_data.width, camera_data.height, camera_data.pixels)
xyz_cones = trasform_img_cones_to_xyz(img_cones, camera_data.width, camera_data.height,
                                      depth_camera_data.config.data_type, depth_camera_data.pixels,
                                      camera_data.config.hfov, camera_data.config.vfov, camera_pos)

# Print detection results
print("Bounding box list in image plain:")
for i, BB in enumerate(img_cones):
    print(f"({i}) u = {BB[0]}, v = {BB[1]}, h = {BB[2]}, w = {BB[3]}, type = {type_map[BB[4]-1]}")
print("Cones X,Y,Z list in ENU coordinate system (X - right, Y - forward, Z - upward):")
for i, xyz_cone in enumerate(xyz_cones):
    print(f"({i}) X = {round(xyz_cone[0],2)}, Y = {round(xyz_cone[1],2)}, Z = {round(xyz_cone[2],2)}, type = {type_map[xyz_cone[3] - 1]}")
# Print GT results
print(json_format.MessageToJson(ground_truth_data))

# Export captured images
img_RGB = Image.frombytes("RGB", (camera_data.width, camera_data.height), camera_data.pixels, 'raw', 'RGBX', 0,-1)
img_RGB.save('1_camera_msg_4.png')
dc_img = Image.frombytes("I;16", (depth_camera_data.width, depth_camera_data.height), depth_camera_data.pixels)
dc_img.save('1_depth_camera_msg_3.png')


# draw detection results on image and export
img_RGB_boxes = draw_results_on_image(img_RGB, img_cones, type_map)
img_RGB_boxes.save('1_RGB_detected_cones.jpg')

# close images
img_RGB.close()
dc_img.close()