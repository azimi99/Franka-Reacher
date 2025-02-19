# import pyrealsense2 as rs

# # Create a context object to access all connected devices
# ctx = rs.context()
# devices = ctx.query_devices()

# if not devices:
#     print("No Intel RealSense devices found.")
#     exit()

# for dev in devices:
#     print(f"\nDevice: {dev.get_info(rs.camera_info.name)}")
    
#     # Iterate over each sensor (e.g., depth, color, IR)
#     for sensor in dev.query_sensors():
#         print(f"\n  Sensor: {sensor.get_info(rs.camera_info.name)}")
        
#         # Iterate over all available stream profiles for this sensor
#         for profile in sensor.get_stream_profiles():
#             try:
#                 # Try to convert to a video stream profile
#                 vprofile = profile.as_video_stream_profile()
#             except Exception:
#                 # Skip profiles that cannot be converted
#                 continue
                
#             fmt = vprofile.format()
#             width = vprofile.width()
#             height = vprofile.height()
#             fps = vprofile.fps()
            
#             print(f"    Format: {fmt}, Resolution: {width}x{height}, FPS: {fps}")



import pyrealsense2 as rs

# Initialize pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)  # Adjust resolution as needed

# Start streaming
pipeline.start(config)

# Get camera intrinsics
profile = pipeline.get_active_profile()
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# Print intrinsics
print("Camera Matrix (Intrinsic Parameters):")
print(f"fx = {intrinsics.fx}, fy = {intrinsics.fy}")
print(f"cx = {intrinsics.ppx}, cy = {intrinsics.ppy}")
print(f"Distortion Model: {intrinsics.model}")
print(f"Distortion Coefficients: {intrinsics.coeffs}")

# Stop streaming
pipeline.stop()
