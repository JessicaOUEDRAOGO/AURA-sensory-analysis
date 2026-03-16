from src.core.mapping.coordinate_mapper import CoordinateMapper

mapper = CoordinateMapper()
mapper.load()

pt_cam = [960, 540]

pt_proj = mapper.camera_raw_to_projector(pt_cam)
pt_graph = mapper.camera_raw_to_graph(pt_cam)

print("Camera raw :", pt_cam)
print("Projector :", pt_proj)
print("Graph :", pt_graph)