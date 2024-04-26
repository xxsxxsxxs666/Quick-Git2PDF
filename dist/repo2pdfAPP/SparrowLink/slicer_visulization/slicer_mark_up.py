import json


def create_extended_plane_markup_json(center, normal, bounds, coordinate_system="LPS", plane_type="pointNormal",
                                      size_mode="absolute", auto_scaling_factor=1.0, markup_orientation=None,
                                      orientation=None, object_to_base=None, base_to_node=None, color=[0, 0, 1]):
    """
    Create an extended JSON string for a plane markup in 3D Slicer, including various properties.

    Parameters:
    - center: The center of the plane in world coordinates.
    - normal: The normal vector of the plane.
    - size: The size of the plane (length, width, height).
    - bounds: The bounds of the plane in world coordinates.
    - coordinate_system: The coordinate system used (default "LPS").
    - plane_type: The type of the plane (default "pointNormal").
    - size_mode: The size mode of the plane (default "absolute").
    - auto_scaling_factor: Auto scaling factor for the plane size (default 1.0).
    - orientation: The orientation matrix of the plane (3x3 matrix).
    - object_to_base: The object to base transform matrix (4x4 matrix).
    - base_to_node: The base to node transform matrix (4x4 matrix).

    Returns:
    - A JSON string representing the extended markup for 3D Slicer.
    """
    # Default values for orientation and transform matrices if not provided
    if orientation is None:
        orientation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    if object_to_base is None:
        object_to_base = orientation[:3] + [0.0] + orientation[3:6] + [0.0] + orientation[6:] + [0.0, 0.0, 0.0, 0.0, 1.0]
    if base_to_node is None:
        base_to_node = markup_orientation[:3] + [center[0]] + markup_orientation[3:6] + [center[1]] + markup_orientation[6:] + [center[2], 0.0, 0.0, 0.0, 1.0]

    # calculate size
    length = bounds[1] - bounds[0]  # xMax - xMin
    width = bounds[3] - bounds[2]  # yMax - yMin
    size = [length, width, 0.0]

    markup_json = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Plane",
                "coordinateSystem": coordinate_system,
                "coordinateUnits": "mm",
                "locked": False,
                "fixedNumberOfControlPoints": False,
                "labelFormat": "%N-%d",
                "lastUsedControlPointNumber": 1,
                "planeType": plane_type,
                "sizeMode": size_mode,
                "autoScalingFactor": auto_scaling_factor,
                "center": center,
                "normal": normal, # "objectToBase": object_to_base,
                "baseToNode": base_to_node,
                "orientation": markup_orientation,
                "size": size,
                "planeBounds": bounds,
                "display":
                    {
                        "color": color,
                    },
                "controlPoints": [
                    {
                        "id": "1",
                        "label": "P-1",
                        "position": center,
                        "orientation": orientation,
                        "selected": True,
                        "locked": False,
                        "visibility": True,
                        "positionStatus": "defined"
                    }
                ],
                "measurements": [
                    {
                        "name": "area",
                        "enabled": False,
                        "units": "cm2",
                        "printFormat": "%-#4.4g%s"
                    }
                ]
            }
        ]
    }

    return json.dumps(markup_json, indent=4)
