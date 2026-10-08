import json5
from qwen_agent.tools.base import BaseTool, register_tool

# Import your native functions from your common utility module
from backend.app.intelli.geo_analysis.tool_utils.common import (
    centroid,
    polygon_area,
    polygon_perimeter,
    pair_distance_and_bearing,
    bbox_intersection_area,
    pixel_area_to_m2,
)


@register_tool('calculate_centroid')
class CentroidTool(BaseTool):
    description = 'Calculate the centroid point of a polygon or line given a list of coordinate points.'
    parameters = [
        {
            'name': 'points',
            'type': 'array',
            'description': 'A list of coordinate pairs, e.g., [[x1, y1], [x2, y2], ...]',
            'required': True
        },
        {
            'name': 'closed',
            'type': 'boolean',
            'description': 'Whether the geometry path is closed. Defaults to False.',
            'required': False
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json5.loads(params)
        cx, cy = centroid(args['points'], closed=args.get('closed', False))
        return json5.dumps({'centroid_x': cx, 'centroid_y': cy}, ensure_ascii=False)


@register_tool('calculate_polygon_area')
class PolygonAreaTool(BaseTool):
    description = 'Calculate the area of a polygon given its vertices.'
    parameters = [
        {
            'name': 'points',
            'type': 'array',
            'description': 'A list of vertex coordinate pairs defining the polygon.',
            'required': True
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json5.loads(params)
        area = polygon_area(args['points'])
        return json5.dumps({'area': area}, ensure_ascii=False)


@register_tool('calculate_polygon_perimeter')
class PolygonPerimeterTool(BaseTool):
    description = 'Calculate the perimeter distance of a polygon given its vertices.'
    parameters = [
        {
            'name': 'points',
            'type': 'array',
            'description': 'A list of vertex coordinate pairs defining the polygon.',
            'required': True
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json5.loads(params)
        perimeter = polygon_perimeter(args['points'])
        return json5.dumps({'perimeter': perimeter}, ensure_ascii=False)


@register_tool('pair_distance_and_bearing')
class DistanceAndBearingTool(BaseTool):
    description = 'Calculate the distance and bearing between two geometric entities (either in geographic or pixel space).'
    parameters = [
        {
            'name': 'a_points',
            'type': 'array',
            'description': 'Points defining the first geometry.',
            'required': True
        },
        {
            'name': 'b_points',
            'type': 'array',
            'description': 'Points defining the second geometry.',
            'required': True
        },
        {
            'name': 'space',
            'type': 'string',
            'description': 'The coordinate space type.',
            'enum': ['geographic', 'pixel'],
            'required': True
        },
        {
            'name': 'closed_a',
            'type': 'boolean',
            'description': 'Whether geometry A is closed.',
            'required': False
        },
        {
            'name': 'closed_b',
            'type': 'boolean',
            'description': 'Whether geometry B is closed.',
            'required': False
        },
        {
            'name': 'gsd_m',
            'type': 'number',
            'description': 'Ground Sample Distance in meters per pixel (optional, required if space is pixel and you want meters).',
            'required': False
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json5.loads(params)
        distance, unit, bearing = pair_distance_and_bearing(
            a_points=args['a_points'],
            b_points=args['b_points'],
            space=args['space'],
            closed_a=args.get('closed_a', False),
            closed_b=args.get('closed_b', False),
            gsd_m=args.get('gsd_m', None)
        )
        return json5.dumps({
            'distance': distance,
            'unit': unit,
            'bearing_degrees': bearing
        }, ensure_ascii=False)


@register_tool('bbox_intersection_area')
class BBoxIntersectionTool(BaseTool):
    description = 'Calculate the intersection area between two bounding boxes.'
    parameters = [
        {
            'name': 'a',
            'type': 'array',
            'description': 'First bounding box as [x_min, y_min, x_max, y_max].',
            'required': True
        },
        {
            'name': 'b',
            'type': 'array',
            'description': 'Second bounding box as [x_min, y_min, x_max, y_max].',
            'required': True
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json5.loads(params)
        intersection = bbox_intersection_area(args['a'], args['b'])
        return json5.dumps({'intersection_area': intersection}, ensure_ascii=False)


@register_tool('pixel_area_to_m2')
class PixelAreaConversionTool(BaseTool):
    description = 'Convert pixel-based area units to square meters using Ground Sample Distance (GSD).'
    parameters = [
        {
            'name': 'area_px',
            'type': 'number',
            'description': 'The area value in pixels squared.',
            'required': True
        },
        {
            'name': 'gsd_m',
            'type': 'number',
            'description': 'Ground Sample Distance in meters per pixel.',
            'required': False
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json5.loads(params)
        area, unit = pixel_area_to_m2(args['area_px'], gsd_m=args.get('gsd_m', None))
        return json5.dumps({'converted_area': area, 'unit': unit}, ensure_ascii=False)