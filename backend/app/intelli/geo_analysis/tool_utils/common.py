from math import atan2, cos, degrees, hypot, radians, sin, sqrt


def centroid(points, closed=False):
    if not points:
        raise ValueError("Geometry has no coordinates")

    if closed and len(points) >= 3:
        cross_sum = sum(
            points[i][0] * points[(i + 1) % len(points)][1]
            - points[(i + 1) % len(points)][0] * points[i][1]
            for i in range(len(points))
        )

        if abs(cross_sum) > 1e-15:
            cx = sum(
                (points[i][0] + points[(i + 1) % len(points)][0])
                * (
                    points[i][0] * points[(i + 1) % len(points)][1]
                    - points[(i + 1) % len(points)][0] * points[i][1]
                )
                for i in range(len(points))
            ) / (3 * cross_sum)

            cy = sum(
                (points[i][1] + points[(i + 1) % len(points)][1])
                * (
                    points[i][0] * points[(i + 1) % len(points)][1]
                    - points[(i + 1) % len(points)][0] * points[i][1]
                )
                for i in range(len(points))
            ) / (3 * cross_sum)

            return cx, cy

    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def polygon_area(points):
    if len(points) < 3:
        raise ValueError("Polygon requires at least three vertices")

    return abs(
        sum(
            points[i][0] * points[(i + 1) % len(points)][1]
            - points[(i + 1) % len(points)][0] * points[i][1]
            for i in range(len(points))
        )
    ) / 2


def polygon_perimeter(points):
    if len(points) < 3:
        raise ValueError("Polygon requires at least three vertices")

    return sum(
        hypot(
            points[i][0] - points[(i + 1) % len(points)][0],
            points[i][1] - points[(i + 1) % len(points)][1],
        )
        for i in range(len(points))
    )


def pair_distance_and_bearing(
    a_points,
    b_points,
    space,
    closed_a=False,
    closed_b=False,
    gsd_m=None,
):
    ax, ay = centroid(a_points, closed_a)
    bx, by = centroid(b_points, closed_b)

    if space == "geographic":
        lat1 = radians(ay)
        lat2 = radians(by)
        dlat = lat2 - lat1
        dlon = radians(bx - ax)

        h = (
            sin(dlat / 2) ** 2
            + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        )

        distance = (
            6_371_008.8
            * 2
            * atan2(sqrt(h), sqrt(max(0.0, 1 - h)))
        )

        y = sin(dlon) * cos(lat2)
        x = (
            cos(lat1) * sin(lat2)
            - sin(lat1) * cos(lat2) * cos(dlon)
        )

        bearing = (degrees(atan2(y, x)) + 360) % 360

        return distance, "m", bearing

    if space == "pixel":
        pixels = hypot(bx - ax, by - ay)

        if gsd_m is not None:
            distance = pixels * gsd_m
            unit = "m"
        else:
            distance = pixels
            unit = "px"

        # Pixel Y increases downward.
        bearing = (
            degrees(atan2(bx - ax, -(by - ay))) + 360
        ) % 360

        return distance, unit, bearing

    raise ValueError("Unsupported geometry space")


def bbox_intersection_area(a, b):
    """
    a and b: (x_min, y_min, x_max, y_max)
    """
    return (
        max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    )


def pixel_area_to_m2(area_px, gsd_m=None):
    if gsd_m is None:
        return area_px, "px^2"

    return area_px * gsd_m**2, "m^2"