# SPDX-License-Identifier: Apache-2.0
"""Explicit ray/ground intersection for supplied camera calibration matrices."""


def lift_pixels(points, image_to_reference, ground_z=-1.0):
    """Lift native camera pixels into the matrix's reference frame.

    image_to_reference is the inverse 4x4 reference-to-image projection.
    Invalid, parallel, or backward rays raise rather than becoming zero points.
    Callers must undo any image resizing, collage offsets and rear rotation first.
    """
    import numpy as np

    points = np.asarray(points, dtype=float)
    matrix = np.asarray(image_to_reference, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or matrix.shape != (4, 4):
        raise ValueError("Expected Nx2 pixels and a 4x4 inverse projection matrix")
    if not np.isfinite(points).all() or not np.isfinite(matrix).all() or not np.isfinite(ground_z):
        raise ValueError("Geometry inputs must be finite")
    center = matrix @ np.array([0.0, 0.0, 0.0, 1.0])
    if abs(center[3]) < 1e-8:
        raise ValueError("Camera center is at infinity")
    center = center[:3] / center[3]
    homogeneous = np.column_stack([points, np.ones((len(points), 2))]) @ matrix.T
    if (abs(homogeneous[:, 3]) < 1e-8).any():
        raise ValueError("Projected points are at infinity")
    rays = homogeneous[:, :3] / homogeneous[:, 3:4] - center
    if (abs(rays[:, 2]) < 1e-8).any():
        raise ValueError("Ray is parallel to the ground plane")
    distance = (ground_z - center[2]) / rays[:, 2]
    if (distance <= 0).any():
        raise ValueError("Ground intersection is behind the camera")
    return center + distance[:, None] * rays
