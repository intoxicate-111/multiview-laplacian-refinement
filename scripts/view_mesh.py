#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description="Visualize a mesh.")
    parser.add_argument("mesh", type=Path, help="Path to OBJ/PLY/STL mesh")
    parser.add_argument("--wireframe", action="store_true")
    parser.add_argument("--normals", action="store_true")
    args = parser.parse_args()

    path = args.mesh.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load mesh: {path}")

    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)

    print(f"mesh: {path}")
    print(f"vertices: {len(mesh.vertices)}")
    print(f"triangles: {len(mesh.triangles)}")
    print(f"centroid: {vertices.mean(axis=0)}")
    print(f"bbox min: {vertices.min(axis=0)}")
    print(f"bbox max: {vertices.max(axis=0)}")

    geometries = [mesh]

    if args.wireframe:
        wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        geometries.append(wire)

    o3d.visualization.draw_geometries(
        geometries,
        window_name=path.name,
        width=1280,
        height=900,
        mesh_show_back_face=True,
        mesh_show_wireframe=args.wireframe,
        point_show_normal=args.normals,
    )


if __name__ == "__main__":
    main()
