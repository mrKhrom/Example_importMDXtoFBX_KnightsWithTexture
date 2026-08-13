#!/usr/bin/env python3
"""CLI: Warcraft 3 MDX -> FBX for Unity."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from fbx_writer import write_fbx  # noqa: E402
from mdx_parser import MDXParseError, dump_model, parse_mdx  # noqa: E402
from scene_ir import (  # noqa: E402
    ConvertOptions,
    bake_scene,
    build_ir,
    mount_report,
    write_anim_json,
    write_obj,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert Warcraft 3 MDX models to FBX (skeleton + animations) for Unity."
    )
    p.add_argument("input", help="Path to a .mdx file, or a directory with --batch")
    p.add_argument("-o", "--output", help="Output .fbx path (single-file mode)")
    p.add_argument("--dump", action="store_true", help="Print model statistics and exit")
    p.add_argument("--batch", action="store_true", help="Convert every .mdx in the input directory")
    p.add_argument("--texture", help="Diffuse texture file for the primary BLP (e.g. KnightTexture.png)")
    p.add_argument("--texture-dir", help="Directory to search for replacement textures")
    p.add_argument(
        "--axis",
        choices=("unity", "raw"),
        default="unity",
        help="unity: WC3 (x,y,z) -> (y,z,x) so the unit stands on Y and faces +Z. raw: keep WC3 axes.",
    )
    p.add_argument(
        "--all-geosets",
        action="store_true",
        help="Export every geoset (death/portrait/guts). Default keeps only Stand-visible parts.",
    )
    p.add_argument(
        "--dump-obj",
        help="Also write a bind-pose OBJ (no skin) next to the FBX, or at this path.",
    )
    p.add_argument("--fps", type=float, default=30.0, help="Bake frame rate (default 30)")
    p.add_argument(
        "--max-influences",
        type=int,
        default=4,
        help="Max bone influences per vertex (Unity default 4)",
    )
    p.add_argument(
        "--include-helpers-as-bones",
        dest="include_helpers",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-include-helpers-as-bones",
        dest="include_helpers",
        action="store_false",
    )
    p.add_argument(
        "--include-attachments",
        dest="include_attachments",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-include-attachments",
        dest="include_attachments",
        action="store_false",
    )
    p.add_argument(
        "--skip-particles",
        dest="skip_particles",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--keep-particles",
        dest="skip_particles",
        action="store_false",
        help="Reserved; particles are still not exported to FBX.",
    )
    p.add_argument(
        "--bake-animations",
        dest="bake_animations",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-bake-animations",
        dest="bake_animations",
        action="store_false",
        help="Use original key times instead of a uniform fps grid.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def options_from_args(args, source_path: str) -> ConvertOptions:
    source_dir = source_path if os.path.isdir(source_path) else os.path.dirname(source_path)
    return ConvertOptions(
        axis=args.axis,
        fps=args.fps,
        max_influences=args.max_influences,
        include_helpers=args.include_helpers,
        include_attachments=args.include_attachments,
        skip_particles=args.skip_particles,
        bake_animations=args.bake_animations,
        geoset_filter="all" if args.all_geosets else "stand",
        explode_mesh=True,
        texture=args.texture,
        texture_dir=args.texture_dir or source_dir,
        source_dir=source_dir,
    )


def convert_one(src: str, dst: Optional[str], args) -> str:
    model = parse_mdx(src)
    if model.version != 800:
        logging.warning(
            "%s: MDX version %d (parser targets classic 800; later versions may miss SKIN/BIDX)",
            os.path.basename(src),
            model.version,
        )
    opts = options_from_args(args, src)
    scene = build_ir(model, opts)

    if args.dump:
        print(dump_model(model))
        print("")
        _print_ir_checks(scene, model)
        return ""

    if not dst:
        dst = os.path.splitext(src)[0] + ".fbx"
    clips = bake_scene(scene, opts)
    write_fbx(dst, scene, clips)
    json_path = os.path.splitext(dst)[0] + ".anim.json"
    write_anim_json(json_path, scene, clips)
    logging.info("wrote %s", json_path)
    obj_path = args.dump_obj
    if obj_path:
        if obj_path in (True, "1") or obj_path.endswith(os.sep):
            obj_path = os.path.splitext(dst)[0] + ".obj"
        elif os.path.isdir(obj_path):
            obj_path = os.path.join(obj_path, os.path.splitext(os.path.basename(dst))[0] + ".obj")
        write_obj(obj_path, scene)
        logging.info("wrote bind-pose OBJ %s", obj_path)
    else:
        # Always write a sibling OBJ so Unity issues can be compared against raw mesh.
        obj_path = os.path.splitext(dst)[0] + ".obj"
        write_obj(obj_path, scene)
        logging.info("wrote bind-pose OBJ %s", obj_path)
    logging.info(
        "wrote %s (%d nodes, %d meshes, %d clips)",
        dst,
        len(scene.nodes),
        len(scene.meshes),
        len(clips),
    )
    for line in mount_report(scene):
        logging.info("%s", line)
    return dst


def _print_ir_checks(scene, model) -> None:
    print("IR nodes             %d" % len(scene.nodes))
    print("  bones              %d" % sum(1 for n in scene.nodes if n.role == "bone"))
    print("  helpers            %d" % sum(1 for n in scene.nodes if n.role == "helper"))
    print("  attachments        %d" % sum(1 for n in scene.nodes if n.role == "attachment"))
    print("IR clips             %s" % [c.name for c in scene.clips])
    check_ids = [i for i in (0, 78, 79, 101, 109) if i in scene.nodes_by_id]
    # Bind check is in WC3 space: rebuild rest-only world from rest_t (pre-export).
    # apply_axis_to_scene already replaced bind_world with export-space matrices.
    # Verify using rest_t chain against original pivots by reconstructing WC3 world.
    from scene_ir import IrNode, compute_world_bind

    wc3_nodes = {}
    for n in scene.nodes:
        wc3_nodes[n.object_id] = IrNode(
            object_id=n.object_id,
            parent_id=n.parent_id,
            name=n.name,
            sanitized=n.sanitized,
            role=n.role,
            flags=n.flags,
            billboarded=n.billboarded,
            rest_t=n.rest_t,
            rest_r=n.rest_r,
            rest_s=n.rest_s,
            tracks=n.tracks,
            wc3_name=n.wc3_name,
        )
    tmp = type("T", (), {})()
    tmp.nodes = list(wc3_nodes.values())
    errors = []
    world = compute_world_bind(wc3_nodes)
    from xform import vlen

    for oid in check_ids:
        if oid >= len(model.pivots):
            continue
        pos = (world[oid][0][3], world[oid][1][3], world[oid][2][3])
        piv = model.pivots[oid]
        dist = vlen((pos[0] - piv[0], pos[1] - piv[1], pos[2] - piv[2]))
        status = "OK" if dist < 1e-3 else "FAIL"
        print("bind id %-3d %s  dist=%.6f" % (oid, status, dist))
        if dist >= 1e-3:
            errors.append(oid)
    if errors:
        print("BIND CHECK FAILED for ids %s" % errors)
    else:
        print("bind check          OK")

    # Walk duration
    walk = next((c for c in scene.clips if c.raw_name == "Walk"), None)
    if walk:
        print("Walk duration       %.3f s" % ((walk.end_ms - walk.start_ms) / 1000.0))

    # Skin clamp
    max_after = 0
    for mesh in scene.meshes:
        for inf in mesh.influences:
            max_after = max(max_after, len(inf))
            s = sum(w for _b, w in inf)
            if inf and abs(s - 1.0) > 1e-4:
                print("weight sum fail mesh %d sum=%s" % (mesh.index, s))
    print("max influences out  %d" % max_after)
    print("exported meshes     %d  %s" % (len(scene.meshes), [m.name for m in scene.meshes]))
    if scene.meshes:
        xs, ys, zs = [], [], []
        for mesh in scene.meshes:
            for p in mesh.positions:
                xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        print(
            "export AABB         (%.1f,%.1f,%.1f) .. (%.1f,%.1f,%.1f)"
            % (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))
        )
        print("export height Y     %.1f" % (max(ys) - min(ys)))
        print("exploded verts      %d" % sum(len(m.positions) for m in scene.meshes))
    print("\n".join(mount_report(scene)))


def collect_batch(path: str) -> List[str]:
    if os.path.isfile(path) and path.lower().endswith(".mdx"):
        return [path]
    if not os.path.isdir(path):
        raise SystemExit("batch input is not a directory: %s" % path)
    files = [
        os.path.join(path, n)
        for n in sorted(os.listdir(path))
        if n.lower().endswith(".mdx")
    ]
    if not files:
        raise SystemExit("no .mdx files in %s" % path)
    return files


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        if args.batch:
            for src in collect_batch(args.input):
                dst = os.path.splitext(src)[0] + ".fbx"
                convert_one(src, dst, args)
        else:
            if not os.path.isfile(args.input):
                raise SystemExit("input file not found: %s" % args.input)
            convert_one(args.input, args.output, args)
    except MDXParseError as exc:
        logging.error("parse error: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
