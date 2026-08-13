"""ASCII FBX 7.4 writer (meshes, skeleton, skin, clips). No Autodesk SDK."""

from __future__ import annotations

import hashlib
import os
import zlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from animation import BakedClip, BakedNode
from mdx_parser import FILTER_NAMES
from scene_ir import IrMaterial, IrMesh, IrNode, IrScene, IrTexture, bind_inverse
from xform import mat4_to_col_major, quat_to_euler_xyz_deg

FBX_SECOND = 46186158000  # KTime ticks per second
FBX_VERSION = 7400


class _Ids:
    def __init__(self) -> None:
        self._used = set()

    def get(self, *parts: object) -> int:
        key = "|".join(str(p) for p in parts)
        n = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:15], 16)
        if n == 0:
            n = 1
        while n in self._used:
            n += 1
        self._used.add(n)
        return n


def _fmt(v: float) -> str:
    if abs(v) < 1e-12:
        return "0"
    s = "%.9g" % v
    return s


def _fmt_list(values: Sequence[float], per_line: int = 12) -> str:
    parts = [_fmt(float(v)) for v in values]
    if len(parts) <= per_line:
        return ",".join(parts)
    lines = []
    for i in range(0, len(parts), per_line):
        lines.append(",".join(parts[i : i + per_line]))
    return ",".join(lines)


def _fmt_ints(values: Sequence[int], per_line: int = 16) -> str:
    parts = [str(int(v)) for v in values]
    if len(parts) <= per_line:
        return ",".join(parts)
    lines = []
    for i in range(0, len(parts), per_line):
        lines.append(",".join(parts[i : i + per_line]))
    return ",".join(lines)


def _sec_to_ktime(t: float) -> int:
    return int(round(t * FBX_SECOND))


def write_solid_png(path: str, rgb: Tuple[int, int, int], size: int = 8) -> None:
    r, g, b = rgb
    raw = b"".join(b"\x00" + bytes([r, g, b]) * size for _ in range(size))

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct_pack_u32(len(data)) + tag + data + struct_pack_u32(crc)

    ihdr = struct_pack_u32(size) + struct_pack_u32(size) + bytes([8, 2, 0, 0, 0])
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as handle:
        handle.write(png)


def struct_pack_u32(n: int) -> bytes:
    import struct

    return struct.pack(">I", n)


def ensure_placeholders(scene: IrScene, out_dir: str) -> None:
    for tex in scene.textures:
        if tex.resolved_path:
            continue
        if tex.replaceable_id == 1:
            path = os.path.join(out_dir, "TeamColor.png")
            if not os.path.isfile(path):
                write_solid_png(path, (255, 0, 255))
            tex.resolved_path = path
        elif tex.replaceable_id == 2:
            path = os.path.join(out_dir, "TeamGlow.png")
            if not os.path.isfile(path):
                write_solid_png(path, (255, 128, 0))
            tex.resolved_path = path


def write_fbx(
    path: str,
    scene: IrScene,
    clips: Sequence[BakedClip],
) -> None:
    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    ensure_placeholders(scene, out_dir)

    ids = _Ids()
    root_id = ids.get("root", scene.name)
    node_ids: Dict[int, int] = {}
    attr_ids: Dict[int, int] = {}
    for n in scene.nodes:
        node_ids[n.object_id] = ids.get("model", n.object_id, n.sanitized)
        attr_ids[n.object_id] = ids.get("attr", n.object_id)

    mesh_model_ids = {}
    geom_ids = {}
    skin_ids = {}
    for m in scene.meshes:
        mesh_model_ids[m.index] = ids.get("mesh", m.index)
        geom_ids[m.index] = ids.get("geom", m.index)
        skin_ids[m.index] = ids.get("skin", m.index)

    # Clusters: (mesh_index, bone_id) -> uid
    cluster_ids: Dict[Tuple[int, int], int] = {}
    cluster_data: Dict[Tuple[int, int], Tuple[List[int], List[float]]] = {}
    for m in scene.meshes:
        buckets: Dict[int, Tuple[List[int], List[float]]] = {}
        for vi, infs in enumerate(m.influences):
            for bone_id, w in infs:
                if bone_id not in node_ids:
                    continue
                bucket = buckets.setdefault(bone_id, ([], []))
                bucket[0].append(vi)
                bucket[1].append(w)
        for bone_id, data in buckets.items():
            cluster_ids[(m.index, bone_id)] = ids.get("cluster", m.index, bone_id)
            cluster_data[(m.index, bone_id)] = data

    mat_ids = {m.index: ids.get("mat", m.index) for m in scene.materials}
    tex_ids = {}
    vid_ids = {}
    for t in scene.textures:
        if t.resolved_path or t.filename:
            tex_ids[t.index] = ids.get("tex", t.index)
            vid_ids[t.index] = ids.get("vid", t.index)

    pose_id = ids.get("pose", "bind")

    stack_ids = []
    layer_ids = []
    for clip in clips:
        stack_ids.append(ids.get("stack", clip.name))
        layer_ids.append(ids.get("layer", clip.name))

    # Count objects for Definitions
    n_models = 1 + len(scene.nodes) + len(scene.meshes)
    n_geom = len(scene.meshes)
    n_attr = len(scene.nodes)
    n_mat = len(scene.materials)
    n_tex = len(tex_ids)
    n_vid = len(vid_ids)
    n_def = len(skin_ids) + len(cluster_ids)
    n_stack = len(clips)
    n_layer = len(clips)
    n_pose = 1

    connections: List[Tuple] = []

    body_objects: List[str] = []

    # Root. Must be parented to the implicit FBX RootNode (id 0).
    # Unity's ImportFBX only treats a node as "in the hierarchy" if that
    # chain reaches 0. Without this link every FbxSkeleton is dropped
    # ("references FbxNode that is not in the hierarchy") and every
    # split take then fails with "Split Animation Take Not Found".
    body_objects.append(_model_null(root_id, scene.name, (0, 0, 0), (0, 0, 0), (1, 1, 1)))
    connections.append(("OO", root_id, 0))

    # Bones / helpers / attachments
    for n in scene.nodes:
        kind = "Null" if n.role == "attachment" else "LimbNode"
        euler = quat_to_euler_xyz_deg(n.export_r)
        body_objects.append(
            _model_node(
                node_ids[n.object_id],
                n.sanitized,
                kind,
                n.export_t,
                euler,
                n.export_s,
                n.wc3_name,
                n.billboarded,
                n.object_id,
                n.role,
            )
        )
        # Autodesk leaves NodeAttribute unnamed ("NodeAttribute::").
        # Naming it after the bone makes Unity create FbxSkeleton 'Bone_X'
        # and then fail to resolve FbxNode 'Bone_X' in the hierarchy.
        body_objects.append(_limb_attribute(attr_ids[n.object_id], kind))
        connections.append(("OO", attr_ids[n.object_id], node_ids[n.object_id]))
        parent = root_id if n.parent_id is None else node_ids.get(n.parent_id, root_id)
        connections.append(("OO", node_ids[n.object_id], parent))

    # Meshes
    for mesh in scene.meshes:
        body_objects.append(
            _mesh_model(
                mesh_model_ids[mesh.index],
                mesh.name,
                1.0 if mesh.visible_at_rest else 0.0,
            )
        )
        body_objects.append(_geometry(geom_ids[mesh.index], mesh))
        connections.append(("OO", geom_ids[mesh.index], mesh_model_ids[mesh.index]))
        connections.append(("OO", mesh_model_ids[mesh.index], root_id))
        if 0 <= mesh.material_id < len(scene.materials):
            connections.append(
                ("OO", mat_ids[mesh.material_id], mesh_model_ids[mesh.index])
            )
        # Skin
        body_objects.append(_skin(skin_ids[mesh.index]))
        connections.append(("OO", skin_ids[mesh.index], geom_ids[mesh.index]))

    # Clusters
    for (mi, bone_id), cuid in sorted(cluster_ids.items()):
        idxs, weights = cluster_data[(mi, bone_id)]
        bone_world = scene.bind_world.get(bone_id)
        if bone_world is None:
            from xform import identity4

            bone_world = identity4()
        transform_link = bone_world
        transform = bind_inverse(scene, bone_id)
        body_objects.append(
            _cluster(cuid, scene.nodes_by_id[bone_id].sanitized, idxs, weights, transform, transform_link)
        )
        connections.append(("OO", cuid, skin_ids[mi]))
        connections.append(("OO", node_ids[bone_id], cuid))

    # Materials / textures
    for mat in scene.materials:
        tex_path = None
        if mat.texture_index is not None and mat.texture_index in tex_ids:
            tex = scene.textures[mat.texture_index]
            tex_path = tex.resolved_path
        body_objects.append(_material(mat_ids[mat.index], mat, tex_path))
        if mat.texture_index is not None and mat.texture_index in tex_ids:
            connections.append(
                ("OP", tex_ids[mat.texture_index], mat_ids[mat.index], "DiffuseColor")
            )

    for t in scene.textures:
        if t.index not in tex_ids:
            continue
        fpath = t.resolved_path or t.filename.replace("\\", "/")
        if fpath.lower().endswith(".blp"):
            fpath = os.path.splitext(fpath)[0] + ".png"
        rel = os.path.basename(fpath)
        body_objects.append(_video(vid_ids[t.index], rel, fpath))
        body_objects.append(_texture(tex_ids[t.index], rel, fpath))
        connections.append(("OO", vid_ids[t.index], tex_ids[t.index]))

    # Bind pose
    pose_nodes = []
    pose_nodes.append((root_id, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]))
    for n in scene.nodes:
        w = scene.bind_world.get(n.object_id)
        if w is None:
            continue
        pose_nodes.append((node_ids[n.object_id], mat4_to_col_major(w)))
    for mesh in scene.meshes:
        pose_nodes.append(
            (mesh_model_ids[mesh.index], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        )
    body_objects.append(_bind_pose(pose_id, pose_nodes))

    # Animations
    for clip, stack_id, layer_id in zip(clips, stack_ids, layer_ids):
        body_objects.append(_anim_stack(stack_id, clip.name, clip.duration_s))
        body_objects.append(_anim_layer(layer_id, clip.name))
        connections.append(("OO", layer_id, stack_id))

        has_trs = False
        for bn in clip.nodes:
            if bn.node_id not in node_ids:
                continue
            model_uid = node_ids[bn.node_id]
            node = scene.nodes_by_id.get(bn.node_id)
            bind_t = node.export_t if node else (0.0, 0.0, 0.0)
            bind_e = quat_to_euler_xyz_deg(node.export_r) if node else (0.0, 0.0, 0.0)
            bind_s = node.export_s if node else (1.0, 1.0, 1.0)
            if _emit_trs_curves(
                ids,
                body_objects,
                connections,
                layer_id,
                model_uid,
                clip.name,
                bn,
                bind_t,
                bind_e,
                bind_s,
            ):
                has_trs = True
        # Unity drops takes that only have Visibility (and we import
        # with importVisibility=off). A 2-key hold on the scene root
        # is enough for Decay_Bone and any other static pose clip.
        if not has_trs:
            _emit_hold_translation(
                ids, body_objects, connections, layer_id, root_id,
                clip.name, clip.duration_s,
            )
        for vis in clip.visibility:
            if vis.mesh_index not in mesh_model_ids:
                continue
            _emit_visibility_curve(
                ids,
                body_objects,
                connections,
                layer_id,
                mesh_model_ids[vis.mesh_index],
                clip.name,
                vis.mesh_index,
                vis.times_s,
                vis.values,
            )

    n_curve_node = sum(1 for o in body_objects if o.lstrip().startswith("AnimationCurveNode:"))
    n_curve = sum(1 for o in body_objects if o.lstrip().startswith("AnimationCurve:"))

    first_clip = clips[0].name if clips else ""
    for stack_id in stack_ids:
        connections.append(("OO", stack_id, 1234567890))

    header = _header(scene.name, first_clip)
    definitions = _definitions(
        n_models,
        n_geom,
        n_attr,
        n_mat,
        n_tex,
        n_vid,
        n_def,
        n_pose,
        n_stack,
        n_layer,
        n_curve_node,
        n_curve,
    )
    objects = "Objects:  {\n" + "\n".join(body_objects) + "\n}\n"
    conns = _connections(connections, root_id)

    text = (
        header
        + "\n"
        + definitions
        + "\n"
        + objects
        + "\n"
        + conns
        + "\n"
        + _takes(clips)
    )
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _emit_trs_curves(
    ids: _Ids,
    objects: List[str],
    connections: List[Tuple],
    layer_id: int,
    model_uid: int,
    clip_name: str,
    bn: BakedNode,
    bind_t,
    bind_e,
    bind_s,
) -> bool:
    times = [_sec_to_ktime(s.time_s) for s in bn.samples]
    emitted = False
    tx = [s.translation[0] for s in bn.samples]
    ty = [s.translation[1] for s in bn.samples]
    tz = [s.translation[2] for s in bn.samples]
    if _channel_needed(tx, ty, tz, bind_t):
        _emit_vec_prop(
            ids, objects, connections, layer_id, model_uid, clip_name,
            "T", "Lcl Translation", times, (tx, ty, tz),
        )
        emitted = True
    ex = [s.euler_deg[0] for s in bn.samples]
    ey = [s.euler_deg[1] for s in bn.samples]
    ez = [s.euler_deg[2] for s in bn.samples]
    if _channel_needed(ex, ey, ez, bind_e, eps=1e-3):
        _emit_vec_prop(
            ids, objects, connections, layer_id, model_uid, clip_name,
            "R", "Lcl Rotation", times, (ex, ey, ez),
        )
        emitted = True
    sx = [s.scaling[0] for s in bn.samples]
    sy = [s.scaling[1] for s in bn.samples]
    sz = [s.scaling[2] for s in bn.samples]
    if _channel_needed(sx, sy, sz, bind_s):
        _emit_vec_prop(
            ids, objects, connections, layer_id, model_uid, clip_name,
            "S", "Lcl Scaling", times, (sx, sy, sz),
        )
        emitted = True
    return emitted


def _emit_hold_translation(
    ids: _Ids,
    objects: List[str],
    connections: List[Tuple],
    layer_id: int,
    model_uid: int,
    clip_name: str,
    duration_s: float,
) -> None:
    stop = max(_sec_to_ktime(duration_s), 1)
    times = [0, stop]
    zeros = ([0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
    _emit_vec_prop(
        ids, objects, connections, layer_id, model_uid, clip_name,
        "T", "Lcl Translation", times, zeros,
    )


def _channel_needed(xs, ys, zs, bind, eps: float = 1e-6) -> bool:
    for a, b, c in zip(xs, ys, zs):
        if abs(a - bind[0]) > eps or abs(b - bind[1]) > eps or abs(c - bind[2]) > eps:
            return True
    return False


def _emit_vec_prop(
    ids, objects, connections, layer_id, model_uid, clip_name, short, prop, times, xyz
) -> None:
    node_uid = ids.get("acn", clip_name, short, model_uid)
    objects.append(
        "    AnimationCurveNode: %d, \"AnimCurveNode::%s\", \"\" {\n"
        "        Properties70:  {\n"
        "            P: \"d\", \"Compound\", \"\", \"\"\n"
        "            P: \"d|X\", \"Number\", \"\", \"A+\", %s\n"
        "            P: \"d|Y\", \"Number\", \"\", \"A+\", %s\n"
        "            P: \"d|Z\", \"Number\", \"\", \"A+\", %s\n"
        "        }\n"
        "    }"
        % (node_uid, short, _fmt(xyz[0][0] if xyz[0] else 0), _fmt(xyz[1][0] if xyz[1] else 0), _fmt(xyz[2][0] if xyz[2] else 0))
    )
    connections.append(("OO", node_uid, layer_id))
    connections.append(("OP", node_uid, model_uid, prop))
    for axis, values in zip("XYZ", xyz):
        cu = ids.get("ac", clip_name, short, axis, model_uid)
        objects.append(_anim_curve(cu, times, values))
        connections.append(("OP", cu, node_uid, "d|%s" % axis))


def _emit_visibility_curve(
    ids, objects, connections, layer_id, model_uid, clip_name, mesh_index, times_s, values
) -> None:
    node_uid = ids.get("acn", clip_name, "vis", mesh_index)
    objects.append(
        "    AnimationCurveNode: %d, \"AnimCurveNode::Visibility\", \"\" {\n"
        "        Properties70:  {\n"
        "            P: \"d\", \"Number\", \"\", \"A+\", %s\n"
        "        }\n"
        "    }"
        % (node_uid, _fmt(values[0] if values else 1.0))
    )
    connections.append(("OO", node_uid, layer_id))
    connections.append(("OP", node_uid, model_uid, "Visibility"))
    cu = ids.get("ac", clip_name, "vis", mesh_index)
    times = [_sec_to_ktime(t) for t in times_s]
    objects.append(_anim_curve(cu, times, values))
    connections.append(("OP", cu, node_uid, "d"))


def _header(name: str, active_stack: str = "") -> str:
    return (
        "; FBX 7.4.0 project file\n"
        "; Generated by mdx2fbx (Warcraft 3 MDX -> FBX)\n"
        "; ----------------------------------------------------\n\n"
        "FBXHeaderExtension:  {\n"
        "    FBXHeaderVersion: 1003\n"
        "    FBXVersion: %d\n"
        "    CreationTimeStamp:  {\n"
        "        Version: 1000\n"
        "        Year: 2026\n"
        "        Month: 8\n"
        "        Day: 14\n"
        "        Hour: 0\n"
        "        Minute: 0\n"
        "        Second: 0\n"
        "        Millisecond: 0\n"
        "    }\n"
        "    Creator: \"mdx2fbx\"\n"
        "}\n\n"
        "GlobalSettings:  {\n"
        "    Version: 1000\n"
        "    Properties70:  {\n"
        "        P: \"UpAxis\", \"int\", \"Integer\", \"\",1\n"
        "        P: \"UpAxisSign\", \"int\", \"Integer\", \"\",1\n"
        "        P: \"FrontAxis\", \"int\", \"Integer\", \"\",2\n"
        "        P: \"FrontAxisSign\", \"int\", \"Integer\", \"\",1\n"
        "        P: \"CoordAxis\", \"int\", \"Integer\", \"\",0\n"
        "        P: \"CoordAxisSign\", \"int\", \"Integer\", \"\",1\n"
        "        P: \"OriginalUpAxis\", \"int\", \"Integer\", \"\",2\n"
        "        P: \"OriginalUpAxisSign\", \"int\", \"Integer\", \"\",1\n"
        "        P: \"UnitScaleFactor\", \"double\", \"Number\", \"\",1\n"
        "        P: \"OriginalUnitScaleFactor\", \"double\", \"Number\", \"\",1\n"
        "        P: \"AmbientColor\", \"ColorRGB\", \"Color\", \"\",0,0,0\n"
        "        P: \"DefaultCamera\", \"KString\", \"\", \"\", \"Producer Perspective\"\n"
        "        P: \"TimeMode\", \"enum\", \"\", \"\",6\n"
        "        P: \"TimeSpanStart\", \"KTime\", \"Time\", \"\",0\n"
        "        P: \"TimeSpanStop\", \"KTime\", \"Time\", \"\",%d\n"
        "        P: \"CustomFrameRate\", \"double\", \"Number\", \"\",30\n"
        "    }\n"
        "}\n\n"
        "Documents:  {\n"
        "    Count: 1\n"
        "    Document: 1234567890, \"Scene\", \"Scene\" {\n"
        "        Properties70:  {\n"
        "            P: \"SourceObject\", \"object\", \"\", \"\"\n"
        "            P: \"ActiveAnimStackName\", \"KString\", \"\", \"\", \"%s\"\n"
        "        }\n"
        "        RootNode: 0\n"
        "    }\n"
        "}\n\n"
        "References:  {\n"
        "}\n"
        % (FBX_VERSION, FBX_SECOND, _escape(active_stack))
    )


def _def_block(typ: str, count: int, extra: str = "") -> str:
    if count <= 0:
        return ""
    return (
        "    ObjectType: \"%s\" {\n"
        "        Count: %d\n"
        "%s"
        "    }\n" % (typ, count, extra)
    )


def _definitions(
    n_models, n_geom, n_attr, n_mat, n_tex, n_vid, n_def, n_pose,
    n_stack, n_layer, n_curve_node, n_curve,
) -> str:
    total = (
        1  # GlobalSettings
        + n_models + n_geom + n_attr + n_mat + n_tex + n_vid
        + n_def + n_pose + n_stack + n_layer + n_curve_node + n_curve
    )
    tmpl = "        PropertyTemplate: \"%s\" {\n            Properties70:  {\n            }\n        }\n"
    parts = [
        "Definitions:  {\n",
        "    Version: 100\n",
        "    Count: %d\n" % total,
        _def_block("GlobalSettings", 1),
        _def_block("Model", n_models),
        _def_block("Geometry", n_geom),
        _def_block("NodeAttribute", n_attr),
        _def_block("Material", n_mat),
        _def_block("Texture", n_tex),
        _def_block("Video", n_vid),
        _def_block("Deformer", n_def),
        _def_block("Pose", n_pose),
        _def_block("AnimationStack", n_stack),
        _def_block("AnimationLayer", n_layer),
        _def_block("AnimationCurveNode", n_curve_node),
        _def_block("AnimationCurve", n_curve),
        "}\n",
    ]
    return "".join(p for p in parts if p)


def _props_lcl(t, e, s, visibility: float = 1.0) -> str:
    return (
        "            P: \"RotationOrder\", \"enum\", \"\", \"\",0\n"
        "            P: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A+\",%s,%s,%s\n"
        "            P: \"Lcl Rotation\", \"Lcl Rotation\", \"\", \"A+\",%s,%s,%s\n"
        "            P: \"Lcl Scaling\", \"Lcl Scaling\", \"\", \"A+\",%s,%s,%s\n"
        "            P: \"Visibility\", \"Visibility\", \"\", \"A+\",%s\n"
        % (
            _fmt(t[0]), _fmt(t[1]), _fmt(t[2]),
            _fmt(e[0]), _fmt(e[1]), _fmt(e[2]),
            _fmt(s[0]), _fmt(s[1]), _fmt(s[2]),
            _fmt(visibility),
        )
    )


def _model_null(uid, name, t, e, s) -> str:
    return (
        "    Model: %d, \"Model::%s\", \"Null\" {\n"
        "        Version: 232\n"
        "        Properties70:  {\n"
        "            P: \"RotationActive\", \"bool\", \"\", \"\",1\n"
        "            P: \"InheritType\", \"enum\", \"\", \"\",0\n"
        "            P: \"ScalingMax\", \"Vector3D\", \"Vector\", \"\",0,0,0\n"
        "            P: \"DefaultAttributeIndex\", \"int\", \"Integer\", \"\",0\n"
        "%s"
        "        }\n"
        "        Shading: Y\n"
        "        Culling: \"CullingOff\"\n"
        "    }"
        % (uid, name, _props_lcl(t, e, s))
    )


def _model_node(uid, name, kind, t, e, s, wc3_name, billboard, oid, role) -> str:
    extra = (
        "            P: \"wc3_name\", \"KString\", \"\", \"\", \"%s\"\n"
        "            P: \"wc3_id\", \"int\", \"Integer\", \"\",%d\n"
        "            P: \"wc3_role\", \"KString\", \"\", \"\", \"%s\"\n"
        % (_escape(wc3_name), oid, role)
    )
    if billboard:
        extra += "            P: \"wc3_billboard\", \"int\", \"Integer\", \"\",1\n"
    return (
        "    Model: %d, \"Model::%s\", \"%s\" {\n"
        "        Version: 232\n"
        "        Properties70:  {\n"
        "            P: \"RotationActive\", \"bool\", \"\", \"\",1\n"
        "            P: \"InheritType\", \"enum\", \"\", \"\",0\n"
        "            P: \"ScalingMax\", \"Vector3D\", \"Vector\", \"\",0,0,0\n"
        "            P: \"DefaultAttributeIndex\", \"int\", \"Integer\", \"\",0\n"
        "%s%s"
        "        }\n"
        "        Shading: Y\n"
        "        Culling: \"CullingOff\"\n"
        "    }"
        % (uid, name, kind, extra, _props_lcl(t, e, s))
    )


def _mesh_model(uid, name, visibility: float = 1.0) -> str:
    props = _props_lcl((0, 0, 0), (0, 0, 0), (1, 1, 1), visibility)
    return (
        "    Model: %d, \"Model::%s\", \"Mesh\" {\n"
        "        Version: 232\n"
        "        Properties70:  {\n"
        "            P: \"RotationActive\", \"bool\", \"\", \"\",1\n"
        "            P: \"InheritType\", \"enum\", \"\", \"\",0\n"
        "            P: \"ScalingMax\", \"Vector3D\", \"Vector\", \"\",0,0,0\n"
        "            P: \"DefaultAttributeIndex\", \"int\", \"Integer\", \"\",0\n"
        "%s"
        "        }\n"
        "        Shading: Y\n"
        "        Culling: \"CullingOff\"\n"
        "    }"
        % (uid, name, props)
    )


def _limb_attribute(uid, kind) -> str:
    # Empty name after "::" is required. A name matching the Model makes
    # Unity's FbxSkeleton lookup treat the attribute as the node.
    if kind == "Null":
        return (
            "    NodeAttribute: %d, \"NodeAttribute::\", \"Null\" {\n"
            "        TypeFlags: \"Null\"\n"
            "    }"
            % uid
        )
    return (
        "    NodeAttribute: %d, \"NodeAttribute::\", \"LimbNode\" {\n"
        "        Properties70:  {\n"
        "            P: \"Size\", \"double\", \"Number\", \"\",10\n"
        "        }\n"
        "        TypeFlags: \"Skeleton\"\n"
        "    }"
        % uid
    )


def _geometry(uid: int, mesh: IrMesh) -> str:
    verts: List[float] = []
    for p in mesh.positions:
        verts.extend(p)
    # PolygonVertexIndex: last index of each tri is bitwise not
    pvi: List[int] = []
    idx = mesh.indices
    for i in range(0, len(idx) - 2, 3):
        pvi.extend((idx[i], idx[i + 1], -idx[i + 2] - 1))
    # ByPolygonVertex / Direct: one normal+uv per corner, in index order.
    norms: List[float] = []
    uvs: List[float] = []
    for vi in idx:
        n = mesh.normals[vi] if vi < len(mesh.normals) else (0.0, 1.0, 0.0)
        norms.extend(n)
        u, v = mesh.uvs[vi] if vi < len(mesh.uvs) else (0.0, 0.0)
        uvs.extend((u, 1.0 - v))  # FBX / Unity expect V flipped vs WC3
    return (
        "    Geometry: %d, \"Geometry::%s\", \"Mesh\" {\n"
        "        Vertices: *%d {\n"
        "            a: %s\n"
        "        }\n"
        "        PolygonVertexIndex: *%d {\n"
        "            a: %s\n"
        "        }\n"
        "        GeometryVersion: 124\n"
        "        LayerElementNormal: 0 {\n"
        "            Version: 101\n"
        "            Name: \"\"\n"
        "            MappingInformationType: \"ByPolygonVertex\"\n"
        "            ReferenceInformationType: \"Direct\"\n"
        "            Normals: *%d {\n"
        "                a: %s\n"
        "            }\n"
        "        }\n"
        "        LayerElementUV: 0 {\n"
        "            Version: 101\n"
        "            Name: \"UVMap\"\n"
        "            MappingInformationType: \"ByPolygonVertex\"\n"
        "            ReferenceInformationType: \"Direct\"\n"
        "            UV: *%d {\n"
        "                a: %s\n"
        "            }\n"
        "        }\n"
        "        LayerElementMaterial: 0 {\n"
        "            Version: 101\n"
        "            Name: \"\"\n"
        "            MappingInformationType: \"AllSame\"\n"
        "            ReferenceInformationType: \"IndexToDirect\"\n"
        "            Materials: *1 {\n"
        "                a: 0\n"
        "            }\n"
        "        }\n"
        "        Layer: 0 {\n"
        "            Version: 100\n"
        "            LayerElement:  {\n"
        "                Type: \"LayerElementNormal\"\n"
        "                TypedIndex: 0\n"
        "            }\n"
        "            LayerElement:  {\n"
        "                Type: \"LayerElementUV\"\n"
        "                TypedIndex: 0\n"
        "            }\n"
        "            LayerElement:  {\n"
        "                Type: \"LayerElementMaterial\"\n"
        "                TypedIndex: 0\n"
        "            }\n"
        "        }\n"
        "    }"
        % (
            uid,
            mesh.name,
            len(verts),
            _fmt_list(verts),
            len(pvi),
            _fmt_ints(pvi),
            len(norms),
            _fmt_list(norms),
            len(uvs),
            _fmt_list(uvs),
        )
    )


def _skin(uid: int) -> str:
    return (
        "    Deformer: %d, \"Deformer::Skin\", \"Skin\" {\n"
        "        Version: 101\n"
        "        Link_DeformAcuracy: 50\n"
        "    }"
        % uid
    )


def _cluster(uid, bone_name, indexes, weights, transform, transform_link) -> str:
    return (
        "    Deformer: %d, \"SubDeformer::Cluster_%s\", \"Cluster\" {\n"
        "        Version: 100\n"
        "        UserData: \"\", \"\"\n"
        "        Indexes: *%d {\n"
        "            a: %s\n"
        "        }\n"
        "        Weights: *%d {\n"
        "            a: %s\n"
        "        }\n"
        "        Transform: *16 {\n"
        "            a: %s\n"
        "        }\n"
        "        TransformLink: *16 {\n"
        "            a: %s\n"
        "        }\n"
        "    }"
        % (
            uid,
            bone_name,
            len(indexes),
            _fmt_ints(indexes),
            len(weights),
            _fmt_list(weights),
            _fmt_list(mat4_to_col_major(transform)),
            _fmt_list(mat4_to_col_major(transform_link)),
        )
    )


def _material(uid: int, mat: IrMaterial, tex_path: Optional[str]) -> str:
    # Approximate WC3 filter as Lambert opacity / shading.
    opacity = 1.0
    if mat.filter_mode in (1, 2, 4):
        opacity = 0.999  # hint transparent
    if mat.filter_mode == 3:
        opacity = 0.8
    return (
        "    Material: %d, \"Material::%s\", \"\" {\n"
        "        Version: 102\n"
        "        ShadingModel: \"lambert\"\n"
        "        MultiLayer: 0\n"
        "        Properties70:  {\n"
        "            P: \"Diffuse\", \"Vector3D\", \"Vector\", \"\",0.8,0.8,0.8\n"
        "            P: \"DiffuseColor\", \"Color\", \"\", \"A\",0.8,0.8,0.8\n"
        "            P: \"AmbientColor\", \"Color\", \"\", \"A\",0.2,0.2,0.2\n"
        "            P: \"TransparencyFactor\", \"Number\", \"\", \"A\",%s\n"
        "            P: \"Opacity\", \"double\", \"Number\", \"\",%s\n"
        "            P: \"wc3_filter\", \"KString\", \"\", \"\", \"%s\"\n"
        "            P: \"wc3_two_sided\", \"int\", \"Integer\", \"\",%d\n"
        "            P: \"wc3_unshaded\", \"int\", \"Integer\", \"\",%d\n"
        "        }\n"
        "    }"
        % (
            uid,
            mat.name,
            _fmt(1.0 - opacity),
            _fmt(opacity),
            FILTER_NAMES.get(mat.filter_mode, str(mat.filter_mode)),
            1 if mat.two_sided else 0,
            1 if mat.unshaded else 0,
        )
    )


def _texture(uid, name, path) -> str:
    return (
        "    Texture: %d, \"Texture::%s\", \"\" {\n"
        "        Type: \"TextureVideoClip\"\n"
        "        Version: 202\n"
        "        TextureName: \"Texture::%s\"\n"
        "        Properties70:  {\n"
        "            P: \"CurrentTextureBlendMode\", \"enum\", \"\", \"\",0\n"
        "            P: \"UVSet\", \"KString\", \"\", \"\", \"UVMap\"\n"
        "            P: \"UseMaterial\", \"bool\", \"\", \"\",1\n"
        "        }\n"
        "        Media: \"Video::%s\"\n"
        "        FileName: \"%s\"\n"
        "        RelativeFilename: \"%s\"\n"
        "        ModelUVTranslation: 0,0\n"
        "        ModelUVScaling: 1,1\n"
        "        Texture_Alpha_Source: \"None\"\n"
        "        Cropping: 0,0,0,0\n"
        "    }"
        % (uid, name, name, name, _escape(path), _escape(os.path.basename(path)))
    )


def _video(uid, name, path) -> str:
    return (
        "    Video: %d, \"Video::%s\", \"Clip\" {\n"
        "        Type: \"Clip\"\n"
        "        Properties70:  {\n"
        "            P: \"Path\", \"KString\", \"XRefUrl\", \"\", \"%s\"\n"
        "        }\n"
        "        UseMipMap: 0\n"
        "        Filename: \"%s\"\n"
        "        RelativeFilename: \"%s\"\n"
        "    }"
        % (uid, name, _escape(path), _escape(path), _escape(os.path.basename(path)))
    )


def _bind_pose(uid, nodes: Sequence[Tuple[int, Sequence[float]]]) -> str:
    parts = [
        "    Pose: %d, \"Pose::BindPose\", \"BindPose\" {\n" % uid,
        "        Type: \"BindPose\"\n",
        "        Version: 100\n",
        "        NbPoseNodes: %d\n" % len(nodes),
    ]
    for nid, matrix in nodes:
        parts.append(
            "        PoseNode:  {\n"
            "            Node: %d\n"
            "            Matrix: *16 {\n"
            "                a: %s\n"
            "            }\n"
            "        }\n"
            % (nid, _fmt_list(matrix))
        )
    parts.append("    }")
    return "".join(parts)


def _anim_stack(uid, name, duration_s) -> str:
    stop = _sec_to_ktime(duration_s)
    return (
        "    AnimationStack: %d, \"AnimStack::%s\", \"\" {\n"
        "        Properties70:  {\n"
        "            P: \"LocalStart\", \"KTime\", \"Time\", \"\",0\n"
        "            P: \"LocalStop\", \"KTime\", \"Time\", \"\",%d\n"
        "            P: \"ReferenceStart\", \"KTime\", \"Time\", \"\",0\n"
        "            P: \"ReferenceStop\", \"KTime\", \"Time\", \"\",%d\n"
        "        }\n"
        "    }"
        % (uid, name, stop, stop)
    )


def _anim_layer(uid, name) -> str:
    return "    AnimationLayer: %d, \"AnimLayer::%s\", \"\" {\n    }" % (uid, name)


def _anim_curve(uid, times: Sequence[int], values: Sequence[float]) -> str:
    n = len(times)
    # 24836 = eInterpolationLinear | standard Autodesk tangent bits.
    # 24840 is cubic; Unity resamples it as flat curves ("no action").
    return (
        "    AnimationCurve: %d, \"AnimCurve::\", \"\" {\n"
        "        Default: %s\n"
        "        KeyVer: 4008\n"
        "        KeyTime: *%d {\n"
        "            a: %s\n"
        "        }\n"
        "        KeyValueFloat: *%d {\n"
        "            a: %s\n"
        "        }\n"
        "        KeyAttrFlags: *1 {\n"
        "            a: 24836\n"
        "        }\n"
        "        KeyAttrDataFloat: *4 {\n"
        "            a: 0,0,0,0\n"
        "        }\n"
        "        KeyAttrRefCount: *1 {\n"
        "            a: %d\n"
        "        }\n"
        "    }"
        % (
            uid,
            _fmt(values[0] if values else 0.0),
            n,
            _fmt_ints(times),
            n,
            _fmt_list(values),
            n,
        )
    )


def _takes(clips: Sequence[BakedClip]) -> str:
    """Unity discovers clips from the Takes block; Blender uses AnimationStack."""
    current = clips[0].name if clips else ""
    lines = ["Takes:  {", '    Current: "%s"' % _escape(current)]
    for clip in clips:
        stop = _sec_to_ktime(clip.duration_s)
        lines.append('    Take: "%s" {' % _escape(clip.name))
        lines.append('        FileName: "%s.tak"' % _escape(clip.name))
        lines.append("        LocalTime: 0,%d" % stop)
        lines.append("        ReferenceTime: 0,%d" % stop)
        lines.append("    }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _connections(conns: Sequence[Tuple], _root_id: int) -> str:
    lines = ["Connections:  {"]
    for c in conns:
        if len(c) == 3:
            typ, a, b = c
            lines.append('    C: "%s",%d,%d' % (typ, a, b))
        else:
            typ, a, b, prop = c
            lines.append('    C: "%s",%d,%d, "%s"' % (typ, a, b, prop))
    lines.append("}")
    return "\n".join(lines) + "\n"


def _escape(s: str) -> str:
    return s.replace("\\", "/").replace('"', "'")
