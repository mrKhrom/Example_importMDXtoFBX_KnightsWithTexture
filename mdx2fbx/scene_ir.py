"""Intermediate scene: nodes, meshes, clips, materials."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from animation import (
    BakedClip,
    bake_node_clip,
    bake_visibility,
    collect_key_times,
    eval_alpha,
    frame_times_ms,
    local_trs,
    sanitize_clip_name,
)
from mdx_parser import (
    FILTER_NAMES,
    MdxModel,
    Node,
    Track,
)
from skinning import Influence, geoset_influences
from xform import (
    IDENTITY_QUAT,
    Quat,
    Vec3,
    apply_axis_matrix,
    apply_axis_point,
    mat4_decompose,
    mat4_from_trs,
    mat4_invert,
    mat4_mul,
    rest_local_translation,
    vlen,
)

log = logging.getLogger("mdx2fbx.ir")


@dataclass
class IrNode:
    object_id: int
    parent_id: Optional[int]
    name: str
    sanitized: str
    role: str  # bone, helper, attachment
    flags: int
    billboarded: bool
    rest_t: Vec3
    rest_r: Quat
    rest_s: Vec3
    tracks: List[Track]
    wc3_name: str
    # After axis conversion these replace rest_* for export.
    export_t: Vec3 = (0.0, 0.0, 0.0)
    export_r: Quat = IDENTITY_QUAT
    export_s: Vec3 = (1.0, 1.0, 1.0)


@dataclass
class IrTexture:
    index: int
    filename: str
    replaceable_id: int
    resolved_path: Optional[str] = None


@dataclass
class IrMaterial:
    index: int
    name: str
    texture_index: Optional[int]
    filter_mode: int
    two_sided: bool
    unshaded: bool
    alpha: float


@dataclass
class IrMesh:
    index: int
    name: str
    positions: List[Vec3]
    normals: List[Vec3]
    uvs: List[Tuple[float, float]]
    indices: List[int]
    material_id: int
    influences: List[List[Influence]]
    visibility_tracks: List[Track]
    default_alpha: float
    visible_at_rest: bool = True


@dataclass
class IrClip:
    name: str
    raw_name: str
    start_ms: int
    end_ms: int
    looping: bool
    move_speed: float
    is_global: bool = False
    global_index: Optional[int] = None


@dataclass
class ConvertOptions:
    axis: str = "unity"
    fps: float = 30.0
    max_influences: int = 4
    include_helpers: bool = True
    include_attachments: bool = True
    skip_particles: bool = True
    bake_animations: bool = True
    geoset_filter: str = "stand"  # stand | all
    explode_mesh: bool = True
    texture: Optional[str] = None
    texture_dir: Optional[str] = None
    source_dir: str = ""


@dataclass
class IrScene:
    name: str
    nodes: List[IrNode]
    meshes: List[IrMesh]
    materials: List[IrMaterial]
    textures: List[IrTexture]
    clips: List[IrClip]
    global_sequences: List[int]
    nodes_by_id: Dict[int, IrNode] = field(default_factory=dict)
    bind_world: Dict[int, List[List[float]]] = field(default_factory=dict)

    def verify_bind_pivots(
        self, pivots: Sequence[Vec3], ids: Sequence[int], eps: float = 1e-3
    ) -> List[str]:
        """Check world(bind) ≈ pivot for selected object ids (WC3 space)."""
        errors = []
        world = compute_world_bind({n.object_id: n for n in self.nodes})
        for oid in ids:
            if oid not in world or oid >= len(pivots):
                errors.append("id %d missing" % oid)
                continue
            pos = (world[oid][0][3], world[oid][1][3], world[oid][2][3])
            piv = pivots[oid]
            dist = vlen((pos[0] - piv[0], pos[1] - piv[1], pos[2] - piv[2]))
            if dist > eps:
                errors.append(
                    "id %d world=%s pivot=%s dist=%.6f" % (oid, pos, piv, dist)
                )
        return errors


def compute_world_bind(nodes: Dict[int, IrNode]) -> Dict[int, List[List[float]]]:
    world: Dict[int, List[List[float]]] = {}

    def rec(oid: int) -> List[List[float]]:
        if oid in world:
            return world[oid]
        node = nodes[oid]
        local = mat4_from_trs(node.rest_t, node.rest_r, node.rest_s)
        if node.parent_id is None or node.parent_id not in nodes:
            world[oid] = local
            return local
        parent = rec(node.parent_id)
        # DontInherit is irrelevant for identity rest rotations, but honour T.
        w = mat4_mul(parent, local)
        world[oid] = w
        return w

    for oid in nodes:
        rec(oid)
    return world


def _unique_name(base: str, used: Set[str]) -> str:
    name = base or "Node"
    if name not in used:
        used.add(name)
        return name
    i = 2
    while True:
        cand = "%s_%d" % (name, i)
        if cand not in used:
            used.add(cand)
            return cand
        i += 1


def sanitize_node_name(raw: str, used: Set[str]) -> str:
    cleaned = raw.replace("\\", "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9_\- ]+", "_", cleaned)
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "Node"
    if cleaned[0].isdigit():
        cleaned = "N_" + cleaned
    return _unique_name(cleaned, used)


def _pick_diffuse_layer(model: MdxModel, material_index: int):
    if material_index < 0 or material_index >= len(model.materials):
        return None
    mat = model.materials[material_index]
    if not mat.layers:
        return None
    # Prefer a layer whose texture is a real file, not team color/glow.
    for layer in reversed(mat.layers):
        if 0 <= layer.texture_id < len(model.textures):
            tex = model.textures[layer.texture_id]
            if tex.filename or tex.replaceable_id == 0:
                return layer
    return mat.layers[-1]


def _resolve_texture(
    filename: str,
    replaceable_id: int,
    options: ConvertOptions,
    is_primary: bool,
) -> Optional[str]:
    search_dirs = []
    if options.texture_dir:
        search_dirs.append(options.texture_dir)
    if options.source_dir:
        search_dirs.append(options.source_dir)

    if is_primary and options.texture and os.path.isfile(options.texture):
        return os.path.abspath(options.texture)

    stem = os.path.splitext(os.path.basename(filename.replace("\\", "/")))[0]
    candidates = []
    if stem:
        for ext in (".png", ".tga", ".jpg", ".jpeg", ".bmp"):
            candidates.append(stem + ext)
        candidates.append(stem + "Texture.png")
        candidates.append("Textures_" + stem + ".png")
        # Knight.blp -> KnightTexture.png
        candidates.append(stem + "Texture.png")
    if replaceable_id == 1:
        candidates.append("TeamColor.png")
    if replaceable_id == 2:
        candidates.append("TeamGlow.png")

    for directory in search_dirs:
        # Also try any *Texture.png next to the model for the primary blp.
        if is_primary:
            try:
                for fn in os.listdir(directory):
                    low = fn.lower()
                    if low.endswith(".png") and "texture" in low:
                        candidates.insert(0, fn)
            except OSError:
                pass
        for name in candidates:
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                return os.path.abspath(path)
        if stem:
            # Knight.blp specifically often ships as KnightTexture.png
            guess = os.path.join(directory, stem + "Texture.png")
            if os.path.isfile(guess):
                return os.path.abspath(guess)
    return None


def build_ir(model: MdxModel, options: ConvertOptions) -> IrScene:
    used_names: Set[str] = set()
    nodes: List[IrNode] = []

    def add_node(node: Node, role: str) -> None:
        if role == "helper" and not options.include_helpers:
            return
        if role == "attachment" and not options.include_attachments:
            return
        pivot = (
            model.pivots[node.object_id]
            if 0 <= node.object_id < len(model.pivots)
            else (0.0, 0.0, 0.0)
        )
        parent_pivot = None
        if node.parent_id is not None and 0 <= node.parent_id < len(model.pivots):
            parent_pivot = model.pivots[node.parent_id]
        rest_t = rest_local_translation(pivot, parent_pivot)
        irn = IrNode(
            object_id=node.object_id,
            parent_id=node.parent_id,
            name=node.name,
            sanitized=sanitize_node_name(node.name, used_names),
            role=role,
            flags=node.flags,
            billboarded=node.billboarded,
            rest_t=rest_t,
            rest_r=IDENTITY_QUAT,
            rest_s=(1.0, 1.0, 1.0),
            tracks=list(node.tracks),
            wc3_name=node.name,
        )
        nodes.append(irn)

    for bone in model.bones:
        add_node(bone.node, "bone")
    if options.include_helpers:
        for helper in model.helpers:
            add_node(helper.node, "helper")
    if options.include_attachments:
        for att in model.attachments:
            add_node(att.node, "attachment")

    nodes.sort(key=lambda n: n.object_id)
    nodes_by_id = {n.object_id: n for n in nodes}

    # Drop parent links that are not exported (particles, events, etc.).
    exported = set(nodes_by_id)
    for n in nodes:
        if n.parent_id is not None and n.parent_id not in exported:
            log.info(
                "reparent %s (%d): parent %d not exported -> root",
                n.sanitized,
                n.object_id,
                n.parent_id,
            )
            # Rest translation was pivot - parent_pivot. If parent is missing,
            # use the world pivot as local so the bone stays in place.
            if 0 <= n.object_id < len(model.pivots):
                n.rest_t = model.pivots[n.object_id]
            n.parent_id = None

    # Textures / materials
    textures: List[IrTexture] = []
    primary_tex_index = 0
    for i, tex in enumerate(model.textures):
        is_primary = i == 0 or tex.filename.lower().endswith("knight.blp")
        resolved = _resolve_texture(
            tex.filename, tex.replaceable_id, options, is_primary=is_primary
        )
        textures.append(
            IrTexture(i, tex.filename, tex.replaceable_id, resolved)
        )

    materials: List[IrMaterial] = []
    for i, mat in enumerate(model.materials):
        layer = _pick_diffuse_layer(model, i)
        tex_index = layer.texture_id if layer is not None else None
        filt = layer.filter_mode if layer is not None else 0
        two = layer.two_sided if layer is not None else False
        unsh = layer.unshaded if layer is not None else False
        alpha = layer.alpha if layer is not None else 1.0
        materials.append(
            IrMaterial(
                index=i,
                name="Mat_%d" % i,
                texture_index=tex_index,
                filter_mode=filt,
                two_sided=two,
                unshaded=unsh,
                alpha=alpha,
            )
        )

    bone_index_to_id = model.bone_index_to_object_id()
    geoanims = {ga.geoset_id: ga for ga in model.geoset_animations}
    rest_time, rest_end = _stand_sample_interval(model)

    meshes: List[IrMesh] = []
    skipped = 0
    for g in model.geosets:
        infl, _ = geoset_influences(g, bone_index_to_id, options.max_influences)
        uvs = g.uvs[0] if g.uvs else [(0.0, 0.0)] * len(g.positions)
        ga = geoanims.get(g.index)
        vis_tracks = ga.tracks if ga is not None else []
        default_alpha = ga.alpha if ga is not None else 1.0
        # Clip-local: no key in Stand → static GeosetAnimation alpha (usually 1).
        # Global hold would leak Death/Portrait keys (0) and drop the sword/lance.
        alpha_rest = eval_alpha(
            vis_tracks,
            rest_time,
            default_alpha,
            clip_start=rest_time,
            clip_end=rest_end,
        )
        visible = alpha_rest >= 0.5
        if options.geoset_filter == "stand" and not visible:
            skipped += 1
            log.info(
                "skip GEOS[%d] hidden at Stand t=%d (alpha=%.2f)",
                g.index,
                rest_time,
                alpha_rest,
            )
            continue
        meshes.append(
            IrMesh(
                index=g.index,
                name="Geoset_%02d" % g.index,
                positions=list(g.positions),
                normals=list(g.normals),
                uvs=list(uvs),
                indices=list(g.indices),
                material_id=g.material_id,
                influences=infl,
                visibility_tracks=vis_tracks,
                default_alpha=default_alpha,
                visible_at_rest=visible,
            )
        )
    if skipped:
        log.info("geoset filter %r: exported %d, skipped %d", options.geoset_filter, len(meshes), skipped)

    used_clips: Set[str] = set()
    clips: List[IrClip] = []
    for seq in model.sequences:
        clips.append(
            IrClip(
                name=sanitize_clip_name(seq.name, used_clips),
                raw_name=seq.name,
                start_ms=seq.start,
                end_ms=seq.end,
                looping=seq.looping,
                move_speed=seq.move_speed,
            )
        )
    for gi, dur in enumerate(model.global_sequences):
        clips.append(
            IrClip(
                name=sanitize_clip_name("GlobalSeq_%d" % gi, used_clips),
                raw_name="GlobalSeq_%d" % gi,
                start_ms=0,
                end_ms=dur,
                looping=True,
                move_speed=0.0,
                is_global=True,
                global_index=gi,
            )
        )

    scene = IrScene(
        name=model.name or "Model",
        nodes=nodes,
        meshes=meshes,
        materials=materials,
        textures=textures,
        clips=clips,
        global_sequences=list(model.global_sequences),
        nodes_by_id=nodes_by_id,
    )
    scene.bind_world = compute_world_bind(nodes_by_id)
    apply_axis_to_scene(scene, options.axis)
    if options.explode_mesh:
        for mesh in scene.meshes:
            explode_mesh(mesh)
    return scene


def _stand_sample_interval(model: MdxModel) -> tuple:
    """Stand (or Walk) interval used to decide which geosets are in the default pose."""
    for seq in model.sequences:
        low = seq.name.lower()
        if low.startswith("stand") and "ready" not in low and "victory" not in low:
            return seq.start, seq.end
    for seq in model.sequences:
        if seq.name.lower().startswith("walk"):
            return seq.start, seq.end
    return 0, 0


def explode_mesh(mesh: IrMesh) -> None:
    """Unweld triangles so each corner is a unique control point.

    Unity's FBX importer mishandles indexed meshes with ByVertice mapping
    and then attaches skin clusters to the wrong vertices.
    """
    new_pos: List[Vec3] = []
    new_nrm: List[Vec3] = []
    new_uv: List[Tuple[float, float]] = []
    new_idx: List[int] = []
    new_inf: List[List[Influence]] = []
    for i in range(0, len(mesh.indices) - 2, 3):
        for k in range(3):
            old = mesh.indices[i + k]
            new_idx.append(len(new_pos))
            new_pos.append(mesh.positions[old])
            if old < len(mesh.normals):
                new_nrm.append(mesh.normals[old])
            else:
                new_nrm.append((0.0, 1.0, 0.0))
            if old < len(mesh.uvs):
                new_uv.append(mesh.uvs[old])
            else:
                new_uv.append((0.0, 0.0))
            if old < len(mesh.influences):
                new_inf.append(list(mesh.influences[old]))
            else:
                new_inf.append([])
    mesh.positions = new_pos
    mesh.normals = new_nrm
    mesh.uvs = new_uv
    mesh.indices = new_idx
    mesh.influences = new_inf


def write_obj(path: str, scene: IrScene) -> None:
    """Write a bind-pose OBJ (no skin) for visual verification."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# mdx2fbx bind-pose preview\n")
        v_off = 1
        for mesh in scene.meshes:
            handle.write("o %s\n" % mesh.name)
            handle.write("g %s\n" % mesh.name)
            for p in mesh.positions:
                handle.write("v %.6f %.6f %.6f\n" % p)
            for n in mesh.normals:
                handle.write("vn %.6f %.6f %.6f\n" % n)
            for u, v in mesh.uvs:
                handle.write("vt %.6f %.6f\n" % (u, 1.0 - v))
            idx = mesh.indices
            for i in range(0, len(idx) - 2, 3):
                a, b, c = idx[i] + v_off, idx[i + 1] + v_off, idx[i + 2] + v_off
                handle.write("f %d/%d/%d %d/%d/%d %d/%d/%d\n" % (a, a, a, b, b, b, c, c, c))
            v_off += len(mesh.positions)


def apply_axis_to_scene(scene: IrScene, mode: str) -> None:
    """Rewrite rest/export TRS and mesh data into the output axis system."""
    for node in scene.nodes:
        local = mat4_from_trs(node.rest_t, node.rest_r, node.rest_s)
        converted = apply_axis_matrix(local, mode)
        t, r, s = mat4_decompose(converted)
        node.export_t, node.export_r, node.export_s = t, r, s

    for mesh in scene.meshes:
        mesh.positions = [apply_axis_point(p, mode) for p in mesh.positions]
        mesh.normals = [apply_axis_point(n, mode) for n in mesh.normals]

    # Recompute bind world in export space from export_t/r/s.
    export_nodes = {}
    for n in scene.nodes:
        clone = IrNode(
            object_id=n.object_id,
            parent_id=n.parent_id,
            name=n.name,
            sanitized=n.sanitized,
            role=n.role,
            flags=n.flags,
            billboarded=n.billboarded,
            rest_t=n.export_t,
            rest_r=n.export_r,
            rest_s=n.export_s,
            tracks=n.tracks,
            wc3_name=n.wc3_name,
        )
        export_nodes[n.object_id] = clone
    scene.bind_world = compute_world_bind(export_nodes)


def bake_scene(scene: IrScene, options: ConvertOptions) -> List[BakedClip]:
    baked: List[BakedClip] = []
    for clip in scene.clips:
        if clip.is_global:
            # Only nodes whose tracks reference this global sequence.
            node_set = []
            for n in scene.nodes:
                for tr in n.tracks:
                    if tr.global_sequence == clip.global_index:
                        node_set.append(n)
                        break
            times = frame_times_ms(
                clip.start_ms, clip.end_ms, options.fps, options.bake_animations
            )
            nodes_out = []
            for n in node_set:
                # Evaluate using the global-sequence track only. We temporarily
                # treat KGSC/KGTR/KGRT with matching global id as regular by
                # cloning tracks with global_sequence cleared.
                fake = []
                for tr in n.tracks:
                    if tr.global_sequence == clip.global_index:
                        clone = Track(tr.tag, tr.interpolation, None, tr.keys)
                        fake.append(clone)
                bn = bake_node_clip(
                    n.object_id, n.rest_t, fake, clip.start_ms, clip.end_ms, times
                )
                if bn:
                    _convert_baked_node(bn, n, options.axis)
                    nodes_out.append(bn)
            baked.append(
                BakedClip(
                    clip.name,
                    (clip.end_ms - clip.start_ms) / 1000.0,
                    True,
                    nodes_out,
                    [],
                )
            )
            continue

        duration_s = (clip.end_ms - clip.start_ms) / 1000.0
        # Long WC3 clips (Decay*) blow up ASCII FBX; Unity then imports zero takes.
        bake = options.bake_animations and duration_s <= 8.0
        times = set(frame_times_ms(clip.start_ms, clip.end_ms, options.fps, bake))
        for n in scene.nodes:
            times.update(collect_key_times(n.tracks, clip.start_ms, clip.end_ms))
        times_ms = sorted(times)

        nodes_out = []
        for n in scene.nodes:
            if n.role == "attachment":
                continue
            bn = bake_node_clip(
                n.object_id,
                n.rest_t,
                n.tracks,
                clip.start_ms,
                clip.end_ms,
                times_ms,
                force=False,
            )
            if bn:
                _convert_baked_node(bn, n, options.axis)
                nodes_out.append(bn)

        vis = []
        for mesh in scene.meshes:
            bv = bake_visibility(
                mesh.index,
                mesh.visibility_tracks,
                mesh.default_alpha,
                clip.start_ms,
                clip.end_ms,
            )
            if bv:
                vis.append(bv)

        baked.append(
            BakedClip(
                clip.name,
                (clip.end_ms - clip.start_ms) / 1000.0,
                clip.looping,
                nodes_out,
                vis,
            )
        )
    return baked


def _convert_baked_node(bn, ir_node: IrNode, axis: str) -> None:
    """Convert each sample from WC3 local TRS into export-space TRS."""
    from animation import Sample

    prev_q = None
    prev_e = None
    for i, s in enumerate(bn.samples):
        local = mat4_from_trs(s.translation, s.rotation, s.scaling)
        converted = apply_axis_matrix(local, axis)
        t, r, sc = mat4_decompose(converted)
        if prev_q is not None:
            from xform import qdot, qneg

            if qdot(prev_q, r) < 0.0:
                r = qneg(r)
        prev_q = r
        from xform import continue_euler, quat_to_euler_xyz_deg

        e = continue_euler(prev_e, quat_to_euler_xyz_deg(r))
        prev_e = e
        bn.samples[i] = Sample(s.time_s, t, r, sc, e)


def _find_node(scene: IrScene, *needles: str) -> Optional[IrNode]:
    for n in scene.nodes:
        compact = n.wc3_name.replace(" ", "").lower()
        if all(s.lower() in compact for s in needles):
            return n
    return None


def world_translation_at(
    scene: IrScene,
    node_id: int,
    time_ms: int,
    clip_start: int,
    clip_end: int,
) -> Vec3:
    """WC3-space world translation (uses rest_t + clip-local tracks)."""
    chain: List[IrNode] = []
    cur: Optional[int] = node_id
    seen = set()
    while cur is not None and cur in scene.nodes_by_id and cur not in seen:
        seen.add(cur)
        node = scene.nodes_by_id[cur]
        chain.append(node)
        cur = node.parent_id
    chain.reverse()
    w = None
    for node in chain:
        t, r, s = local_trs(
            node.rest_t, node.tracks, time_ms, clip_start, clip_end
        )
        local = mat4_from_trs(t, r, s)
        w = local if w is None else mat4_mul(w, local)
    if w is None:
        return (0.0, 0.0, 0.0)
    return (w[0][3], w[1][3], w[2][3])


def mount_report(scene: IrScene, threshold: float = 40.0) -> List[str]:
    """Check rider root stays near the horse mount dummy in every clip."""
    rider = _find_node(scene, "bone_root") or _find_node(scene, "Bone_Root")
    horse = _find_node(scene, "abdomen01") or _find_node(scene, "ABDOMEN01")
    if rider is None or horse is None:
        return ["mount check skipped (Bone_Root / HorseBONE_ABDOMEN01 not found)"]
    bind_r = world_translation_at(scene, rider.object_id, 0, 0, 0)
    bind_h = world_translation_at(scene, horse.object_id, 0, 0, 0)
    bind_d = vlen(
        (bind_r[0] - bind_h[0], bind_r[1] - bind_h[1], bind_r[2] - bind_h[2])
    )
    lines = [
        "mount bind dist      %.2f  (rider %s  horse %s)"
        % (bind_d, rider.wc3_name.strip(), horse.wc3_name.strip())
    ]
    for clip in scene.clips:
        if clip.is_global:
            continue
        samples = [clip.start_ms]
        mid = (clip.start_ms + clip.end_ms) // 2
        samples.append(mid)
        samples.append(clip.end_ms)
        worst = 0.0
        worst_t = clip.start_ms
        for t in samples:
            rp = world_translation_at(
                scene, rider.object_id, t, clip.start_ms, clip.end_ms
            )
            hp = world_translation_at(
                scene, horse.object_id, t, clip.start_ms, clip.end_ms
            )
            d = vlen((rp[0] - hp[0], rp[1] - hp[1], rp[2] - hp[2]))
            if d > worst:
                worst, worst_t = d, t
        # Death/Decay Flesh may legitimately throw the rider; still report.
        status = "OK" if worst <= bind_d + threshold else "OFF-HORSE"
        if "death" in clip.name.lower() or (
            "decay" in clip.name.lower() and "flesh" in clip.raw_name.lower()
        ):
            status = "OK-death" if worst > bind_d + threshold else "OK"
        lines.append(
            "  %-18s %s  maxDist=%.1f at t=%d  (bind=%.1f)"
            % (clip.name, status, worst, worst_t, bind_d)
        )
    return lines


def node_path(scene: IrScene, node_id: int) -> str:
    names: List[str] = []
    cur: Optional[int] = node_id
    seen = set()
    while cur is not None and cur in scene.nodes_by_id and cur not in seen:
        seen.add(cur)
        names.append(scene.nodes_by_id[cur].sanitized)
        cur = scene.nodes_by_id[cur].parent_id
    names.reverse()
    return "/".join(names)


def write_anim_json(path: str, scene: IrScene, clips: Sequence[BakedClip]) -> None:
    """Sidecar used by Unity Editor to build AnimationClips if FBX takes fail."""
    payload = {"root": scene.name, "clips": []}
    for clip in clips:
        nodes = []
        for bn in clip.nodes:
            nodes.append(
                {
                    "id": bn.node_id,
                    "path": node_path(scene, bn.node_id),
                    "times": [s.time_s for s in bn.samples],
                    "tx": [s.translation[0] for s in bn.samples],
                    "ty": [s.translation[1] for s in bn.samples],
                    "tz": [s.translation[2] for s in bn.samples],
                    "rx": [s.euler_deg[0] for s in bn.samples],
                    "ry": [s.euler_deg[1] for s in bn.samples],
                    "rz": [s.euler_deg[2] for s in bn.samples],
                    "sx": [s.scaling[0] for s in bn.samples],
                    "sy": [s.scaling[1] for s in bn.samples],
                    "sz": [s.scaling[2] for s in bn.samples],
                }
            )
        payload["clips"].append(
            {
                "name": clip.name,
                "length": clip.duration_s,
                "loop": clip.looping,
                "nodes": nodes,
            }
        )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))


def bind_inverse(scene: IrScene, node_id: int):
    w = scene.bind_world.get(node_id)
    if w is None:
        from xform import identity4

        return identity4()
    return mat4_invert(w)
