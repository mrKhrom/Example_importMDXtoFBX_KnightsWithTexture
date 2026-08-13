"""Binary Warcraft 3 MDX parser (classic v800, tolerant of later versions)."""

from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass, field
from typing import BinaryIO, Dict, List, Optional, Tuple, Union

NO_ID = 0xFFFFFFFF

# Node flags
DONT_INHERIT_TRANSLATION = 0x1
DONT_INHERIT_ROTATION = 0x2
DONT_INHERIT_SCALING = 0x4
BILLBOARDED = 0x8
BILLBOARDED_LOCK_X = 0x10
BILLBOARDED_LOCK_Y = 0x20
BILLBOARDED_LOCK_Z = 0x40
CAMERA_ANCHORED = 0x80
FLAG_BONE = 0x100
FLAG_LIGHT = 0x200
FLAG_EVENT = 0x400
FLAG_ATTACHMENT = 0x800
FLAG_PARTICLE = 0x1000
FLAG_COLLISION = 0x2000
FLAG_RIBBON = 0x4000

INTERP_NONE = 0
INTERP_LINEAR = 1
INTERP_HERMITE = 2
INTERP_BEZIER = 3
INTERP_NAMES = {0: "None", 1: "Linear", 2: "Hermite", 3: "Bezier"}

FILTER_NAMES = {
    0: "None",
    1: "Transparent",
    2: "Blend",
    3: "Additive",
    4: "AddAlpha",
    5: "Modulate",
    6: "Modulate2x",
}

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]
Value = Union[float, int, Vec3, Quat]


class MDXParseError(Exception):
    pass


def _u32(data: bytes, off: int) -> int:
    return struct.unpack_from("<I", data, off)[0]


def _i32(data: bytes, off: int) -> int:
    return struct.unpack_from("<i", data, off)[0]


def _f32(data: bytes, off: int) -> float:
    return struct.unpack_from("<f", data, off)[0]


def _vec3(data: bytes, off: int) -> Vec3:
    return struct.unpack_from("<3f", data, off)


def _quat(data: bytes, off: int) -> Quat:
    return struct.unpack_from("<4f", data, off)


def _cstring(data: bytes, off: int, size: int) -> str:
    raw = data[off : off + size]
    return raw.split(b"\x00", 1)[0].decode("latin1", "replace")


def _tag(data: bytes, off: int) -> bytes:
    return data[off : off + 4]


# Track tag -> (kind, value_size)
# kind: "f3", "f4", "f1", "u1"
TRACK_KINDS: Dict[bytes, str] = {
    b"KGTR": "f3",
    b"KGRT": "f4",
    b"KGSC": "f3",
    b"KGAO": "f1",
    b"KGAC": "f3",
    b"KMTA": "f1",
    b"KMTF": "u1",
    b"KATV": "f1",
    b"KLAV": "f1",
    b"KLAC": "f3",
    b"KLAI": "f1",
    b"KLBC": "f3",
    b"KLBI": "f1",
    b"KPEV": "f1",
    b"KP2V": "f1",
    b"KP2E": "f1",
    b"KP2W": "f1",
    b"KP2N": "f1",
    b"KP2S": "f1",
    b"KRVS": "f1",
    b"KRHA": "f1",
    b"KRHB": "f1",
    b"KTAT": "f3",
    b"KTAR": "f4",
    b"KTAS": "f3",
    b"KCTR": "f3",
    b"KTTR": "f3",
    b"KCRL": "f1",
}

KIND_SIZE = {"f3": 12, "f4": 16, "f1": 4, "u1": 4}


@dataclass
class Keyframe:
    time: int
    value: Value
    in_tan: Optional[Value] = None
    out_tan: Optional[Value] = None


@dataclass
class Track:
    tag: str
    interpolation: int
    global_sequence: Optional[int]
    keys: List[Keyframe] = field(default_factory=list)

    @property
    def interp_name(self) -> str:
        return INTERP_NAMES.get(self.interpolation, str(self.interpolation))


@dataclass
class Extent:
    bounds_radius: float
    minimum: Vec3
    maximum: Vec3


@dataclass
class Sequence:
    name: str
    start: int
    end: int
    move_speed: float
    flags: int
    rarity: float
    sync_point: int
    extent: Extent

    @property
    def looping(self) -> bool:
        return (self.flags & 1) == 0

    @property
    def duration_ms(self) -> int:
        return max(0, self.end - self.start)


@dataclass
class Texture:
    replaceable_id: int
    filename: str
    flags: int


@dataclass
class Layer:
    filter_mode: int
    shading_flags: int
    texture_id: int
    texture_animation_id: Optional[int]
    coord_id: int
    alpha: float
    tracks: List[Track] = field(default_factory=list)

    @property
    def two_sided(self) -> bool:
        return bool(self.shading_flags & 16)

    @property
    def unshaded(self) -> bool:
        return bool(self.shading_flags & 1)


@dataclass
class Material:
    priority_plane: int
    flags: int
    layers: List[Layer] = field(default_factory=list)


@dataclass
class Geoset:
    index: int
    positions: List[Vec3]
    normals: List[Vec3]
    face_type: int
    indices: List[int]
    vertex_groups: List[int]
    matrix_group_sizes: List[int]
    matrix_indices: List[int]
    material_id: int
    selection_group: int
    selection_flags: int
    extent: Extent
    anim_extents: List[Extent]
    uvs: List[List[Tuple[float, float]]]


@dataclass
class GeosetAnimation:
    alpha: float
    flags: int
    color: Vec3
    geoset_id: int
    tracks: List[Track] = field(default_factory=list)


@dataclass
class Node:
    name: str
    object_id: int
    parent_id: Optional[int]
    flags: int
    tracks: List[Track] = field(default_factory=list)

    @property
    def billboarded(self) -> bool:
        return bool(self.flags & BILLBOARDED)

    @property
    def dont_inherit_translation(self) -> bool:
        return bool(self.flags & DONT_INHERIT_TRANSLATION)

    @property
    def dont_inherit_rotation(self) -> bool:
        return bool(self.flags & DONT_INHERIT_ROTATION)

    @property
    def dont_inherit_scaling(self) -> bool:
        return bool(self.flags & DONT_INHERIT_SCALING)


@dataclass
class Bone:
    node: Node
    geoset_id: Optional[int]
    geoset_anim_id: Optional[int]


@dataclass
class Helper:
    node: Node


@dataclass
class Attachment:
    node: Node
    path: str
    attachment_id: int
    tracks: List[Track] = field(default_factory=list)


@dataclass
class EventObject:
    node: Node
    times: List[int]
    global_sequence: Optional[int]


@dataclass
class ParticleEmitter2:
    node: Node


@dataclass
class RibbonEmitter:
    node: Node


@dataclass
class Camera:
    name: str


@dataclass
class CollisionShape:
    node: Node
    shape_type: int


@dataclass
class MdxModel:
    version: int
    name: str
    animation_file: str
    extent: Extent
    blend_time: int
    sequences: List[Sequence] = field(default_factory=list)
    global_sequences: List[int] = field(default_factory=list)
    textures: List[Texture] = field(default_factory=list)
    materials: List[Material] = field(default_factory=list)
    geosets: List[Geoset] = field(default_factory=list)
    geoset_animations: List[GeosetAnimation] = field(default_factory=list)
    bones: List[Bone] = field(default_factory=list)
    helpers: List[Helper] = field(default_factory=list)
    attachments: List[Attachment] = field(default_factory=list)
    particle_emitters2: List[ParticleEmitter2] = field(default_factory=list)
    ribbon_emitters: List[RibbonEmitter] = field(default_factory=list)
    cameras: List[Camera] = field(default_factory=list)
    event_objects: List[EventObject] = field(default_factory=list)
    collision_shapes: List[CollisionShape] = field(default_factory=list)
    pivots: List[Vec3] = field(default_factory=list)
    unknown_chunks: List[str] = field(default_factory=list)

    def node_by_id(self) -> Dict[int, Node]:
        out: Dict[int, Node] = {}
        for bone in self.bones:
            out[bone.node.object_id] = bone.node
        for helper in self.helpers:
            out[helper.node.object_id] = helper.node
        for att in self.attachments:
            out[att.node.object_id] = att.node
        for pe in self.particle_emitters2:
            out[pe.node.object_id] = pe.node
        for rb in self.ribbon_emitters:
            out[rb.node.object_id] = rb.node
        for ev in self.event_objects:
            out[ev.node.object_id] = ev.node
        for cs in self.collision_shapes:
            out[cs.node.object_id] = cs.node
        return out

    def bone_index_to_object_id(self) -> List[int]:
        return [b.node.object_id for b in self.bones]


def parse_mdx(path: str) -> MdxModel:
    with open(path, "rb") as handle:
        data = handle.read()
    return parse_mdx_bytes(data)


def parse_mdx_bytes(data: bytes) -> MdxModel:
    if len(data) < 8 or data[:4] != b"MDLX":
        raise MDXParseError("not an MDX file (missing MDLX magic)")

    chunks = _split_chunks(data)
    model = MdxModel(
        version=800,
        name="",
        animation_file="",
        extent=Extent(0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        blend_time=0,
    )

    parsers = {
        b"VERS": _parse_vers,
        b"MODL": _parse_modl,
        b"SEQS": _parse_seqs,
        b"GLBS": _parse_glbs,
        b"TEXS": _parse_texs,
        b"MTLS": _parse_mtls,
        b"GEOS": _parse_geos,
        b"GEOA": _parse_geoa,
        b"BONE": _parse_bone,
        b"HELP": _parse_help,
        b"ATCH": _parse_atch,
        b"PIVT": _parse_pivt,
        b"PRE2": _parse_pre2,
        b"RIBB": _parse_ribb,
        b"CAMS": _parse_cams,
        b"EVTS": _parse_evts,
        b"CLID": _parse_clid,
        b"PREM": _parse_skip,
        b"LITE": _parse_skip,
        b"TXAN": _parse_skip,
    }

    for tag, payload, file_off in chunks:
        fn = parsers.get(tag)
        if fn is None:
            try:
                model.unknown_chunks.append(tag.decode("ascii"))
            except UnicodeDecodeError:
                model.unknown_chunks.append(repr(tag))
            continue
        try:
            fn(model, payload, file_off)
        except MDXParseError:
            raise
        except Exception as exc:
            raise MDXParseError(
                "failed parsing chunk %s at offset %d: %s" % (tag, file_off, exc)
            ) from exc

    return model


def _split_chunks(data: bytes) -> List[Tuple[bytes, bytes, int]]:
    off = 4
    out: List[Tuple[bytes, bytes, int]] = []
    n = len(data)
    while off + 8 <= n:
        tag = data[off : off + 4]
        size = _u32(data, off + 4)
        start = off + 8
        end = start + size
        if end > n:
            raise MDXParseError(
                "chunk %r at %d size %d overruns file (%d bytes)"
                % (tag, off, size, n)
            )
        out.append((tag, data[start:end], start))
        off = end
    return out


def _parse_vers(model: MdxModel, payload: bytes, _off: int) -> None:
    if len(payload) < 4:
        raise MDXParseError("VERS chunk too small")
    model.version = _u32(payload, 0)


def _parse_modl(model: MdxModel, payload: bytes, _off: int) -> None:
    if len(payload) < 80 + 260 + 28 + 4:
        raise MDXParseError("MODL chunk too small")
    model.name = _cstring(payload, 0, 80)
    model.animation_file = _cstring(payload, 80, 260)
    p = 340
    br = _f32(payload, p)
    mn = _vec3(payload, p + 4)
    mx = _vec3(payload, p + 16)
    model.extent = Extent(br, mn, mx)
    model.blend_time = _u32(payload, p + 28)


def _parse_seqs(model: MdxModel, payload: bytes, _off: int) -> None:
    rec = 132
    if len(payload) % rec != 0:
        raise MDXParseError("SEQS size %d is not a multiple of 132" % len(payload))
    for i in range(len(payload) // rec):
        p = i * rec
        name = _cstring(payload, p, 80)
        start, end = struct.unpack_from("<II", payload, p + 80)
        move, flags, rarity, sync = struct.unpack_from("<fIfI", payload, p + 88)
        br = _f32(payload, p + 104)
        mn = _vec3(payload, p + 108)
        mx = _vec3(payload, p + 120)
        model.sequences.append(
            Sequence(name, start, end, move, flags, rarity, sync, Extent(br, mn, mx))
        )


def _parse_glbs(model: MdxModel, payload: bytes, _off: int) -> None:
    if len(payload) % 4 != 0:
        raise MDXParseError("GLBS size %d is not a multiple of 4" % len(payload))
    model.global_sequences = [
        _u32(payload, i * 4) for i in range(len(payload) // 4)
    ]


def _parse_texs(model: MdxModel, payload: bytes, _off: int) -> None:
    rec = 268
    if len(payload) % rec != 0:
        raise MDXParseError("TEXS size %d is not a multiple of 268" % len(payload))
    for i in range(len(payload) // rec):
        p = i * rec
        rid = _u32(payload, p)
        fname = _cstring(payload, p + 4, 260)
        flags = _u32(payload, p + 264)
        model.textures.append(Texture(rid, fname, flags))


def _read_value(kind: str, data: bytes, off: int) -> Tuple[Value, int]:
    if kind == "f3":
        return _vec3(data, off), off + 12
    if kind == "f4":
        return _quat(data, off), off + 16
    if kind == "f1":
        return _f32(data, off), off + 4
    if kind == "u1":
        return _u32(data, off), off + 4
    raise MDXParseError("unknown track kind %s" % kind)


def try_parse_track(data: bytes, off: int, end: int) -> Optional[Tuple[Track, int]]:
    if off + 4 > end:
        return None
    tag = data[off : off + 4]
    if tag == b"KEVT":
        if off + 12 > end:
            return None
        count = _u32(data, off + 4)
        gsid = _u32(data, off + 8)
        need = 12 + count * 4
        if off + need > end:
            raise MDXParseError("truncated KEVT at %d" % off)
        times = [_u32(data, off + 12 + i * 4) for i in range(count)]
        keys = [Keyframe(t, 0.0) for t in times]
        glob = None if gsid == NO_ID else gsid
        return Track("KEVT", INTERP_NONE, glob, keys), off + need
    if tag not in TRACK_KINDS:
        return None
    if off + 16 > end:
        raise MDXParseError("truncated track %s at %d" % (tag, off))
    kind = TRACK_KINDS[tag]
    ntr, itype, gsid = struct.unpack_from("<III", data, off + 4)
    extra = 2 if itype > 1 else 0
    vsz = KIND_SIZE[kind]
    rec = 4 + vsz * (1 + extra)
    need = 16 + ntr * rec
    if off + need > end:
        raise MDXParseError(
            "truncated track %s at %d: need %d bytes, have %d"
            % (tag, off, need, end - off)
        )
    keys: List[Keyframe] = []
    p = off + 16
    for _ in range(ntr):
        time = _u32(data, p)
        p += 4
        value, p = _read_value(kind, data, p)
        in_tan = out_tan = None
        if extra:
            in_tan, p = _read_value(kind, data, p)
            out_tan, p = _read_value(kind, data, p)
        keys.append(Keyframe(time, value, in_tan, out_tan))
    glob = None if gsid == NO_ID else gsid
    return Track(tag.decode("ascii"), itype, glob, keys), p


def _parse_track_list(data: bytes, off: int, end: int) -> Tuple[List[Track], int]:
    tracks: List[Track] = []
    p = off
    while p < end:
        parsed = try_parse_track(data, p, end)
        if parsed is None:
            break
        track, p = parsed
        tracks.append(track)
    return tracks, p


def _parse_node(data: bytes, off: int, end: int) -> Tuple[Node, int]:
    if off + 96 > end:
        raise MDXParseError("truncated Node at %d" % off)
    incl = _u32(data, off)
    nend = off + incl
    if nend > end:
        raise MDXParseError("Node InclusiveSize %d overruns parent at %d" % (incl, off))
    name = _cstring(data, off + 4, 80)
    oid, pid, flags = struct.unpack_from("<III", data, off + 84)
    tracks, p = _parse_track_list(data, off + 96, nend)
    if p != nend:
        # leftover bytes inside node are unusual but not fatal
        pass
    parent = None if pid == NO_ID else pid
    return Node(name, oid, parent, flags, tracks), nend


def _parse_mtls(model: MdxModel, payload: bytes, file_off: int) -> None:
    p = 0
    end = len(payload)
    while p < end:
        if p + 12 > end:
            raise MDXParseError("truncated material at %d" % (file_off + p))
        incl = _u32(payload, p)
        mend = p + incl
        if mend > end:
            raise MDXParseError("material InclusiveSize overruns MTLS")
        prio, flags = struct.unpack_from("<II", payload, p + 4)
        q = p + 12
        if payload[q : q + 4] != b"LAYS":
            raise MDXParseError("expected LAYS at %d" % (file_off + q))
        nlayers = _u32(payload, q + 4)
        q += 8
        layers: List[Layer] = []
        for _ in range(nlayers):
            if q + 28 > mend:
                raise MDXParseError("truncated layer at %d" % (file_off + q))
            lincl = _u32(payload, q)
            lend = q + lincl
            fm, sh, tid, taid, cid = struct.unpack_from("<IIIII", payload, q + 4)
            alpha = _f32(payload, q + 24)
            tracks, _ = _parse_track_list(payload, q + 28, lend)
            ta = None if taid == NO_ID else taid
            layers.append(Layer(fm, sh, tid, ta, cid, alpha, tracks))
            q = lend
        model.materials.append(Material(prio, flags, layers))
        p = mend


def _expect_tag(payload: bytes, q: int, tag: bytes, geoset_i: int, file_off: int) -> None:
    got = payload[q : q + 4]
    if got != tag:
        raise MDXParseError(
            "GEOS[%d] expected %s at offset %d, got %r"
            % (geoset_i, tag.decode("ascii"), file_off + q, got)
        )


def _parse_geos(model: MdxModel, payload: bytes, file_off: int) -> None:
    p = 0
    end = len(payload)
    gi = 0
    while p < end:
        if p + 4 > end:
            raise MDXParseError("truncated geoset header")
        incl = _u32(payload, p)
        gend = p + incl
        if gend > end:
            raise MDXParseError("GEOS[%d] InclusiveSize overruns chunk" % gi)
        q = p + 4

        _expect_tag(payload, q, b"VRTX", gi, file_off)
        nv = _u32(payload, q + 4)
        q += 8
        positions = [_vec3(payload, q + i * 12) for i in range(nv)]
        q += nv * 12

        _expect_tag(payload, q, b"NRMS", gi, file_off)
        nn = _u32(payload, q + 4)
        q += 8
        normals = [_vec3(payload, q + i * 12) for i in range(nn)]
        q += nn * 12

        _expect_tag(payload, q, b"PTYP", gi, file_off)
        ntg = _u32(payload, q + 4)
        q += 8
        face_types = [_u32(payload, q + i * 4) for i in range(ntg)]
        q += ntg * 4
        face_type = face_types[0] if face_types else 4

        _expect_tag(payload, q, b"PCNT", gi, file_off)
        nfg = _u32(payload, q + 4)
        q += 8
        q += nfg * 4

        _expect_tag(payload, q, b"PVTX", gi, file_off)
        nidx = _u32(payload, q + 4)
        q += 8
        indices = list(struct.unpack_from("<%dH" % nidx, payload, q))
        q += nidx * 2

        _expect_tag(payload, q, b"GNDX", gi, file_off)
        ng = _u32(payload, q + 4)
        q += 8
        vertex_groups = list(payload[q : q + ng])
        q += ng

        _expect_tag(payload, q, b"MTGC", gi, file_off)
        nmg = _u32(payload, q + 4)
        q += 8
        matrix_group_sizes = [_u32(payload, q + i * 4) for i in range(nmg)]
        q += nmg * 4

        _expect_tag(payload, q, b"MATS", gi, file_off)
        nmi = _u32(payload, q + 4)
        q += 8
        matrix_indices = [_u32(payload, q + i * 4) for i in range(nmi)]
        q += nmi * 4

        if q + 12 + 28 + 4 > gend:
            raise MDXParseError("GEOS[%d] truncated after MATS" % gi)
        matid, selg, selfl = struct.unpack_from("<III", payload, q)
        q += 12
        br = _f32(payload, q)
        mn = _vec3(payload, q + 4)
        mx = _vec3(payload, q + 16)
        q += 28
        nextents = _u32(payload, q)
        q += 4
        # Each sequence extent is BoundsRadius + Min + Max (28 bytes).
        anim_extents: List[Extent] = []
        for _ in range(nextents):
            if q + 28 > gend:
                raise MDXParseError("GEOS[%d] truncated sequence extents" % gi)
            ebr = _f32(payload, q)
            emn = _vec3(payload, q + 4)
            emx = _vec3(payload, q + 16)
            anim_extents.append(Extent(ebr, emn, emx))
            q += 28

        _expect_tag(payload, q, b"UVAS", gi, file_off)
        nuvg = _u32(payload, q + 4)
        q += 8
        uvs: List[List[Tuple[float, float]]] = []
        for _ in range(nuvg):
            _expect_tag(payload, q, b"UVBS", gi, file_off)
            nuv = _u32(payload, q + 4)
            q += 8
            channel = [struct.unpack_from("<2f", payload, q + i * 8) for i in range(nuv)]
            uvs.append(channel)
            q += nuv * 8

        # Skip optional Reforged TANG / SKIN if present.
        while q + 8 <= gend:
            tag = payload[q : q + 4]
            if tag == b"TANG":
                n = _u32(payload, q + 4)
                q += 8 + n * 16
            elif tag == b"SKIN":
                n = _u32(payload, q + 4)
                q += 8 + n
            else:
                break

        model.geosets.append(
            Geoset(
                index=gi,
                positions=positions,
                normals=normals,
                face_type=face_type,
                indices=indices,
                vertex_groups=vertex_groups,
                matrix_group_sizes=matrix_group_sizes,
                matrix_indices=matrix_indices,
                material_id=matid,
                selection_group=selg,
                selection_flags=selfl,
                extent=Extent(br, mn, mx),
                anim_extents=anim_extents,
                uvs=uvs,
            )
        )
        gi += 1
        p = gend


def _parse_geoa(model: MdxModel, payload: bytes, file_off: int) -> None:
    p = 0
    end = len(payload)
    while p < end:
        incl = _u32(payload, p)
        gend = p + incl
        if gend > end:
            raise MDXParseError("GEOA InclusiveSize overruns chunk at %d" % (file_off + p))
        alpha = _f32(payload, p + 4)
        flags = _u32(payload, p + 8)
        color = _vec3(payload, p + 12)
        gid = _u32(payload, p + 24)
        tracks, _ = _parse_track_list(payload, p + 28, gend)
        model.geoset_animations.append(
            GeosetAnimation(alpha, flags, color, gid, tracks)
        )
        p = gend


def _opt_id(value: int) -> Optional[int]:
    return None if value == NO_ID else value


def _parse_bone(model: MdxModel, payload: bytes, _off: int) -> None:
    p = 0
    end = len(payload)
    while p < end:
        node, q = _parse_node(payload, p, end)
        if q + 8 > end:
            raise MDXParseError("truncated Bone after node %r" % node.name)
        gid, gaid = struct.unpack_from("<II", payload, q)
        model.bones.append(Bone(node, _opt_id(gid), _opt_id(gaid)))
        p = q + 8


def _parse_help(model: MdxModel, payload: bytes, _off: int) -> None:
    p = 0
    end = len(payload)
    while p < end:
        node, p = _parse_node(payload, p, end)
        model.helpers.append(Helper(node))


def _parse_atch(model: MdxModel, payload: bytes, _off: int) -> None:
    p = 0
    end = len(payload)
    while p < end:
        incl = _u32(payload, p)
        aend = p + incl
        if aend > end:
            raise MDXParseError("attachment InclusiveSize overruns ATCH")
        node, q = _parse_node(payload, p + 4, aend)
        if q + 264 > aend:
            raise MDXParseError("truncated attachment %r" % node.name)
        path = _cstring(payload, q, 260)
        attid = _u32(payload, q + 260)
        tracks, _ = _parse_track_list(payload, q + 264, aend)
        model.attachments.append(Attachment(node, path, attid, tracks))
        p = aend


def _parse_pivt(model: MdxModel, payload: bytes, _off: int) -> None:
    if len(payload) % 12 != 0:
        raise MDXParseError("PIVT size %d is not a multiple of 12" % len(payload))
    model.pivots = [
        _vec3(payload, i * 12) for i in range(len(payload) // 12)
    ]


def _parse_wrapped_node_list(payload: bytes, ctor) -> list:
    items = []
    p = 0
    end = len(payload)
    while p < end:
        incl = _u32(payload, p)
        nend = p + incl
        if nend > end:
            raise MDXParseError("wrapped node InclusiveSize overruns chunk")
        node, _ = _parse_node(payload, p + 4, nend)
        items.append(ctor(node))
        p = nend
    return items


def _parse_pre2(model: MdxModel, payload: bytes, _off: int) -> None:
    model.particle_emitters2 = _parse_wrapped_node_list(payload, ParticleEmitter2)


def _parse_ribb(model: MdxModel, payload: bytes, _off: int) -> None:
    model.ribbon_emitters = _parse_wrapped_node_list(payload, RibbonEmitter)


def _parse_cams(model: MdxModel, payload: bytes, _off: int) -> None:
    p = 0
    end = len(payload)
    while p < end:
        incl = _u32(payload, p)
        nend = p + incl
        if nend > end:
            raise MDXParseError("camera InclusiveSize overruns CAMS")
        name = _cstring(payload, p + 4, 80)
        model.cameras.append(Camera(name))
        p = nend


def _parse_evts(model: MdxModel, payload: bytes, _off: int) -> None:
    p = 0
    end = len(payload)
    while p < end:
        node, q = _parse_node(payload, p, end)
        parsed = try_parse_track(payload, q, end)
        times: List[int] = []
        glob: Optional[int] = None
        if parsed is not None:
            track, q = parsed
            times = [k.time for k in track.keys]
            glob = track.global_sequence
        model.event_objects.append(EventObject(node, times, glob))
        p = q


def _parse_clid(model: MdxModel, payload: bytes, _off: int) -> None:
    p = 0
    end = len(payload)
    while p < end:
        node, q = _parse_node(payload, p, end)
        if q + 4 > end:
            raise MDXParseError("truncated collision shape")
        typ = _u32(payload, q)
        q += 4
        if typ == 0:
            q += 24
        elif typ == 2:
            q += 16
        else:
            # Unknown type: stop rather than desync.
            model.collision_shapes.append(CollisionShape(node, typ))
            break
        model.collision_shapes.append(CollisionShape(node, typ))
        p = q


def _parse_skip(_model: MdxModel, _payload: bytes, _off: int) -> None:
    return


def dump_model(model: MdxModel) -> str:
    """Human-readable statistics used as the acceptance check."""
    lines: List[str] = []
    ext = model.extent
    verts = sum(len(g.positions) for g in model.geosets)
    tris = sum(len(g.indices) // 3 for g in model.geosets)
    sizes: List[int] = []
    max_inf = 0
    for g in model.geosets:
        sizes.extend(g.matrix_group_sizes)
        if g.matrix_group_sizes:
            max_inf = max(max_inf, max(g.matrix_group_sizes))
    hist = dict(sorted(Counter(sizes).items()))

    bone_ids = [b.node.object_id for b in model.bones]
    helper_ids = [h.node.object_id for h in model.helpers]

    interp = Counter()
    glob_tracks = []
    billboards = []
    for bone in model.bones:
        for tr in bone.node.tracks:
            interp[tr.interp_name] += 1
            if tr.global_sequence is not None:
                glob_tracks.append((bone.node.name, tr.tag, tr.global_sequence))
        if bone.node.billboarded:
            billboards.append((bone.node.object_id, bone.node.name, "bone"))
    for helper in model.helpers:
        for tr in helper.node.tracks:
            interp[tr.interp_name] += 1
            if tr.global_sequence is not None:
                glob_tracks.append((helper.node.name, tr.tag, tr.global_sequence))
        if helper.node.billboarded:
            billboards.append((helper.node.object_id, helper.node.name, "helper"))

    lines.append("magic              MDLX")
    lines.append("version            %d" % model.version)
    lines.append("name               %s" % model.name)
    lines.append("bounds radius      %.6f" % ext.bounds_radius)
    lines.append(
        "AABB min           (%.6f, %.6f, %.6f)"
        % ext.minimum
    )
    lines.append(
        "AABB max           (%.6f, %.6f, %.6f)"
        % ext.maximum
    )
    lines.append("sequences          %d" % len(model.sequences))
    lines.append(
        "global sequences   %d %s"
        % (len(model.global_sequences), model.global_sequences)
    )
    lines.append("textures           %d" % len(model.textures))
    lines.append("materials          %d" % len(model.materials))
    two_layer = sum(1 for m in model.materials if len(m.layers) > 1)
    lines.append("  multi-layer      %d" % two_layer)
    lines.append("geosets            %d" % len(model.geosets))
    lines.append("vertices total     %d" % verts)
    lines.append("triangles total    %d" % tris)
    lines.append("geoset animations  %d" % len(model.geoset_animations))
    lines.append(
        "bones              %d  ObjectId %s"
        % (
            len(model.bones),
            ("%d..%d" % (min(bone_ids), max(bone_ids))) if bone_ids else "-",
        )
    )
    lines.append(
        "helpers            %d  ObjectId %s"
        % (
            len(model.helpers),
            ("%d..%d" % (min(helper_ids), max(helper_ids))) if helper_ids else "-",
        )
    )
    lines.append("attachments        %d" % len(model.attachments))
    lines.append("particle emitters2 %d" % len(model.particle_emitters2))
    lines.append("ribbon emitters    %d" % len(model.ribbon_emitters))
    lines.append(
        "cameras            %d %s"
        % (len(model.cameras), [c.name for c in model.cameras])
    )
    lines.append("event objects      %d" % len(model.event_objects))
    lines.append("collision shapes   %d" % len(model.collision_shapes))
    lines.append("pivot points       %d" % len(model.pivots))
    lines.append("max influences     %d" % max_inf)
    lines.append("matrix group hist  %s" % hist)
    lines.append("bone/helper interp %s" % dict(interp))
    lines.append("global-seq tracks  %d %s" % (len(glob_tracks), glob_tracks))
    lines.append("billboard          %s" % billboards)

    lines.append("")
    lines.append("Sequences:")
    for i, seq in enumerate(model.sequences):
        kind = "loop" if seq.looping else "oneshot"
        extra = ""
        if seq.move_speed:
            extra += " speed=%.1f" % seq.move_speed
        if seq.rarity:
            extra += " rarity=%.1f" % seq.rarity
        lines.append(
            "  [%02d] %-22s %6d-%-6d %s%s"
            % (i, seq.name, seq.start, seq.end, kind, extra)
        )

    lines.append("")
    lines.append("Textures:")
    for i, tex in enumerate(model.textures):
        lines.append(
            "  [%d] replaceable=%d file=%r flags=%d"
            % (i, tex.replaceable_id, tex.filename, tex.flags)
        )

    lines.append("")
    lines.append("Geosets:")
    for g in model.geosets:
        mi = max(g.matrix_group_sizes) if g.matrix_group_sizes else 0
        lines.append(
            "  [%02d] v=%4d f=%4d mat=%d groups=%d maxInf=%d uv=%d"
            % (
                g.index,
                len(g.positions),
                len(g.indices) // 3,
                g.material_id,
                len(g.matrix_group_sizes),
                mi,
                len(g.uvs),
            )
        )

    lines.append("")
    lines.append("Materials:")
    for i, mat in enumerate(model.materials):
        lines.append(
            "  [%d] layers=%d flags=%d"
            % (i, len(mat.layers), mat.flags)
        )
        for li, layer in enumerate(mat.layers):
            lines.append(
                "    L%d filter=%s(%d) shade=%d tex=%d alpha=%.3f"
                % (
                    li,
                    FILTER_NAMES.get(layer.filter_mode, "?"),
                    layer.filter_mode,
                    layer.shading_flags,
                    layer.texture_id,
                    layer.alpha,
                )
            )
    return "\n".join(lines)
