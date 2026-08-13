"""Re-export ASCII FBX as binary via Blender so Unity's Animation tab works."""
import sys
from pathlib import Path

import bpy

src = Path(sys.argv[sys.argv.index("--") + 1])
dst = Path(sys.argv[sys.argv.index("--") + 2])

bpy.ops.wm.read_factory_settings(use_empty=True)
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

imported = False
errors = []
for op_name, kwargs in (
    ("import_scene.fbx", {"filepath": str(src), "automatic_bone_orientation": True, "use_anim": True}),
    ("wm.fbx_import", {"filepath": str(src)}),
):
    op = bpy.ops
    for part in op_name.split("."):
        op = getattr(op, part, None)
        if op is None:
            break
    if op is None:
        continue
    try:
        result = op(**kwargs)
        print("import", op_name, result)
        imported = "FINISHED" in result
        if imported:
            break
    except Exception as exc:
        errors.append("%s: %s" % (op_name, exc))

if not imported:
    raise SystemExit("FBX import failed: " + "; ".join(errors))

# Pack each action as an NLA strip so export writes one take per clip.
for obj in bpy.data.objects:
    if obj.animation_data and obj.animation_data.action:
        ad = obj.animation_data
        track = ad.nla_tracks.new()
        track.name = ad.action.name
        start = int(ad.action.frame_range[0])
        track.strips.new(ad.action.name, start, ad.action)

exported = False
for op_name, kwargs in (
    (
        "export_scene.fbx",
        dict(
            filepath=str(dst),
            check_existing=False,
            use_selection=False,
            apply_scale_options="FBX_SCALE_ALL",
            axis_forward="-Z",
            axis_up="Y",
            use_mesh_modifiers=True,
            add_leaf_bones=False,
            bake_anim=True,
            bake_anim_use_all_bones=True,
            bake_anim_use_nla_strips=True,
            bake_anim_use_all_actions=True,
            bake_anim_force_startend_keying=True,
            path_mode="AUTO",
        ),
    ),
    ("wm.fbx_export", {"filepath": str(dst)}),
):
    op = bpy.ops
    for part in op_name.split("."):
        op = getattr(op, part, None)
        if op is None:
            break
    if op is None:
        continue
    try:
        result = op(**kwargs)
        print("export", op_name, result)
        exported = "FINISHED" in result
        if exported:
            break
    except Exception as exc:
        errors.append("%s: %s" % (op_name, exc))

if not exported:
    raise SystemExit("FBX export failed: " + "; ".join(errors))

print("OK", dst, "bytes", dst.stat().st_size)
