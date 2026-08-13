# mdx2fbx

Python 3.9+ converter: Warcraft 3 classic **MDX v800** → **ASCII FBX 7.4** for Unity, plus a Unity Editor script that finishes the import (rig, materials, clips, controller, prefab).

Exports geometry, skeleton (bones **and** helpers), skin weights, animation clips, UVs and materials. Does **not** recreate Warcraft particle emitters, ribbons, cameras or event sounds.

No Autodesk FBX SDK and no Blender required.

---

## Full process: MDX → Unity

Do these steps in order. Paths below match the Knight pack next to this folder.

### 1. Convert MDX to FBX

```bash
cd /Users/alekseykhromov/Downloads/KnightsWithTexture

python3 mdx2fbx/convert.py KnightV2.mdx  --texture KnightTexture.png
python3 mdx2fbx/convert.py KnightAV2.mdx --texture KnightTexture.png
python3 mdx2fbx/convert.py KnightBV2.mdx --texture KnightTexture.png
python3 mdx2fbx/convert.py KnightCV2.mdx --texture KnightTexture.png
```

Or every `.mdx` in the folder at once:

```bash
python3 mdx2fbx/convert.py /Users/alekseykhromov/Downloads/KnightsWithTexture \
  --batch \
  --texture /Users/alekseykhromov/Downloads/KnightsWithTexture/KnightTexture.png
```

Each run writes, next to the source MDX:

| File | What it is |
|---|---|
| `KnightV2.fbx` | Model + skeleton + 20 animation takes |
| `KnightV2.anim.json` | Sidecar clip list (debug / fallback) |
| `KnightV2.obj` | Bind-pose mesh only (no skin) |
| `TeamColor.png` / `TeamGlow.png` | Placeholders if the MDX uses replaceable textures |

`--dump` inspects a file without writing anything:

```bash
python3 mdx2fbx/convert.py KnightV2.mdx --dump
```

For `KnightV2.mdx` that must report version **800**, 19 geosets, 1540 verts, 1454 tris, 79 bones, 35 helpers, 19 sequences.

### 2. Open the Unity project

Open the target project in the Unity Editor (2022.3 LTS is what this pack was tested on). **Do not** start a second Editor on the same project — Unity locks the Library folder.

If the Editor is already open, leave it open. You will copy files into `Assets/` and let it refresh.

### 3. Create folders and copy files

In the Project window create (or reuse):

```
Assets/Models/Humans/Knight/
Assets/Models/Humans/Knight/Textures/
Assets/Models/Humans/Knight/Materials/
Assets/Scripts/Editor/
Assets/Prefabs/
```

Copy from `KnightsWithTexture/` into the Unity project **with these names** (the Editor script looks up `Knight.fbx`, not `KnightV2.fbx`):

| Source | Destination in Unity |
|---|---|
| `KnightV2.fbx` | `Assets/Models/Humans/Knight/Knight.fbx` |
| `KnightAV2.fbx` | `Assets/Models/Humans/Knight/KnightA.fbx` |
| `KnightBV2.fbx` | `Assets/Models/Humans/Knight/KnightB.fbx` |
| `KnightCV2.fbx` | `Assets/Models/Humans/Knight/KnightC.fbx` |
| `KnightTexture.png` | `Assets/Models/Humans/Knight/Textures/KnightTexture.png` |
| `mdx2fbx/unity/KnightModelPostprocessor.cs` | `Assets/Scripts/Editor/KnightModelPostprocessor.cs` |

The `.obj` / `.anim.json` files are optional. You do not need them for a normal import.

Wait until Unity finishes compiling the Editor script (progress bar / console). There must be no `CS0104` or other errors in `KnightModelPostprocessor.cs`.

### 4. Create the material

1. In `Assets/Models/Humans/Knight/Materials/` create a material named **`Knight`**.
2. Shader: Standard (or URP/Lit if the project uses URP).
3. Assign `Textures/KnightTexture.png` to **Albedo / Base Map**.

The postprocessor remaps every FBX slot (`Mat_0` … `Mat_11`, `lambert`, …) onto this one material.

### 5. Run the import script

Menu: **Tools → Setup Knight Model**

Run it **once**. It will:

- set Rig = **Generic**, Avatar = Create From This Model
- turn **Import Animation** on, **Weld Vertices** off, **Optimize Bones** off
- enable **Loop Time** on Walk / Stand_* / Portrait_* / GlobalSeq_0
- leave loop **off** on Attack_* / Death / Decay_* / Spell
- remap materials to `Knight.mat`
- write `Assets/Models/Humans/Knight/Knight.controller`
- write `Assets/Prefabs/Knight.prefab`

Do not mash the menu item. One pass is enough; the script no longer reimports in a loop.

### 6. Check the result

1. Select `Assets/Models/Humans/Knight/Knight.fbx`.
2. Inspector → **Rig**: Generic.
3. Inspector → **Animation**: 20 clips — `Walk`, `Stand_1`…`Stand_4`, `Stand_Ready`, `Stand_Victory`, `Attack_1`, `Attack_2`, `Death`, `Decay_Flesh`, `Decay_Bone`, `Portrait_1`…`4`, `Portrait_Talk_1`/`_2`, `Spell`, `GlobalSeq_0`.
4. Preview: rider sits on the horse in Stand / Attack / Portrait / Spell. Death / Decay_Flesh leave the rider (that is the Warcraft clip).
5. Drag **`Assets/Prefabs/Knight.prefab`** into the scene (not the raw FBX).
6. Play: Animator default state is `Stand_1`. Set float `Speed` > 0.1 to walk. Fire trigger `Attack` for `Attack_1`.

### 7. Manual import (if you skip the script)

If you do not copy `KnightModelPostprocessor.cs`, set the FBX yourself:

1. Select the FBX → **Model**: Scale Factor `1`, **Weld Vertices** off.
2. **Rig**: Generic. Avatar Definition: Create From This Model.
3. **Animation**: Import Animation on. Loop Time on idle / walk / stand / portrait only.
4. Skinned Mesh Renderer → Blend Weights: **4 Bones**.
5. Assign `Knight.mat` on every renderer.

---

## Quick convert commands

```bash
cd /Users/alekseykhromov/Downloads/KnightsWithTexture/mdx2fbx

python3 convert.py /Users/alekseykhromov/Downloads/KnightsWithTexture/KnightV2.mdx \
  -o /Users/alekseykhromov/Downloads/KnightsWithTexture/KnightV2.fbx \
  --texture /Users/alekseykhromov/Downloads/KnightsWithTexture/KnightTexture.png
```

```bash
python3 convert.py /Users/alekseykhromov/Downloads/KnightsWithTexture \
  --batch \
  --texture /Users/alekseykhromov/Downloads/KnightsWithTexture/KnightTexture.png
```

## CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--dump` | | Parse only, print statistics |
| `--axis unity\|raw` | `unity` | `unity` remaps WC3 `(x,y,z)` → `(y,z,x)` (stand on Y, face +Z) |
| `--all-geosets` | off | Also export death/portrait/guts geosets (hidden in Stand) |
| `--fps 30` | 30 | Bake sample rate |
| `--max-influences 4` | 4 | Clamp WC3 matrix groups for Unity |
| `--texture PATH` | | Force the main diffuse map (e.g. `KnightTexture.png`) |
| `--texture-dir DIR` | folder of the MDX | Where to look for `.png` replacements of `.blp` |
| `--no-bake-animations` | bake on | Use original key times only |
| `--no-include-attachments` | include | Skip attachment empties |

If the prefab comes in rotated, the FBX was likely exported with `--axis raw`. Re-export with `--axis unity` (default).

A sibling `.obj` is written next to each FBX. Open it in Blender or Preview to check bind-pose topology without skinning. If the OBJ looks correct but the FBX is scrambled, the problem is the Unity skin importer — keep **Weld Vertices** off.

## Unity Editor script

Path in this repo: [`unity/KnightModelPostprocessor.cs`](unity/KnightModelPostprocessor.cs)

Copy it to `Assets/Scripts/Editor/` in the Unity project. It only touches files under `Assets/Models/Humans/Knight/` named `Knight.fbx` / `KnightA.fbx` / `KnightB.fbx` / `KnightC.fbx`.

| Menu | What it does |
|---|---|
| **Tools → Setup Knight Model** | Reimport those four FBX files, apply clip loop flags, build controller + prefab |

On every import of those FBX files it also:

- forces Generic rig + Create From This Model
- remaps `Mat_*` / default FBX materials to `Materials/Knight.mat`
- sets Loop Time from the clip name
- sets SkinnedMeshRenderer quality to 4 bones

It does **not** call `SaveAndReimport` from `OnPostprocessAllAssets` (that used to reimport the model ~100 times).

## What transfers

- One mesh per geoset (so decay / portrait / gore can hide independently via Visibility curves).
- Full bone+helper hierarchy. On the reference Knight the horse/rider IK-style chain lives in **helpers**; dropping them breaks the legs.
- Skinning: WC3 matrix groups → equal weights, then clamp to `--max-influences`.
- Sequences on the global WC3 timeline are split into separate FBX takes.
- Tracks that use a Global Sequence become `GlobalSeq_N` (looping).
- Static takes with no bone motion (e.g. `Decay_Bone`) still get a hold translation on the scene root so Unity does not drop the take.

## Limitations

- Team colour / team glow layers become a placeholder PNG (`TeamColor.png` magenta, `TeamGlow.png` orange) or are skipped in favour of the real diffuse layer.
- Additive WC3 materials are only approximated (Lambert).
- Billboard nodes (`Plane01`, `Dummy02` on the reference) export as ordinary bones, tagged `wc3_billboard=1`. Face-camera in Unity yourself if you need the star sprite.
- Particles, ribbons, event objects (`SND*`, `FPT*`, `SPL*`), cameras and collision shapes are not turned into Unity systems.
- Reforged HD (v1000/1100 `SKIN` chunk) is not implemented. Classic v800 is the supported path.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Empty model / no Animation clips, `FbxSkeleton … not in the hierarchy` | Re-export with this converter (root must connect to FBX `RootNode 0`; `NodeAttribute` names must be empty). |
| `Split Animation Take Not Found 'Decay_Bone'` | Re-export. Static poses now write a hold `Lcl Translation` so Unity keeps the take. |
| Import runs over and over | Use the script from `unity/KnightModelPostprocessor.cs` (no `SaveAndReimport` loop). Run **Tools → Setup Knight Model** only once. |
| `CS0104: 'Object' is an ambiguous reference` | Use the shipped script (`UnityEngine.Object.DestroyImmediate`). |
| Rider beside the horse on Stand/Attack | Old bake leaked Death offsets. Re-export with the current converter. |
| Scrambled vertices | Keep **Weld Vertices** off. Compare the sibling `.obj`. |
| Prefab rotated 90° | Re-export with `--axis unity`. |
| Unity already open, batchmode fails | Do not launch a second Editor. Copy files and Refresh / use the menu. |
