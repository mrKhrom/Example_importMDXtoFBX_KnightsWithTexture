using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;

/// <summary>
/// Knight FBX import for Unity (copy this file into Assets/Scripts/Editor/).
///
/// Expected layout after you copy the converted files:
///   Assets/Models/Humans/Knight/Knight.fbx      (from KnightV2.fbx)
///   Assets/Models/Humans/Knight/KnightA.fbx     (from KnightAV2.fbx)
///   Assets/Models/Humans/Knight/KnightB.fbx     (from KnightBV2.fbx)
///   Assets/Models/Humans/Knight/KnightC.fbx     (from KnightCV2.fbx)
///   Assets/Models/Humans/Knight/Materials/Knight.mat
///   Assets/Models/Humans/Knight/Textures/KnightTexture.png
///   Assets/Prefabs/Knight.prefab                (created by Tools/Setup Knight Model)
///
/// Menu: Tools/Setup Knight Model
/// Sets Generic rig, remaps Mat_* to Knight.mat, Loop Time on idle/walk/portrait,
/// builds Knight.controller and Prefabs/Knight.prefab.
/// Clip settings are written only when they change — no reimport loop.
/// </summary>
public class KnightModelPostprocessor : AssetPostprocessor
{
    const string Folder = "Assets/Models/Humans/Knight/";
    const string MainFbx = Folder + "Knight.fbx";
    const string MatPath = Folder + "Materials/Knight.mat";
    const string ControllerPath = Folder + "Knight.controller";
    const string PrefabPath = "Assets/Prefabs/Knight.prefab";

    static readonly string[] KnightFiles =
    {
        Folder + "Knight.fbx",
        Folder + "KnightA.fbx",
        Folder + "KnightB.fbx",
        Folder + "KnightC.fbx",
    };

    static readonly string[] LoopClips =
    {
        "Walk", "Stand_1", "Stand_2", "Stand_3", "Stand_4", "Stand_Ready",
        "Stand_Victory", "Portrait_1", "Portrait_2", "Portrait_3", "Portrait_4",
        "Portrait_Talk_1", "Portrait_Talk_2", "GlobalSeq_0"
    };

    static bool s_Busy;

    static bool IsKnightFbx(string path)
    {
        if (string.IsNullOrEmpty(path) || !path.StartsWith(Folder)) return false;
        var name = Path.GetFileName(path);
        return name == "Knight.fbx" || name == "KnightA.fbx"
            || name == "KnightB.fbx" || name == "KnightC.fbx";
    }

    void OnPreprocessModel()
    {
        if (!IsKnightFbx(assetPath)) return;

        var imp = (ModelImporter)assetImporter;
        imp.animationType = ModelImporterAnimationType.Generic;
        imp.avatarSetup = ModelImporterAvatarSetup.CreateFromThisModel;
        imp.importAnimation = true;
        imp.animationCompression = ModelImporterAnimationCompression.Off;
        imp.resampleCurves = false;
        imp.weldVertices = false;
        imp.optimizeBones = false;
        imp.importVisibility = false;
        imp.preserveHierarchy = true;
        imp.importCameras = false;
        imp.importLights = false;
        imp.materialImportMode = ModelImporterMaterialImportMode.ImportViaMaterialDescription;
        imp.materialLocation = ModelImporterMaterialLocation.InPrefab;
        imp.materialSearch = ModelImporterMaterialSearch.Local;

        var mat = AssetDatabase.LoadAssetAtPath<Material>(MatPath);
        if (mat == null) return;

        foreach (var k in imp.GetExternalObjectMap().Keys.Where(k => k.type == typeof(Material)).ToList())
            imp.RemoveRemap(k);

        var names = new List<string>
        {
            "Knight", "Material", "No Name", "Fbx Default Material", "DefaultMaterial",
            "lambert", "lambert1", "Scene", "KnightTexture.png"
        };
        for (int i = 0; i < 12; i++)
            names.Add("Mat_" + i);

        foreach (var n in names)
            imp.AddRemap(new AssetImporter.SourceAssetIdentifier(typeof(Material), n), mat);
    }

    void OnPostprocessModel(GameObject g)
    {
        if (!IsKnightFbx(assetPath)) return;

        var mat = AssetDatabase.LoadAssetAtPath<Material>(MatPath);
        foreach (var smr in g.GetComponentsInChildren<SkinnedMeshRenderer>(true))
        {
            smr.updateWhenOffscreen = true;
            smr.quality = SkinQuality.Bone4;
            if (mat == null) continue;
            var mats = smr.sharedMaterials;
            if (mats == null || mats.Length == 0)
            {
                smr.sharedMaterial = mat;
                continue;
            }
            for (int i = 0; i < mats.Length; i++)
                if (mats[i] == null) mats[i] = mat;
            smr.sharedMaterials = mats;
        }

        var animator = g.GetComponent<Animator>();
        if (animator != null)
        {
            animator.applyRootMotion = false;
            animator.cullingMode = AnimatorCullingMode.AlwaysAnimate;
        }
    }

    static void OnPostprocessAllAssets(string[] imported, string[] deleted, string[] movedTo, string[] movedFrom)
    {
        if (s_Busy || imported == null) return;
        var knights = imported.Where(IsKnightFbx).ToList();
        if (knights.Count == 0) return;

        EditorApplication.delayCall += () =>
        {
            if (s_Busy) return;
            s_Busy = true;
            try
            {
                foreach (var path in knights)
                    ApplyClipSettings(path, reimport: false);
            }
            finally
            {
                s_Busy = false;
            }
        };
    }

    static bool ShouldLoop(string take)
    {
        if (string.IsNullOrEmpty(take)) return false;
        if (LoopClips.Contains(take)) return true;
        var low = take.ToLowerInvariant();
        if (low.StartsWith("stand") || low.StartsWith("walk") || low.StartsWith("portrait"))
            return true;
        if (low.StartsWith("attack") || low.StartsWith("death") || low.StartsWith("decay") || low.StartsWith("spell"))
            return false;
        return false;
    }

    /// <summary>
    /// Rebuild clipAnimations from takes Unity actually found.
    /// Drops stale take names and never calls SaveAndReimport unless
    /// the caller asked and the table actually changed.
    /// </summary>
    static bool ApplyClipSettings(string fbxPath, bool reimport)
    {
        var imp = AssetImporter.GetAtPath(fbxPath) as ModelImporter;
        if (imp == null) return false;

        var defaults = imp.defaultClipAnimations;
        if (defaults == null || defaults.Length == 0)
        {
            if (imp.clipAnimations != null && imp.clipAnimations.Length > 0)
            {
                imp.clipAnimations = new ModelImporterClipAnimation[0];
                EditorUtility.SetDirty(imp);
            }
            return false;
        }

        var clips = new ModelImporterClipAnimation[defaults.Length];
        for (int i = 0; i < defaults.Length; i++)
        {
            var c = defaults[i];
            bool loop = ShouldLoop(c.takeName);
            c.loopTime = loop;
            c.loop = loop;
            c.name = string.IsNullOrEmpty(c.takeName) ? c.name : c.takeName;
            c.keepOriginalOrientation = true;
            c.keepOriginalPositionY = true;
            c.keepOriginalPositionXZ = true;
            clips[i] = c;
        }

        if (ClipsMatch(imp.clipAnimations, clips))
            return false;

        imp.clipAnimations = clips;
        if (reimport)
            imp.SaveAndReimport();
        else
            EditorUtility.SetDirty(imp);
        return true;
    }

    static bool ClipsMatch(ModelImporterClipAnimation[] current, ModelImporterClipAnimation[] desired)
    {
        if (current == null || current.Length != desired.Length) return false;
        for (int i = 0; i < desired.Length; i++)
        {
            var a = current[i];
            var b = desired[i];
            if (a.takeName != b.takeName) return false;
            if (a.name != b.name) return false;
            if (a.loopTime != b.loopTime) return false;
            if (a.loop != b.loop) return false;
        }
        return true;
    }

    static void BuildControllerAndPrefab()
    {
        var model = AssetDatabase.LoadAssetAtPath<GameObject>(MainFbx);
        if (model == null)
        {
            Debug.LogError("[Knight] FBX not loaded: " + MainFbx);
            return;
        }

        var clips = LoadClips(MainFbx);

        AnimatorController ctrl = null;
        if (clips.Count > 0)
        {
            if (File.Exists(ControllerPath))
                AssetDatabase.DeleteAsset(ControllerPath);

            ctrl = AnimatorController.CreateAnimatorControllerAtPath(ControllerPath);
            var sm = ctrl.layers[0].stateMachine;
            foreach (var st in sm.states.ToList())
                sm.RemoveState(st.state);

            if (!ctrl.parameters.Any(p => p.name == "Attack"))
                ctrl.AddParameter("Attack", AnimatorControllerParameterType.Trigger);
            if (!ctrl.parameters.Any(p => p.name == "Speed"))
                ctrl.AddParameter("Speed", AnimatorControllerParameterType.Float);

            var byName = new Dictionary<string, AnimatorState>();
            float y = 0f;
            foreach (var clip in clips)
            {
                var state = sm.AddState(clip.name, new Vector3(280f, y, 0f));
                state.motion = clip;
                byName[clip.name] = state;
                y += 50f;
            }

            AnimatorState idle = null;
            foreach (var name in new[] { "Stand_1", "Stand_Ready", "Stand_2", "Walk" })
            {
                if (byName.TryGetValue(name, out idle))
                    break;
            }
            if (idle == null)
                idle = byName.Values.First();
            sm.defaultState = idle;

            if (byName.TryGetValue("Walk", out var walk) && idle != walk)
            {
                var toWalk = idle.AddTransition(walk);
                toWalk.hasExitTime = false;
                toWalk.duration = 0.15f;
                toWalk.AddCondition(AnimatorConditionMode.Greater, 0.1f, "Speed");

                var toIdle = walk.AddTransition(idle);
                toIdle.hasExitTime = false;
                toIdle.duration = 0.15f;
                toIdle.AddCondition(AnimatorConditionMode.Less, 0.1f, "Speed");
            }

            AnimatorState attack = null;
            foreach (var name in new[] { "Attack_1", "Attack_2", "Spell" })
            {
                if (byName.TryGetValue(name, out attack))
                    break;
            }
            if (attack != null && idle != null && attack != idle)
            {
                var toAtk = idle.AddTransition(attack);
                toAtk.hasExitTime = false;
                toAtk.duration = 0.08f;
                toAtk.AddCondition(AnimatorConditionMode.If, 0f, "Attack");

                var back = attack.AddTransition(idle);
                back.hasExitTime = true;
                back.exitTime = 0.9f;
                back.duration = 0.12f;
            }

            EditorUtility.SetDirty(ctrl);
        }

        var mat = AssetDatabase.LoadAssetAtPath<Material>(MatPath);
        var instance = (GameObject)PrefabUtility.InstantiatePrefab(model);
        instance.name = "Model";

        var root = new GameObject("Knight");
        instance.transform.SetParent(root.transform, false);
        instance.transform.localPosition = Vector3.zero;
        instance.transform.localRotation = Quaternion.identity;
        instance.transform.localScale = Vector3.one;

        foreach (var smr in root.GetComponentsInChildren<SkinnedMeshRenderer>(true))
        {
            smr.updateWhenOffscreen = true;
            if (mat == null) continue;
            var mats = smr.sharedMaterials;
            for (int i = 0; i < mats.Length; i++)
                if (mats[i] == null) mats[i] = mat;
            if (mats.Length == 0) smr.sharedMaterial = mat;
            else smr.sharedMaterials = mats;
        }

        var animator = instance.GetComponent<Animator>() ?? instance.AddComponent<Animator>();
        animator.applyRootMotion = false;
        animator.cullingMode = AnimatorCullingMode.AlwaysAnimate;
        if (ctrl != null)
            animator.runtimeAnimatorController = ctrl;

        if (!AssetDatabase.IsValidFolder("Assets/Prefabs"))
            AssetDatabase.CreateFolder("Assets", "Prefabs");

        PrefabUtility.SaveAsPrefabAsset(root, PrefabPath);
        UnityEngine.Object.DestroyImmediate(root);

        AssetDatabase.SaveAssets();
        Debug.Log($"[Knight] Ready: {PrefabPath} | clips={clips.Count} | mat={MatPath}");
    }

    static List<AnimationClip> LoadClips(string fbxPath)
    {
        return AssetDatabase.LoadAllAssetsAtPath(fbxPath)
            .OfType<AnimationClip>()
            .Where(c => c != null && !c.name.StartsWith("__preview__"))
            .GroupBy(c => c.name)
            .Select(g => g.First())
            .OrderBy(c => c.name)
            .ToList();
    }

    [MenuItem("Tools/Setup Knight Model")]
    public static void Run()
    {
        if (s_Busy) return;
        s_Busy = true;
        try
        {
            foreach (var path in KnightFiles)
            {
                if (File.Exists(path))
                    AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceUpdate);
            }

            foreach (var path in KnightFiles)
            {
                if (File.Exists(path))
                    ApplyClipSettings(path, reimport: false);
            }

            AssetDatabase.SaveAssets();
            BuildControllerAndPrefab();
            Debug.Log("[Knight] Setup finished. Select Knight.fbx → Animation tab. Use Prefabs/Knight.prefab.");
        }
        catch (System.Exception ex)
        {
            Debug.LogError("[Knight] Setup failed: " + ex);
            if (Application.isBatchMode) EditorApplication.Exit(1);
            return;
        }
        finally
        {
            s_Busy = false;
        }
        if (Application.isBatchMode)
            EditorApplication.Exit(0);
    }
}
