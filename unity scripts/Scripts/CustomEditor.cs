using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;
using System.Collections;
using UnityEditor;

[CustomEditor(typeof(AutoRender))]
public class AutoRenderEditor : Editor
{
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();
        string path = "";
        if (GUILayout.Button("Browse for Output Path folder"))
        {
            path = EditorUtility.OpenFolderPanel("Select output folder", "%USERPROFILE%", "");
            AutoRender r = (AutoRender)target;
            r.outputPath = path;
        }

        EditorGUILayout.LabelField(path);
    }

}
