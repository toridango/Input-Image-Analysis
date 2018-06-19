using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;
using UnityEditor;
using UnityEngine.SceneManagement;
using LitJson;

[Serializable]
public class BBox
{
    public Vector3 bbmin; // x y z
    public Vector3 bbmax;

    public BBox(Vector3 mins, Vector3 maxs)
    {
        bbmin = mins;
        bbmax = maxs;
    }

}

[Serializable]
public class GameObjData
{
    public Vector3 position;
    public Vector3 rotation;
    public Vector3 scale;

    public GameObjData(Transform t)
    {
        position = t.position;
        rotation = t.rotation.eulerAngles;
        scale = t.localScale;
        //Quaternion r = Quaternion.Euler(rotation);
        //Debug.Log(t.rotation + "\n" + r);
    }

    /*public Transform GetTransform()
    {
        Transform t = new Transform()
        t.position = position;
        t.rotation = Quaternion.Euler(rotation);
        t.localScale = scale;


        return t;
    }*/
}

[Serializable]
public class SceneData
{

    public GameObjData cameraData;
    public GameObjData objData;
    public GameObjData lightData;
    public GameObjData cylinderData;

    private Dictionary<string, int> infoRegister;

    /*public SceneData()
    {
        infoRegister.Add("cam", 0);
        infoRegister.Add("obj", 0);
        infoRegister.Add("lit", 0);
        infoRegister.Add("cyl", 0);
    }*/

    public SceneData(GameObject camera, GameObject obj, GameObject light, GameObject cylinder)
    {
        infoRegister = new Dictionary<string, int>();

        cameraData = new GameObjData(camera.transform);
        objData = new GameObjData(obj.transform);
        lightData = new GameObjData(light.transform);
        cylinderData = new GameObjData(cylinder.transform);

        infoRegister.Add("cam", 1);
        infoRegister.Add("obj", 1);
        infoRegister.Add("lit", 1);
        infoRegister.Add("cyl", 1);
    }

    public bool IsFull()
    {
        bool full = true;

        foreach (KeyValuePair<string, int> entry in infoRegister)
        {
            if (entry.Value != 1) full = false;
        }

        return full;
    }

    private GameObject GetGameObjectFromGameObjectData(GameObjData god, GameObject go)
    {
        //GameObject go = new GameObject();
        go.transform.position = god.position;
        go.transform.rotation = Quaternion.Euler(god.rotation);
        go.transform.localScale = god.scale;
        return go;
    }

    public GameObject GetCamera(GameObject go) { return GetGameObjectFromGameObjectData(cameraData, go); }
    public GameObject GetObject(GameObject go) { return GetGameObjectFromGameObjectData(objData, go); }
    public GameObject GetLight(GameObject go) { return GetGameObjectFromGameObjectData(lightData, go); }
    public GameObject GetCylinder(GameObject go) { return GetGameObjectFromGameObjectData(cylinderData, go); }
}

public class SceneDataManager
{
    public string GetJSON(GameObject camera, GameObject obj, GameObject light, GameObject cylinder)
    {
        SceneData SD = new SceneData(camera, obj, light, cylinder);
        string json = JsonUtility.ToJson(SD, true);
        //Debug.Log("Scene Data " + JsonUtility.ToJson(SD, true));
        return json;
    }
}