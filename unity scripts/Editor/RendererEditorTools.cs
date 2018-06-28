using UnityEngine;
using UnityEditor;
using UnityEngine.SceneManagement;
using System.IO;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;

public class RendererEditorTools : MonoBehaviour
{

    private static GameObject m_camera = GameObject.Find("Main Camera");
    private static GameObject m_obj = GameObject.FindWithTag("CompositionObject");
    private static GameObject m_light = GameObject.Find("Directional Light");
    private static GameObject m_cylinder = GameObject.Find("Cylinder");

    /**
     * This file contains many methods to be called from the Unity Editor
     * Which were written to be tried out, or to help make some tasks faster
     * 
     * It can be ignored
     */


    static GameObject GetCylinder(GameObject[] go_array)
    {

        GameObject o_out = new GameObject();
        foreach (GameObject go in go_array)
        {
            if (go.name == "Cylinder")
            {
                o_out = go;
            }
        }

        return o_out;
    }

    static GameObject GetCar(GameObject[] go_array)
    {
        GameObject o_out = new GameObject();
        foreach (GameObject go in go_array)
        {
            if (go.name.EndsWith("_car"))
            {
                o_out = go;
            }
        }

        return o_out;
    }

    static void SetCylinderActive(GameObject[] go_array, bool active)
    {
        foreach (GameObject go in go_array)
        {
            if (go.name == "Cylinder" && go.activeSelf != active)
            {
                go.SetActive(active);
            }
        }

    }

    static void SetCarActive(GameObject[] go_array, bool active)
    {
        foreach (GameObject go in go_array)
        {
            if (go.name.EndsWith("_car") && go.activeSelf != active)
            {
                go.SetActive(active);
            }
        }

    }

    // Order: {carBool, cylBool}
    static void SetCarCyl(GameObject[] go_array, bool[] active)
    {

        foreach (GameObject go in go_array)
        {
            // activeSelf for the object's boolean
            // activeInHierarchy to see if it's ultimately active (to check if all parents are also active)
            if (go.name.Equals("Cylinder") && go.activeSelf != active[0])
            {
                go.SetActive(true);
                Debug.Log("Activated " + go.name);
            }
            else if (go.name.EndsWith("_car") && go.activeSelf != active[1])
            {
                go.SetActive(true);
                Debug.Log("Activated " + go.name);
            }
        }
    }

    // Order: {carBool, cylBool}
    static void SetObjCyl(GameObject[] go_array, bool[] active)
    {

        foreach (GameObject go in go_array)
        {
            // activeSelf for the object's boolean
            // activeInHierarchy to see if it's ultimately active (to check if all parents are also active)
            if (go.name.Equals("Cylinder") && go.activeSelf != active[0])
            {
                go.SetActive(true);
                Debug.Log("Activated " + go.name);
            }
            else if (go.name.EndsWith("_obj") && go.activeSelf != active[1])
            {
                go.SetActive(true);
                Debug.Log("Activated " + go.name);
            }
        }
    }



    public class RenderPass
    {
        public Octane.RenderPassId _renderPass;
        public Octane.ImageSaveType _renderPassOutputType;
        public bool _capturePass;
        public string _outputDir;

        public RenderPass(Octane.RenderPassId renderPass, Octane.ImageSaveType renderPassOutputType, bool capturePass = true, string outputDir = "/../Output/Octane")
        {
            _renderPass = renderPass;
            _renderPassOutputType = renderPassOutputType;
            _capturePass = capturePass;
            _outputDir = Application.dataPath + outputDir;
        }
    }





    [MenuItem("Automation/Save PNG")]
    static void SaveImage()
    {
        Scene aS = SceneManager.GetActiveScene();
        GameObject[] go_array = aS.GetRootGameObjects();
        // Debug.Log("Active scene: " + aS.name + " | " + aS.name.StartsWith("VonR3_car"));
        // Component[] comps;


        foreach (GameObject go in go_array)
        {
            if (go.name.StartsWith("PBR"))
            {
                Debug.Log("Name: " + go.name + "\nType: " + go.GetType());

                // Component[] comps = go.GetComponents(go.GetType());
                Debug.Log("Rendering: " + Octane.Renderer.IsRendering.ToString());
                Octane.RenderTarget rt = Octane.Renderer.RenderTarget;

                List<RenderPass> my_render_passes = new List<RenderPass>();
                my_render_passes.Add(new RenderPass(Octane.RenderPassId.RENDER_PASS_BEAUTY,
                                                    Octane.ImageSaveType.IMAGE_SAVE_TYPE_PNG8,
                                                    true,
                                                    "/../Output/Octane/Beauty"));


            }
        }
    }


    static List<String> readTextFile(String filepath)
    {
        List<String> results = new List<string>();

        int counter = 0;
        string line;

        // Read the file and display it line by line.  
        System.IO.StreamReader file = new System.IO.StreamReader(filepath);
        while ((line = file.ReadLine()) != null)
        {
            //Debug.Log(line);
            results.Add(line);
            counter++;
        }

        file.Close();
        //Debug.Log(string.Format("There were {0} lines.", counter.ToString()));

        return results;
    }

    [MenuItem("Automation/Prepare Irpv")]
    static void irpv()
    {

        m_obj.SetActive(true);
        m_cylinder.SetActive(true);

    }

    [MenuItem("Automation/Prepare Ir")]
    static void ir()
    {

        m_obj.SetActive(false);
        m_cylinder.SetActive(true);

    }

    [MenuItem("Automation/Prepare Alpha")]
    static void alpha()
    {

        m_obj.SetActive(true);
        m_cylinder.SetActive(false);

    }


    [MenuItem("Automation/Render Scenes")]
    static void Render_scenes()
    {


        Scene aS = SceneManager.GetActiveScene();
        GameObject[] go_array = aS.GetRootGameObjects();

        List<String> pathList = readTextFile(Application.dataPath + "/SceneData/imageNames.txt");

        string sd_path = Application.dataPath + "/SceneData/";
        string target = sd_path + pathList[4] + ".json";

        if (File.Exists(target))
        {
            // Read the json from the file into a string
            string dataAsJson = File.ReadAllText(target);
            // Pass the json to JsonUtility, and tell it to create a GameData object from it
            SceneData loadedData = JsonUtility.FromJson<SceneData>(dataAsJson);

            m_camera = loadedData.GetCamera(m_camera);
            m_obj = loadedData.GetObject(m_obj);
            m_light = loadedData.GetLight(m_light);
            m_cylinder = loadedData.GetCylinder(m_cylinder);


            Debug.Log("First Image");
            ir();
            Octane.Renderer.SaveImage(Octane.RenderPassId.RENDER_PASS_BEAUTY,
                sd_path + "irpv.exr",
                Octane.ImageSaveType.IMAGE_SAVE_TYPE_EXR16,
                true);

            Debug.Log("Second Image");
            irpv();
            Octane.Renderer.SaveImage(Octane.RenderPassId.RENDER_PASS_BEAUTY,
                sd_path + "ir.exr",
                Octane.ImageSaveType.IMAGE_SAVE_TYPE_EXR16,
                true);


        }
        else
        {
            Debug.Log(target);
            Debug.LogError("Cannot load game data!");
        }

    }




    [MenuItem("Automation/Load Scene")]
    static void Load_scene()
    {
        Scene aS = SceneManager.GetActiveScene();
        GameObject[] go_array = aS.GetRootGameObjects();

        List<String> pathList = readTextFile(Application.dataPath + "/SceneData/imageNames.txt");


        string sd_path = Application.dataPath + "/SceneData/";
        string target = sd_path + pathList[4] + ".json";

        if (File.Exists(target))
        {
            // Read the json from the file into a string
            string dataAsJson = File.ReadAllText(target);
            // Pass the json to JsonUtility, and tell it to create a GameData object from it
            SceneData loadedData = JsonUtility.FromJson<SceneData>(dataAsJson);



            GameObject camera = GameObject.Find("PBR Render Target");
            GameObject obj = GameObject.FindWithTag("CompositionObject");
            GameObject light = GameObject.Find("Directional Light");
            GameObject cylinder = GameObject.Find("Cylinder");

            camera = loadedData.GetCamera(camera);
            obj = loadedData.GetObject(obj);
            light = loadedData.GetLight(light);
            cylinder = loadedData.GetCylinder(cylinder);


        }
        else
        {
            Debug.Log(target);
            Debug.LogError("Cannot load game data!");
        }

    }

    [MenuItem("Automation/Save Scene")]
    static void Save_scene()
    {

        GameObject camera = GameObject.Find("Main Camera");
        GameObject obj = GameObject.FindWithTag("CompositionObject");
        GameObject light = GameObject.Find("Directional Light");
        GameObject cylinder = GameObject.Find("Cylinder");

        SceneDataManager SDM = new SceneDataManager();
        String json = SDM.GetJSON(camera, obj, light, cylinder);
        //Debug.Log(json);

        // if false, overwrite
        bool append = false;
        string filePath = "sceneData.json";

        string fullPath = Path.GetFullPath(filePath);
        TextWriter writer = null;
        Debug.Log(String.Format("Saving in {0}", fullPath));

        try
        {
            writer = new StreamWriter(fullPath, append);
            writer.Write(json);
        }
        finally
        {
            if (writer != null)
                writer.Close();
        }

    }

    
    public static Texture2D LoadPNG(string filePath)
    {

        Texture2D tex = null;
        byte[] fileData;

        if (File.Exists(filePath))
        {
            fileData = File.ReadAllBytes(filePath);
            tex = new Texture2D(2, 2);
            tex.LoadImage(fileData); //..this will auto-resize the texture dimensions.
        }
        return tex;
    }

    // Texture2D to Texture2D correction
    public static Texture2D CorrectColoursOnTexture(Texture2D tex, Color[] oldColours, Color[] newColours, bool verbose = false)
    {
        int counter = 0;
        int minColourIndex = System.Math.Min(oldColours.Length, newColours.Length);

        for (int i = 0; i < tex.height; i++)
        {
            for (int j = 0; j < tex.width; j++)
            {
                Color px = tex.GetPixel(j, i);
                for (int k = 0; k < minColourIndex; k++)
                {
                    if (px[0] == oldColours[k].r &&
                        px[1] == oldColours[k].g &&
                        px[2] == oldColours[k].b)
                    {
                        tex.SetPixel(j, i, newColours[k]);
                        counter++;
                    }
                }

            }
        }
        if (verbose) Debug.Log(String.Format("{0} pixels changed", counter));
        tex.Apply();

        return tex;
    }

    public int To8bit(float f) { return (int)(255 * f); }

    /**
     * This function assumes that the old colours are passed as 8bit and new colours are passed in float32 with range (0-1)
     * In our case: the buffer has HDR with float32 but we use 8bit png values to compare because it helps avoid
     * floating point errors
     */
    public static Texture2D CorrectColoursOnPixelBuffer(Octane.Renderer.PixelBuffer pb, int channels, Color[] oldColours, Color[] newColours, bool verbose = false)
    {
        if (pb == null) throw new System.ArgumentNullException("Pixel Buffer is null. Render pass might not have been found", "pb");


        IntPtr pixels = pb.Pixels;
        Texture2D tex = new Texture2D((int)pb.ResX, (int)pb.ResY);
        int counter = 0;
        int minColourIndex = System.Math.Min(oldColours.Length, newColours.Length);

        if (verbose) Debug.Log(String.Format("Pixel Buffer Info -> Sizes: {0}, {1}. Format: {2}", pb.ResX, pb.ResY, pb.OctaneImageFormat));
        unsafe
        {

            for (int i = 0; i < pb.ResY; i++)
            {
                float* pI = (float*)pixels.ToPointer() + i * pb.ResX * channels; //pointer to start of row
                for (int j = 0; j < pb.ResX; j++)
                {
                    for (int k = 0; k < minColourIndex; k++)
                    {
                        // We don't care about the old alpha
                        if ((int)(255 * pI[j * channels + 0]) == (int)(255 * oldColours[k].r) &&
                            (int)(255 * pI[j * channels + 1]) == (int)(255 * oldColours[k].g) &&
                            (int)(255 * pI[j * channels + 2]) == (int)(255 * oldColours[k].b))
                        {
                            counter++;
                            tex.SetPixel(j, i, newColours[k]);
                        }
                        else
                        {
                            tex.SetPixel(j, i, new Color(pI[j * channels + 0], pI[j * channels + 1], pI[j * channels + 2], pI[j * channels + 3]));
                        }
                    }
                }
            }
        }
        if (verbose) Debug.Log(String.Format("{0} pixels changed", counter));
        return tex;
    }


    [MenuItem("Automation/Correct Colours and Save")]
    static void Correct_colours()
    {

        Octane.Renderer.PixelBuffer pb = Octane.Renderer.GetPixels(Octane.RenderPassId.RENDER_PASS_RENDER_LAYER_ID);
        if (pb == null) throw new System.ArgumentNullException("No Render Layer ID render pass found", "pb");

        string sd_path = Application.dataPath;
        string savepath = "E:\\Dan\\Projects";
        

        // This method reads the given pixel buffer to copy it into a Texture2D with the correct format
        // (in our case PNG8 for the old colours and float32 for the new ones) and returns the texture


        Color[] oldC = { new Color((float)186 / 255.0f, 0.0f, 0.0f, 1.0f) };
        Color[] newC = { new Color((float)142 / 255.0f, 0.0f, 0.0f, 1.0f) };
        Texture2D tex = CorrectColoursOnPixelBuffer(pb, 4, oldC, newC);
        File.WriteAllBytes(savepath + "\\corrected.png", tex.EncodeToPNG());
        
    }


}