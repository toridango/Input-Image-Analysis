using System.Collections;
using System.Collections.Generic;
using System.IO;
using System;
using System.Linq;
using UnityEngine;
using UnityEditor;
using UnityEngine.SceneManagement;

struct Param
{
    public string obj;
    public string name;
}


public class AutoRender : MonoBehaviour
{

    public GameObject m_camera;
    public GameObject m_obj;
    public GameObject m_light;
    public GameObject m_surface;
    public bool useAlphaKernel = false;
    public string outputPath;

    private int m_standby;
    private string m_currStage;
    private List<String> pathList;
    private List<Param> paramList;
    private string sd_path;
    private string target;
    private int imgCount;
    private int i_img;
    private uint m_MaxSamplesPerPixel = 0;
    private float start_time;
    private bool restarted;
    private int m_waittime;
    private bool reset;

    /*
     * Check if necessary objects are set in the editor inspector
     * Try to find them if they aren't
     */
    private void CheckObjectsSet()
    {
        if (m_camera == null) { m_camera = GameObject.FindWithTag("Camera"); }
        if (m_obj == null) { m_obj = GameObject.FindWithTag("CompositionObject"); }
        if (m_light == null) { m_light = GameObject.Find("Directional Light"); }
        if (m_surface == null) { m_surface = GameObject.FindWithTag("Surface"); }
    }

    /*
     * Check if all the objects are ready
     * Throw an exception if they aren't
     */
    private void CheckObjectsNotNull()
    {
        if (m_camera == null) { throw new System.ArgumentException("Parameter cannot be null", "Main Camera"); }
        if (m_obj == null) { throw new System.ArgumentException("Parameter cannot be null", "Composition Object"); }
        if (m_light == null) { throw new System.ArgumentException("Parameter cannot be null", "Directional Light"); }
        if (m_surface == null) { throw new System.ArgumentException("Parameter cannot be null", "Surface"); }
    }

    private void OnGUI()
    {

    }

    private void setIRPV()
    {
        if (!m_obj.activeSelf) m_obj.SetActive(true);
        if (!m_surface.activeSelf) m_surface.SetActive(true);
    }

    private void setIR()
    {
        if (m_obj.activeSelf) m_obj.SetActive(false);
        if (!m_surface.activeSelf) m_surface.SetActive(true);
    }

    private void setAlpha()
    {
        if (!m_obj.activeSelf) m_obj.SetActive(true);
        if (m_surface.activeSelf) m_surface.SetActive(false);
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

    /*
     * Sets the current scene, varies with i_img
     */
    private void setScene()
    {

        target = sd_path + paramList[i_img].name + ".json";

        if (File.Exists(target))
        {
            // Read the json from the file into a string
            string dataAsJson = File.ReadAllText(target);
            // Pass the json to JsonUtility, and tell it to create a GameData object from it
            SceneData loadedData = JsonUtility.FromJson<SceneData>(dataAsJson);

            CheckObjectsSet();
            CheckObjectsNotNull();

            m_camera = loadedData.GetCamera(m_camera);
            m_obj = loadedData.GetObject(m_obj);
            m_light = loadedData.GetLight(m_light);
            m_surface = loadedData.GetCylinder(m_surface);

        }
        else
        {
            Debug.Log(target);
            Debug.LogError("Cannot load game data!");
        }

    }

    private bool FinishedRendering()
    {
        return
        (
            /* Octane should not be currently compiling the current scene frame */
            !Octane.Renderer.IsCompiling &&
            /* ensure desired spp has been achieved */
            Octane.Renderer.SampleCount.Equals(m_MaxSamplesPerPixel)

        );
    }

    private bool IsActive(GameObject go) { return go.activeSelf && go.activeInHierarchy; }

    private bool AllClear()
    {
        bool check = false;

        switch (m_currStage)
        {
            case "irpv":
                check = IsActive(m_obj) && IsActive(m_surface);// && !useAlphaKernel;
                break;
            case "ir":
                check = !IsActive(m_obj) && IsActive(m_surface);// && !useAlphaKernel;
                break;
            case "alpha":
                check = IsActive(m_obj) && !IsActive(m_surface);// && useAlphaKernel;
                break;
            default:
                check = false;
                break;
        }

        if (!check) Debug.Log(m_currStage + " doesn't check out");

        check = check && Directory.Exists(outputPath);

        return check;
    }


    private void ResetStage()
    {

        switch (m_currStage)
        {
            case "irpv":
                setIRPV();
                break;
            case "ir":
                setIR();
                break;
            case "alpha":
                setAlpha();
                break;
            default:
                throw new System.ArgumentException("Stage not recognised", "m_currStage");
        }

    }


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

    private static List<Param> ExtractFromtextFile(string filename, string separator = " - ")
    {
        List<String> data = readTextFile(Application.dataPath + "/SceneData/imageNames.txt");

        // make a list in which, for every line in the file, extract both parameters and save them in a struct
        return (from s 
                in data
                select new Param { obj = s.Substring(0, s.IndexOf(separator)), name = s.Substring(s.IndexOf(separator) + separator.Length) }
                ).ToList();
        
    }


    // Use this for initialization
    void Start()
    {
        start_time = Time.time;

        //Octane.Renderer.Restart();

        CheckObjectsSet();
        CheckObjectsNotNull();

        paramList = ExtractFromtextFile(Application.dataPath + "/SceneData/imageNames.txt");
        //pathList = readTextFile(Application.dataPath + "/SceneData/imageNames.txt");
        m_MaxSamplesPerPixel = Octane.Renderer.GetLatestRenderStatistics(Octane.RenderPassId.RENDER_PASS_BEAUTY).maxSamplesPerPixel;
        //Debug.Log(m_MaxSamplesPerPixel);
        imgCount = paramList.Count;
        i_img = 0;

        //Debug.Log(String.Format("Number of input images: {0}", imgCount));

        sd_path = Application.dataPath + "/SceneData/";


        if (useAlphaKernel)
        {
            m_currStage = "alpha";
            setAlpha();
        }
        else
        {
            m_currStage = "irpv";
            setIRPV();
        }

        // 0: nothing prepared // 1: Scene set // 2: time waited, standby is false
        m_standby = 0;
        restarted = false;
        m_waittime = 20;
        reset = false;
    }

    IEnumerator OpenTheGates(int time)
    {
        yield return new WaitForSecondsRealtime(time / 2);
        Debug.Log("Restarting");
        setScene();
        Octane.Renderer.Restart();
        yield return new WaitForSecondsRealtime(time / 2);
        m_standby++;
    }

    // Update is called once per frame
    void Update()
    {

        float t_since_start = (float)(Math.Round((double)Time.time - start_time, 3));
        // LAST RESORT IDEA TO KEEP OCTANE FROM SAVING THE IMAGES BEFORE THE RENDER HAS BEGUN
        // Force a wait time until 5/7/10/... seconds after start has been called


        if (restarted)
        {
            if (i_img < imgCount)
            {
                target = sd_path + paramList[i_img].name + ".json";
                Debug.Log(m_standby);
                //Octane.Renderer.Stop();

                if (m_standby < 3)
                {
                    //Debug.Log("Opening Gates");
                    if (m_standby < 1)
                    {
                        Debug.Log("Setting first scene");
                        setScene();
                        m_standby++;
                    }
                    else if (m_standby == 1)
                    {
                        //if(!useAlphaKernel)
                        m_standby++;
                        StartCoroutine(OpenTheGates(m_waittime));
                        //else StartCoroutine(OpenTheGates(10));
                    }

                }
                else
                {

                    if (FinishedRendering())
                    {
                        if (AllClear())
                        {
                            // string fullpath = sd_path + "\\Output\\" + pathList[i_img] + "_" + m_currStage + ".exr";
                            string fullPath = outputPath + "/" + paramList[i_img].name + "_" + m_currStage + ".exr";
                            Debug.Log(String.Format("{0}/{1} - Name: {2} - Saving: {3}\nFull Path: {4}", i_img + 1, imgCount, paramList[i_img].name, m_currStage, fullPath));

                            if (m_currStage == "alpha")
                            {
                                string semanticPath = outputPath + "/" + paramList[i_img].name + "_semantic.png";
                                Debug.Log(String.Format("{0}/{1} - Name: {2} - Saving: {3}\nFull Path: {4}", i_img + 1, imgCount, paramList[i_img].name, "semantic", semanticPath));
                                /*Octane.Renderer.SaveImage(Octane.RenderPassId.RENDER_PASS_RENDER_LAYER_ID,
                                    semanticPath,
                                    Octane.ImageSaveType.IMAGE_SAVE_TYPE_PNG8,
                                    true);*/
                                Octane.Renderer.PixelBuffer pb = Octane.Renderer.GetPixels(Octane.RenderPassId.RENDER_PASS_RENDER_LAYER_ID);
                                if (pb == null) throw new System.ArgumentNullException("No Render Layer ID render pass found", "pb");
                                //Color[] oldC = { new Color((float)186 / 255.0f, 0.0f, 0.0f, 1.0f) };
                                //Color[] newC = { new Color((float)142 / 255.0f, 0.0f, 0.0f, 1.0f) };
                                Color[] oldC = SemanticTools.GetOldColoursByObj(paramList[i_img].obj);
                                Color[] newC = SemanticTools.GetNewColoursByObj(paramList[i_img].obj);
                                Texture2D tex = CorrectColoursOnPixelBuffer(pb, 4, oldC, newC);
                                File.WriteAllBytes(semanticPath, tex.EncodeToPNG());
                            }

                            Octane.Renderer.SaveImage(Octane.RenderPassId.RENDER_PASS_BEAUTY,
                                fullPath,
                                Octane.ImageSaveType.IMAGE_SAVE_TYPE_EXR16,
                                true);


                            switch (m_currStage)
                            {
                                case "irpv":
                                    m_currStage = "ir";
                                    setIR();
                                    break;
                                case "ir":
                                    m_currStage = "irpv";
                                    //setAlpha();
                                    setIRPV();
                                    i_img++;
                                    if (i_img < imgCount) setScene(); // Crashed here due to this being right after i++
                                    break;
                                case "alpha":
                                    //m_currStage = "irpv";
                                    //setIRPV();
                                    setAlpha();
                                    i_img++;
                                    if (i_img < imgCount) setScene();
                                    break;
                                default:
                                    throw new System.ArgumentException("Stage not recognised", "m_currStage");
                            }

                            if (i_img < imgCount)
                            {
                                restarted = false;
                                m_waittime = 5;
                                reset = false;
                            }
                            //Octane.Renderer.Restart();

                        }
                        else
                        {
                            ResetStage();
                        }
                    }
                }
            }
            else
            {
                float rounded_timedelta = (float)(Math.Round((double)Time.time - start_time, 3));

                Debug.Log(String.Format("Finished rendering!\nTime elapsed: {0}s", rounded_timedelta));
                Application.Quit(); // Doesn't quit in Editor mode, but I left it just in case
                UnityEditor.EditorApplication.isPlaying = false;
            }
        }
        else
        {
            restarted = Octane.Renderer.SampleCount < m_MaxSamplesPerPixel; // 128;
            if (!restarted && !reset)
            {
                setScene();
                Octane.Renderer.Restart();
                m_standby = 1;
                reset = true;
            }
            Debug.Log(String.Format("Restarted: {0}", restarted));
            Debug.Log(String.Format("Reset: {0}", reset));
        }

    }
}
