using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public static class SemanticTools
{
    // rendered obj: old colours, new colours
    //private Dictionary<string, Dictionary<string, Color[]>> objColours;


    //public Dictionary<string, Dictionary<string, Color[]>> GetObjColoursDict()
    //{ return objColours; }

    // RGB int8 to float32 array
    private static float[] ToFloatArray(int r, int g, int b)
    { return (from num in new int[] { r, g, b } select (float)(num / 255.0f)).ToArray(); }

    // gets array of 3 float32 values into a Color object
    private static Color ToColour(float[] fArray)
    { return new Color(fArray[0], fArray[1], fArray[2], 1.0f); }

    public static Color[] GetNewColoursByObj(string obj)
    {
        Color[] colours = { };

        switch (obj)
        {
            case "fence" :
                colours = new Color[]{ ToColour(ToFloatArray(190, 153, 153)) };
                break;
            case "guard rail" :
                colours = new Color[]{ ToColour(ToFloatArray(180, 165, 180)) };
                break;
            case "pole" :
            case "polegroup" :
                colours = new Color[]{ ToColour(ToFloatArray(153, 153, 153)) };
                break;
            case "traffic light" :
                colours = new Color[]{ ToColour(ToFloatArray(250, 170, 30)) };
                break;
            case "traffic sign" :
                colours = new Color[]{ ToColour(ToFloatArray(220, 220, 0)) };
                break;
            case "vegetation" :
                colours = new Color[]{ ToColour(ToFloatArray(107, 142, 35)) };
                break;
            case "person" :
                colours = new Color[]{ ToColour(ToFloatArray(220, 20, 60)) };
                break;
            case "rider" :
                colours = new Color[]{ ToColour(ToFloatArray(255, 0, 0)) };
                break;
            case "car" : 
                colours = new Color[]{ ToColour(ToFloatArray(0, 0, 142)) };
                break;
            case "truck" :
                colours = new Color[]{ ToColour(ToFloatArray(0, 0, 70)) };
                break;
            case "bus" :
                colours = new Color[]{ ToColour(ToFloatArray(0, 60, 100)) };
                break;
            case "motorcycle" :
                colours = new Color[]{ ToColour(ToFloatArray(0, 0, 230)) };
                break;
            case "bicycle":
                colours = new Color[]{ ToColour(ToFloatArray(119, 11, 32)) };
                break;
            default:
                colours = new Color[]{ ToColour(ToFloatArray(0, 0, 0)) };
                break;
        }
        return colours;
    }

    // At the moment they're also the same colour since the rendered object always has 1 as Render Layer ID
    public static Color[] GetOldColoursByObj(string obj)
    {
        Color[] colours = { };

        switch (obj)
        {
            case "fence":
            case "guard rail": 
            case "pole": 
            case "polegroup":
            case "traffic light": 
            case "traffic sign": 
            case "vegetation": 
            case "person": 
            case "rider": 
            case "car": 
            case "truck": 
            case "bus": 
            case "motorcycle": 
            case "bicycle":
            default:
                colours = new Color[] { ToColour(ToFloatArray(186, 0, 0)) };
                break;
        }
        return colours;
    }
}
