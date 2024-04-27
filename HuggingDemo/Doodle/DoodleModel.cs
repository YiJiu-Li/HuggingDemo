using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using static HuggingDemo.Doodle.DoodleEnum;

namespace HuggingDemo.Doodle
{
    public class DoodleModel
    {
        public DoodleModel()
        {
            points = new List<DoodlePoint>();
        }


        public IList<DoodlePoint> points;
        public DoodleEnumColor lineColor;
        public DoodleEnumType drawType;
        public double lineWidth;


        public DoodlePoint startPoint
        {
            get
            {
                if (points.Count > 0)
                {
                    return points[0];
                }
                else
                {
                    return null;
                }
            }
        }
    }
}
