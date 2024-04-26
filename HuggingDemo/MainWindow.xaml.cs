using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;


namespace HuggingDemo
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private const int imageSize = 224;
        public MainWindow()
        {
            InitializeComponent();
        }
        public void ModelInit()
        {
 
        }
        byte[] pixelData;
        private Color GetPixelColor(WriteableBitmap bitmap, int x, int y)
        {
            int stride = bitmap.PixelWidth * (bitmap.Format.BitsPerPixel / 8);
            int offset = y * stride + x * (bitmap.Format.BitsPerPixel / 8);

            byte[] pixel = new byte[4]; // RGBA
            Array.Copy(pixelData, offset, pixel, 0, 4);

            return Color.FromArgb(pixel[3], pixel[2], pixel[1], pixel[0]);
        }
        private void Button_Click(object sender, RoutedEventArgs e)
        {
            Uri uri01 = new Uri(textBox.Text);
            BitmapImage imageSource = new BitmapImage(uri01);
            WriteableBitmap writableBitmap = new WriteableBitmap(imageSource);
            pictureBox.Source = imageSource;
            // 图片加载成功后，从图片控件中取出224*224的位图对象

            int stride = writableBitmap.PixelWidth * (writableBitmap.Format.BitsPerPixel / 8);
            int pixelCount = writableBitmap.PixelWidth * writableBitmap.PixelHeight;
            pixelData = new byte[stride * writableBitmap.PixelHeight];
            writableBitmap.CopyPixels(pixelData, stride, 0);    

            float[] imageArray = new float[imageSize * imageSize * 3];

            // 按照先行后列的方式依次取出图片的每个像素值
            for (int y = 0; y < imageSize; y++)
            {
                for (int x = 0; x < imageSize; x++)
                {
                    var color = GetPixelColor(writableBitmap,x, y);

                    // 使用Netron查看模型的输入发现
                    // 需要依次放置224 *224的蓝色分量、224*224的绿色分量、224*224的红色分量
                    imageArray[y * imageSize + x] = color.B;
                    imageArray[y * imageSize + x + 1 * imageSize * imageSize] = color.G;
                    imageArray[y * imageSize + x + 2 * imageSize * imageSize] = color.R;
                }
            }

            // 设置要加载的模型的路径，跟据需要改为你的模型名称
            string modelPath = AppDomain.CurrentDomain.BaseDirectory + "BearModel.onnx";

            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();

                // 用Netron看到需要的输入类型是float32[None,3,224,224]
                // 第一维None表示可以传入多张图片进行推理
                // 这里只使用一张图片，所以使用的输入数据尺寸为[1, 3, 224, 224]
                var shape = new int[] { 1, 3, imageSize, imageSize };
                var tensor = new DenseTensor<float>(imageArray, shape);

                // 支持多个输入，对于mnist模型，只需要一个输入，输入的名称是data
                container.Add(NamedOnnxValue.CreateFromTensor<float>("data", tensor));

                // 推理
                var results = session.Run(container);

                // 输出结果有两个，classLabel和loss，这里只关心classLabel
                var label = results.FirstOrDefault(item => item.Name == "classLabel")? // 取出名为classLabel的输出
                    .AsTensor<string>()?
                    .FirstOrDefault(); // 支持多张图片同时推理，这里只推理了一张，取第一个结果值

                // 显示在控件中
                textBlock.Text = label;
            }
        }
    }
}
