using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections;
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
using System.Windows.Media.Media3D;
using System.Windows.Navigation;
using System.Windows.Shapes;
using OpenCvSharp;
using System.Threading.Channels;
using System.Runtime.InteropServices;
using System.IO;

namespace HuggingDemo
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : System.Windows.Window
    {
        private  int imageSize = 224;
        private BitmapImage imageSource;
        public MainWindow()
        {
            InitializeComponent();
            //textBox.Text = AppDomain.CurrentDomain.BaseDirectory;
            textBox.Text = "D:\\WEB\\HuggingDemo\\HuggingDemo\\bin\\Debug\\net6.0-windows\\MI-GAN-main\\examples\\places2_512_freeform\\images\\Places365_val_00006549.jpg";
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
        private void Button_Click1(object sender, RoutedEventArgs e)
        {
            Uri uri01 = new Uri(textBox.Text);
            imageSource = new BitmapImage(uri01);
            doodle.pictureBox.Source = imageSource;
            doodle.doodleCanvas.Visibility = Visibility.Visible;
        }
        private void Button_Click(object sender, RoutedEventArgs e)
        {
            MiganModel_Run();
        }
        
        public void BearModel_Run()
        {
            imageSize = 224;
            Uri uri01 = new Uri(textBox.Text);
            BitmapImage imageSource = new BitmapImage(uri01);
            WriteableBitmap writableBitmap = new WriteableBitmap(imageSource);
            doodle.pictureBox.Source = imageSource;
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
                    var color = GetPixelColor(writableBitmap, x, y);

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
        public void MiganModel_Run()
        {
            //imageSize = 512;
            int imageSizeH = 512;
            int imageSizeW = 512;
            // 图片加载成功后，从图片控件中取出224*224的位图对象
            Mat image = Cv2.ImRead(textBox.Text);
            Mat resizedImage = new Mat();
            Cv2.Resize(image, resizedImage, new OpenCvSharp.Size(512, 512));
            Mat image1 = new Mat();
            Cv2.CvtColor(resizedImage, image1, ColorConversionCodes.BGRA2BGR);
            // 获取图像的通道数
            Mat[] channels = Cv2.Split(image1);

            int C = image1.Channels();
            var H = image1.Height;// 图像高度
            var W = image1.Width; // 图像宽度
            var chwArray = new byte[C * H * W];
            for (int c = 0; c < C; c++)
            {
                byte[] channelData = new byte[H * W];
                Marshal.Copy(channels[c].Data, channelData, 0, H * W);
                for (int h = 0; h < H; h++)
                {
                    for (int w = 0; w < W; w++)
                    {
                        chwArray[c * H * W + h * W + w] = channelData[h * W + w];
                    }
                }
            }
            //doodle.doodleCanvas.Background = Brushes.White;
            // 渲染 InkCanvas 为位图
            RenderTargetBitmap renderBitmap = new RenderTargetBitmap((int)doodle.doodleCanvas.ActualWidth, (int)doodle.doodleCanvas.ActualHeight, 96, 96, PixelFormats.Pbgra32);
            renderBitmap.Render(doodle.doodleCanvas);
            using (FileStream fs = new FileStream(AppDomain.CurrentDomain.BaseDirectory+"/output.png", FileMode.Create))
            {
                PngBitmapEncoder encoder = new PngBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(renderBitmap));
                encoder.Save(fs);
            }

            //MessageBox.Show("图片已保存为 output.png");
            Mat imageMark = Cv2.ImRead(AppDomain.CurrentDomain.BaseDirectory + "/output.png");
            // 将图像从 RGBA 转换为灰度图像
            Mat gray = new Mat();
            Cv2.CvtColor(imageMark, gray, ColorConversionCodes.BGRA2GRAY);

            // 将灰度图像进行二值化
            Mat binary = new Mat();
            Cv2.Threshold(gray, binary, 0, 255, ThresholdTypes.Binary);

            Mat[] channelsMark = Cv2.Split(binary);

            // 获取图像的高度和宽度
            int HMark = binary.Height;
            int WMark = binary.Width;

            // 创建一个新的数组来存储转换后的数据
            byte[] chwArrayMark = new byte[HMark * WMark];

            // 将图像的单个通道中像素值不为 255 的像素设置为 255，其余像素值保持不变
            byte[] channelDataMark = new byte[HMark * HMark];
            Marshal.Copy(channelsMark[0].Data, channelDataMark, 0, HMark * WMark);

            for (int h = 0; h < HMark; h++)
            {
                for (int w = 0; w < WMark; w++)
                {
                    chwArrayMark[h * WMark + w] = (channelDataMark[h * WMark + w] != 255) ? (byte)255 : (byte)0;
                }
            }

            Cv2.ImWrite(AppDomain.CurrentDomain.BaseDirectory + "/output.png", binary); // 将图像保存为 PNG 格式的文件
            // 显示二值化图像
            //Cv2.ImShow("Binary Image", binary);
            //Cv2.WaitKey(0);
            //Cv2.DestroyAllWindows();
            //return;

            // 设置要加载的模型的路径，跟据需要改为你的模型名称
            string modelPath = AppDomain.CurrentDomain.BaseDirectory + "migan_pipeline_v2.onnx";

            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();

                // 用Netron看到需要的输入类型是float32[None,3,224,224]
                // 第一维None表示可以传入多张图片进行推理
                // 这里只使用一张图片，所以使用的输入数据尺寸为[1, 3, 224, 224]
                var shape = new int[] { 1, 3, imageSizeW, imageSizeH };//batch_size
                var tensor = new DenseTensor<byte>(chwArray, shape);

                // 支持多个输入，对于mnist模型，只需要一个输入，输入的名称是data
                container.Add(NamedOnnxValue.CreateFromTensor("image", tensor));

                var shapeMask = new int[] { 1, 1, imageSizeW, imageSizeH };//batch_size
                var tensorMask = new DenseTensor<byte>(chwArrayMark, shapeMask);
                container.Add(NamedOnnxValue.CreateFromTensor("mask", tensorMask));

                // 推理
                var results = session.Run(container);

                // 输出结果有两个，classLabel和loss，这里只关心classLabel
                var label = results.FirstOrDefault(item => item.Name == "result")? // 取出名为classLabel的输出
                    .AsTensor<byte>();
                int pixelWidth = label.Dimensions[2];
                int pixelHeight = label.Dimensions[3];
                byte[] pixels = label.ToArray(); // 获取图像的字节数据
                int size = pixelWidth * pixelHeight;
                byte[] hwcData = new byte[pixelWidth * pixelWidth * 4]; // HWC 格式，包括 Alpha 通道

                for (int h = 0; h < pixelHeight; h++)
                {
                    for (int w = 0; w < pixelWidth; w++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            // RGB 通道
                            int chwIndex = c * size + h * pixelWidth + w;
                            byte pixelVal = pixels[chwIndex];
                            byte newPixel = pixelVal;
                            // 对像素值进行归一化和反转
                            if (pixelVal > 255)
                            {
                                newPixel = 255;
                            }
                            else if (pixelVal < 0)
                            {
                                newPixel = 0;
                            }
                            hwcData[h * pixelWidth * 4 + w * 4 + c] = newPixel; // 将像素值放入 HWC 数据中
                        }
                        hwcData[h * pixelWidth * 4 + w * 4 + 3] = 255; // Alpha 通道
                    }
                }

                // .FirstOrDefault(); // 支持多张图片同时推理，这里只推理了一张，取第一个结果值
                //byte[] rgbData = label;

                double dpiX = 96;
                double dpiY = 96;
                PixelFormat pixelFormat = PixelFormats.Pbgra32; // 假设是 BGR24 格式
                BitmapPalette palette = null; // 不使用调色板

                int stride = pixelWidth * 4; // 计算跨度

                // 使用 Create 方法创建 BitmapSource 对象
                BitmapSource bitmap = BitmapSource.Create(
                    pixelWidth, pixelHeight, dpiX, dpiY,
                    pixelFormat, palette, hwcData, stride);
                doodle.pictureBox.Source = bitmap;
                doodle.doodleCanvas.Visibility = Visibility.Collapsed;
            }
        }

        private void buttonDraw_Click(object sender, RoutedEventArgs e)
        {
            doodle.SetDrawType(Doodle.DoodleEnum.DoodleEnumType.draw);
        }

        private void buttonEraser_Click(object sender, RoutedEventArgs e)
        {
            doodle.SetDrawType(Doodle.DoodleEnum.DoodleEnumType.eraser);
        }

        private void buttonBlack_Click(object sender, RoutedEventArgs e)
        {
            doodle.SetDrawColor(Doodle.DoodleEnum.DoodleEnumColor.black);
        }

        private void buttonBlue_Click(object sender, RoutedEventArgs e)
        {
            doodle.SetDrawColor(Doodle.DoodleEnum.DoodleEnumColor.blue);
        }

        private void buttonRed_Click(object sender, RoutedEventArgs e)
        {
            doodle.SetDrawColor(Doodle.DoodleEnum.DoodleEnumColor.red);
        }

        private void buttonGreen_Click(object sender, RoutedEventArgs e)
        {
            doodle.SetDrawColor(Doodle.DoodleEnum.DoodleEnumColor.green);
        }

        private void buttonOrange_Click(object sender, RoutedEventArgs e)
        {
            doodle.SetDrawColor(Doodle.DoodleEnum.DoodleEnumColor.orange);
        }

        private void buttonBrush1_Click(object sender, RoutedEventArgs e)
        {
            doodle.SetDrawWidth(Doodle.DoodleEnum.DoodleEnumBrushType.small);
        }

        private void buttonBrush2_Click(object sender, RoutedEventArgs e)
        {
            doodle.SetDrawWidth(Doodle.DoodleEnum.DoodleEnumBrushType.middle);
        }

        private void buttonBrush3_Click(object sender, RoutedEventArgs e)
        {
            doodle.SetDrawWidth(Doodle.DoodleEnum.DoodleEnumBrushType.big);
        }

        private void buttonBrush4_Click(object sender, RoutedEventArgs e)
        {
            doodle.SetDrawWidth(Doodle.DoodleEnum.DoodleEnumBrushType.bigger);
        }
    }
}
