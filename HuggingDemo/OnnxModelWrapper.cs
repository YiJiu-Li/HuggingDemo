using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HuggingDemo
{
    public class OnnxModelWrapper
    {
        private string _modelPath;
        private InferenceSession _session;
        private readonly string _inputName;

        public OnnxModelWrapper(string modelPath)
        {
            _modelPath = modelPath;
            _session = new InferenceSession(_modelPath);
            // 获取模型的输入节点名称  
            _inputName = _session.InputMetadata.Keys.First();
        }

        public float[] RunInference(float[] inputData, int inputSize)
        {
            // 将输入数据调整为 (1, 28, 28) 形状的张量  
            var reshapedInputData = new DenseTensor<float>(new[] { 1, 28, 28 });
            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    reshapedInputData[0, i, j] = inputData[i * 28 + j];
                }
            }

            // 创建输入 NamedOnnxValue  
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, reshapedInputData) };

            // 运行模型推理  
            using var results = _session.Run(inputs);

            // 获取输出数据  
            float[] outputData = results.ToArray()[0].AsEnumerable<float>().ToArray();

            return outputData;
        }
    }  
}
