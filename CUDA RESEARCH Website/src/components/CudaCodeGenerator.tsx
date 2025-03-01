import React, { useState } from 'react';
import { Code, Copy, Check } from 'lucide-react';
import CodeSnippet from './CodeSnippet';

interface TemplateOptions {
  kernelName: string;
  dataType: string;
  blockSize: number;
  useSharedMemory: boolean;
  includeErrorChecking: boolean;
}

const CudaCodeGenerator: React.FC = () => {
  const [options, setOptions] = useState<TemplateOptions>({
    kernelName: 'myKernel',
    dataType: 'float',
    blockSize: 256,
    useSharedMemory: false,
    includeErrorChecking: true,
  });

  const generateCode = () => {
    const sharedMemoryCode = options.useSharedMemory ? `
    __shared__ ${options.dataType} sharedData[${options.blockSize}];` : '';

    const errorChecking = options.includeErrorChecking ? `
    // Error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
        return -1;
    }` : '';

    return `#include <cuda_runtime.h>
#include <stdio.h>

__global__ void ${options.kernelName}(${options.dataType}* input, ${options.dataType}* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;${sharedMemoryCode}

    if (tid < n) {
        ${options.dataType} value = input[tid];
        // Add your computation here
        output[tid] = value;
    }
}

int main() {
    const int N = 1024;
    size_t size = N * sizeof(${options.dataType});

    // Allocate host memory
    ${options.dataType} *h_input = (${options.dataType}*)malloc(size);
    ${options.dataType} *h_output = (${options.dataType}*)malloc(size);

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<${options.dataType}>(i);
    }

    // Allocate device memory
    ${options.dataType} *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy input to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = ${options.blockSize};
    int numBlocks = (N + blockSize - 1) / blockSize;
    ${options.kernelName}<<<numBlocks, blockSize>>>(d_input, d_output, N);${errorChecking}

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}`;
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center mb-6">
        <Code className="h-6 w-6 text-blue-600 mr-2" />
        <h2 className="text-2xl font-bold">CUDA Code Generator</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Kernel Name
            </label>
            <input
              type="text"
              value={options.kernelName}
              onChange={(e) => setOptions({ ...options, kernelName: e.target.value })}
              className="w-full px-3 py-2 border rounded-md"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Data Type
            </label>
            <select
              value={options.dataType}
              onChange={(e) => setOptions({ ...options, dataType: e.target.value })}
              className="w-full px-3 py-2 border rounded-md"
            >
              <option value="float">float</option>
              <option value="double">double</option>
              <option value="int">int</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Block Size
            </label>
            <input
              type="number"
              value={options.blockSize}
              onChange={(e) => setOptions({ ...options, blockSize: parseInt(e.target.value) })}
              className="w-full px-3 py-2 border rounded-md"
            />
          </div>

          <div className="flex items-center">
            <input
              type="checkbox"
              checked={options.useSharedMemory}
              onChange={(e) => setOptions({ ...options, useSharedMemory: e.target.checked })}
              className="mr-2"
            />
            <label className="text-sm font-medium text-gray-700">
              Use Shared Memory
            </label>
          </div>

          <div className="flex items-center">
            <input
              type="checkbox"
              checked={options.includeErrorChecking}
              onChange={(e) => setOptions({ ...options, includeErrorChecking: e.target.checked })}
              className="mr-2"
            />
            <label className="text-sm font-medium text-gray-700">
              Include Error Checking
            </label>
          </div>
        </div>

        <div>
          <CodeSnippet
            code={generateCode()}
            language="cpp"
            title="Generated CUDA Code"
          />
        </div>
      </div>
    </div>
  );
};

export default CudaCodeGenerator;