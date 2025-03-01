export interface OptimizationTechnique {
  id: number;
  title: string;
  description: string;
  benefits: string[];
  codeSnippet: string;
  performanceImprovement: string;
  energySavings: string;
}

export const optimizationTechniques: OptimizationTechnique[] = [
  {
    id: 1,
    title: "Coalesced Memory Access",
    description: "Ensuring that threads in a warp access memory in a coalesced pattern to maximize memory bandwidth utilization.",
    benefits: [
      "Reduced memory access latency",
      "Improved memory throughput",
      "Lower energy consumption per operation"
    ],
    codeSnippet: `// Unoptimized memory access
__global__ void unoptimizedKernel(float* data, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        // Non-coalesced access pattern
        float value = data[idy * width + idx];
        // Process value...
    }
}

// Optimized coalesced memory access
__global__ void optimizedKernel(float* data, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < width && idy < height) {
        // Coalesced access pattern
        float value = data[idx * height + idy];
        // Process value...
    }
}`,
    performanceImprovement: "Up to 3.5x speedup for memory-bound applications",
    energySavings: "25-30% reduction in energy consumption"
  },
  {
    id: 2,
    title: "Shared Memory Utilization",
    description: "Using shared memory as a software-managed cache to reduce global memory accesses and improve data locality.",
    benefits: [
      "Reduced global memory bandwidth requirements",
      "Lower memory access latency",
      "Improved thread cooperation"
    ],
    codeSnippet: `// Without shared memory
__global__ void withoutSharedMemKernel(float* input, float* output, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread reads from global memory multiple times
    float sum = 0.0f;
    for (int i = -2; i <= 2; i++) {
        int pos = idx + i;
        if (pos >= 0 && pos < width) {
            sum += input[pos];
        }
    }
    output[idx] = sum / 5.0f;
}

// With shared memory
__global__ void withSharedMemKernel(float* input, float* output, int width) {
    __shared__ float sharedData[BLOCK_SIZE + 4];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x + 2;  // Offset for halo cells
    
    // Load data into shared memory (including halo cells)
    sharedData[localIdx] = (idx < width) ? input[idx] : 0.0f;
    
    // Load halo cells
    if (threadIdx.x < 2) {
        int pos = idx - 2;
        sharedData[localIdx - 2] = (pos >= 0) ? input[pos] : 0.0f;
    }
    if (threadIdx.x >= BLOCK_SIZE - 2) {
        int pos = idx + 2;
        sharedData[localIdx + 2] = (pos < width) ? input[pos] : 0.0f;
    }
    
    __syncthreads();
    
    // Compute using shared memory
    if (idx < width) {
        float sum = 0.0f;
        for (int i = -2; i <= 2; i++) {
            sum += sharedData[localIdx + i];
        }
        output[idx] = sum / 5.0f;
    }
}`,
    performanceImprovement: "2-4x speedup for algorithms with data reuse",
    energySavings: "20-40% reduction in energy consumption"
  },
  {
    id: 3,
    title: "Warp Divergence Minimization",
    description: "Restructuring code to minimize control flow divergence within warps, ensuring threads execute the same instructions.",
    benefits: [
      "Improved instruction throughput",
      "Reduced execution time",
      "Better utilization of SIMT architecture"
    ],
    codeSnippet: `// With warp divergence
__global__ void divergentKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Threads in the same warp may take different paths
        if (data[idx] > 0) {
            data[idx] = sqrt(data[idx]);
        } else {
            data[idx] = 0;
        }
    }
}

// Minimized warp divergence
__global__ void nonDivergentKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // All threads execute the same instructions
        float value = data[idx];
        float result = (value > 0) ? sqrt(value) : 0;
        data[idx] = result;
    }
}`,
    performanceImprovement: "Up to 2x speedup for control-flow heavy code",
    energySavings: "15-25% reduction in energy consumption"
  },
  {
    id: 4,
    title: "Dynamic Parallelism",
    description: "Using CUDA dynamic parallelism to launch nested kernels from within a kernel, optimizing workload distribution.",
    benefits: [
      "Reduced kernel launch overhead",
      "Better adaptation to data-dependent workloads",
      "Improved load balancing"
    ],
    codeSnippet: `// Without dynamic parallelism
__global__ void processDataKernel(DataNode* nodes, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numNodes) {
        // Process node with fixed work
        processNode(nodes[idx]);
    }
}

// With dynamic parallelism
__global__ void adaptiveProcessingKernel(DataNode* nodes, int numNodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numNodes) {
        DataNode& node = nodes[idx];
        
        // Dynamically launch child kernels based on node complexity
        if (node.complexity > THRESHOLD) {
            int childBlocks = (node.dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
            processComplexNodeKernel<<<childBlocks, BLOCK_SIZE>>>(node);
        } else {
            // Process simple node directly
            processSimpleNode(node);
        }
    }
}`,
    performanceImprovement: "1.5-3x speedup for irregular workloads",
    energySavings: "10-30% reduction in energy consumption"
  }
];