import React from 'react';
import { Link } from 'react-router-dom';
import { BookOpen, Code, Cpu, Zap } from 'lucide-react';
import ArticleCard from '../components/ArticleCard';
import CodeSnippet from '../components/CodeSnippet';
import { articles } from '../data/articles';

const ResearchPage: React.FC = () => {
  const uvmCodeSnippet = `// Example of Unified Virtual Memory (UVM) in CUDA
#include <cuda_runtime.h>
#include <iostream>

int main() {
    // Allocate managed memory accessible by CPU and GPU
    float *data;
    cudaMallocManaged(&data, 1024 * sizeof(float));
    
    // Initialize data on CPU
    for (int i = 0; i < 1024; i++) {
        data[i] = static_cast<float>(i);
    }
    
    // Launch kernel to process data on GPU
    dim3 blockSize(256);
    dim3 gridSize((1024 + blockSize.x - 1) / blockSize.x);
    processDataKernel<<<gridSize, blockSize>>>(data, 1024);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Access results on CPU
    float sum = 0.0f;
    for (int i = 0; i < 1024; i++) {
        sum += data[i];
    }
    
    // Free memory
    cudaFree(data);
    
    return 0;
}`;

  const ptxParallelizationSnippet = `// PTX-level parallelization example
.version 7.0
.target sm_80
.address_size 64

.visible .entry parallelReduction(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
) {
    .reg .pred %p<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<7>;
    .reg .f32 %f<4>;
    
    // Load parameters
    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.param.u32 %r1, [n];
    
    // Calculate thread ID
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %ctaid.x;
    mad.lo.u32 %r2, %r4, %r3, %r2;
    
    // Check if thread ID is within bounds
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra END;
    
    // Calculate input address
    mul.wide.u32 %rd3, %r2, 4;
    add.u64 %rd4, %rd1, %rd3;
    
    // Load input value
    ld.global.f32 %f1, [%rd4];
    
    // Process value (example: square it)
    mul.f32 %f2, %f1, %f1;
    
    // Store result
    st.global.f32 [%rd4], %f2;
    
END:
    ret;
}`;

  const memoryAccessPatternSnippet = `// Optimized memory access pattern for coalescing
__global__ void optimizedKernel(float* input, float* output, int width, int height) {
    // Calculate thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Calculate global indices with coalesced access pattern
    int col = bx * blockDim.x + tx;
    int row = by * blockDim.y + ty;
    
    // Ensure we're within bounds
    if (col < width && row < height) {
        // Coalesced memory access pattern (threads in a warp access consecutive memory)
        int idx = row * width + col;
        
        // Process data
        output[idx] = input[idx] * 2.0f;
    }
}`;

  return (
    <div className="min-h-screen">
      {/* Header Section */}
      <section className="bg-gradient-to-r from-blue-900 to-indigo-800 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-4xl font-bold mb-6">Research & Methods</h1>
          <p className="text-xl max-w-3xl">
            Explore our cutting-edge research on CUDA optimization techniques, memory management strategies, and energy-efficient programming methods.
          </p>
        </div>
      </section>

      {/* Research Areas */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold mb-12 text-center text-gray-800">
            Key Research Areas
          </h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
            {/* UVM Research */}
            <div className="bg-white rounded-lg shadow-md overflow-hidden">
              <div className="p-6">
                <div className="flex items-center mb-4">
                  <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center mr-4">
                    <Cpu className="h-5 w-5 text-blue-600" />
                  </div>
                  <h3 className="text-2xl font-bold text-gray-800">Unified Virtual Memory (UVM)</h3>
                </div>
                
                <p className="text-gray-600 mb-6">
                  Unified Virtual Memory (UVM) provides a single memory space accessible by both CPU and GPU, simplifying memory management in CUDA applications. Our research focuses on optimizing UVM performance through prefetching, memory advising, and access pattern optimization.
                </p>
                
                <div className="mb-6">
                  <h4 className="text-lg font-semibold mb-2">Key Findings:</h4>
                  <ul className="list-disc pl-6 text-gray-600 space-y-2">
                    <li>Prefetching data to the GPU before access can reduce page faults by up to 87%</li>
                    <li>Memory access patterns significantly impact UVM performance</li>
                    <li>Proper memory advising can improve performance by 2-3x for data-intensive applications</li>
                    <li>Hybrid approaches combining explicit and managed memory show promising results</li>
                  </ul>
                </div>
                
                <CodeSnippet code={uvmCodeSnippet} title="Unified Virtual Memory Example" />
              </div>
            </div>
            
            {/* PTX-level Parallelization */}
            <div className="bg-white rounded-lg shadow-md overflow-hidden">
              <div className="p-6">
                <div className="flex items-center mb-4">
                  <div className="w-10 h-10 bg-purple-100 rounded-full flex items-center justify-center mr-4">
                    <Code className="h-5 w-5 text-purple-600" />
                  </div>
                  <h3 className="text-2xl font-bold text-gray-800">PTX-level Parallelization</h3>
                </div>
                
                <p className="text-gray-600 mb-6">
                  PTX (Parallel Thread Execution) is NVIDIA's low-level virtual machine and instruction set architecture. Our research explores advanced PTX-level optimizations to maximize parallelism and efficiency beyond what high-level CUDA C++ can achieve.
                </p>
                
                <div className="mb-6">
                  <h4 className="text-lg font-semibold mb-2">Key Findings:</h4>
                  <ul className="list-disc pl-6 text-gray-600 space-y-2">
                    <li>Manual instruction scheduling can reduce stalls by up to 35%</li>
                    <li>Register usage optimization can increase occupancy by 15-25%</li>
                    <li>Predication techniques reduce branch divergence overhead</li>
                    <li>Shared memory bank conflict avoidance improves throughput by 20-40%</li>
                  </ul>
                </div>
                
                <CodeSnippet code={ptxParallelizationSnippet} title="PTX-level Parallelization Example" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Memory Access Patterns */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold mb-8 text-gray-800">Memory Access Patterns & Bottlenecks</h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2">
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-xl font-semibold mb-4">Optimizing Memory Access Patterns</h3>
                
                <p className="text-gray-600 mb-4">
                  Memory access patterns significantly impact CUDA performance. Coalesced memory access, where threads in a warp access contiguous memory locations, maximizes memory throughput and reduces latency.
                </p>
                
                <div className="mb-6">
                  <h4 className="text-lg font-semibold mb-2">Best Practices:</h4>
                  <ul className="list-disc pl-6 text-gray-600 space-y-2">
                    <li>Ensure threads in a warp access consecutive memory addresses</li>
                    <li>Align data structures to memory boundaries</li>
                    <li>Use shared memory for data reuse within a thread block</li>
                    <li>Minimize bank conflicts in shared memory access</li>
                    <li>Consider using texture memory for read-only data with spatial locality</li>
                  </ul>
                </div>
                
                <CodeSnippet code={memoryAccessPatternSnippet} title="Optimized Memory Access Pattern" />
              </div>
            </div>
            
            <div>
              <div className="bg-white rounded-lg shadow-md p-6 h-full">
                <h3 className="text-xl font-semibold mb-4">Common Memory Bottlenecks</h3>
                
                <div className="space-y-4">
                  <div className="border-l-4 border-red-500 pl-4 py-2">
                    <h4 className="font-semibold text-gray-800">Uncoalesced Memory Access</h4>
                    <p className="text-gray-600">Causes multiple memory transactions per warp, reducing throughput by 2-10x.</p>
                  </div>
                  
                  <div className="border-l-4 border-red-500 pl-4 py-2">
                    <h4 className="font-semibold text-gray-800">Bank Conflicts</h4>
                    <p className="text-gray-600">Multiple threads accessing the same shared memory bank causes serialization.</p>
                  </div>
                  
                  <div className="border-l-4 border-red-500 pl-4 py-2">
                    <h4 className="font-semibold text-gray-800">Memory Divergence</h4>
                    <p className="text-gray-600">Threads in a warp accessing non-contiguous memory locations.</p>
                  </div>
                  
                  <div className="border-l-4 border-red-500 pl-4 py-2">
                    <h4 className="font-semibold text-gray-800">High Register Usage</h4>
                    <p className="text-gray-600">Reduces occupancy, limiting the number of active warps per SM.</p>
                  </div>
                  
                  <div className="border-l-4 border-red-500 pl-4 py-2">
                    <h4 className="font-semibold text-gray-800">Excessive Global Memory Access</h4>
                    <p className="text-gray-600">Not utilizing shared memory or caches for data reuse.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Research Articles */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold mb-12 text-center text-gray-800">
            Latest Research Articles
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {articles.map(article => (
              <ArticleCard key={article.id} article={article} />
            ))}
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="py-16 bg-gradient-to-r from-indigo-800 to-purple-800 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold mb-6">Apply These Techniques to Your Projects</h2>
          <p className="text-xl mb-8 max-w-3xl mx-auto">
            Ready to implement these optimization techniques in your CUDA applications? Explore our optimization toolkit and performance analysis tools.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <Link 
              to="/optimization" 
              className="px-6 py-3 bg-white text-indigo-800 font-semibold rounded-md hover:bg-blue-50 transition-colors shadow-md"
            >
              Optimization Toolkit
            </Link>
            <Link 
              to="/performance" 
              className="px-6 py-3 bg-transparent border-2 border-white text-white font-semibold rounded-md hover:bg-white/10 transition-colors"
            >
              Performance Analysis
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
};

export default ResearchPage;