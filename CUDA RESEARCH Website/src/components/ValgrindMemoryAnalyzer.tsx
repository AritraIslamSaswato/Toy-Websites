import React, { useState } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { AlertCircle, Check, HardDrive, FileSearch, ArrowRight } from 'lucide-react';

interface MemoryLeak {
  id: string;
  address: string;
  size: number;
  allocatedAt: string;
  kernelName: string;
  stackTrace: string[];
  allocationTime: number;
}

interface MemoryUsageSnapshot {
  timestamp: number;
  totalAllocated: number;
  activeAllocations: number;
  potentialLeaks: number;
}

const ValgrindMemoryAnalyzer: React.FC = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [memoryLeaks, setMemoryLeaks] = useState<MemoryLeak[]>([]);
  const [memoryUsageHistory, setMemoryUsageHistory] = useState<MemoryUsageSnapshot[]>([]);
  const [selectedLeak, setSelectedLeak] = useState<MemoryLeak | null>(null);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  
  // Sample data - would be replaced with real analysis
  const sampleMemoryLeaks: MemoryLeak[] = [
    {
      id: "leak1",
      address: "0x7f8a4c3b2d1e",
      size: 1024,
      allocatedAt: "matrixMultiplication.cu:156",
      kernelName: "MatrixMul",
      stackTrace: [
        "cudaMalloc() at /usr/local/cuda/include/cuda_runtime.h:407",
        "allocateDeviceMemory() at matrixMultiplication.cu:78",
        "main() at matrixMultiplication.cu:156"
      ],
      allocationTime: 1000
    },
    {
      id: "leak2",
      address: "0x7f8a4c3b3d1e",
      size: 4096,
      allocatedAt: "convolution.cu:92",
      kernelName: "Convolution",
      stackTrace: [
        "cudaMalloc() at /usr/local/cuda/include/cuda_runtime.h:407",
        "setupConvolutionKernel() at convolution.cu:45",
        "main() at convolution.cu:92"
      ],
      allocationTime: 2000
    },
    {
      id: "leak3",
      address: "0x7f8a4c3b4d1e",
      size: 512,
      allocatedAt: "fft.cu:203",
      kernelName: "FFT",
      stackTrace: [
        "cudaMalloc() at /usr/local/cuda/include/cuda_runtime.h:407",
        "allocateFFTBuffers() at fft.cu:128",
        "setupFFT() at fft.cu:175",
        "main() at fft.cu:203"
      ],
      allocationTime: 3000
    }
  ];
  
  const sampleMemoryUsageHistory: MemoryUsageSnapshot[] = [
    { timestamp: 0, totalAllocated: 0, activeAllocations: 0, potentialLeaks: 0 },
    { timestamp: 1000, totalAllocated: 1024, activeAllocations: 1, potentialLeaks: 0 },
    { timestamp: 2000, totalAllocated: 5120, activeAllocations: 2, potentialLeaks: 1 },
    { timestamp: 3000, totalAllocated: 5632, activeAllocations: 3, potentialLeaks: 1 },
    { timestamp: 4000, totalAllocated: 5632, activeAllocations: 3, potentialLeaks: 2 },
    { timestamp: 5000, totalAllocated: 5632, activeAllocations: 3, potentialLeaks: 3 }
  ];
  
  const runAnalysis = () => {
    setIsAnalyzing(true);
    setAnalysisComplete(false);
    
    // Simulate analysis process
    setTimeout(() => {
      setMemoryLeaks(sampleMemoryLeaks);
      setMemoryUsageHistory(sampleMemoryUsageHistory);
      setIsAnalyzing(false);
      setAnalysisComplete(true);
    }, 3000);
  };
  
  const formatBytes = (bytes: number): string => {
    if (bytes < 1024) return bytes + ' B';
    else if (bytes < 1048576) return (bytes / 1024).toFixed(2) + ' KB';
    else return (bytes / 1048576).toFixed(2) + ' MB';
  };
  
  const getTotalLeakedMemory = (): number => {
    return memoryLeaks.reduce((total, leak) => total + leak.size, 0);
  };
  
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center mb-6">
        <HardDrive className="h-6 w-6 text-purple-600 mr-2" />
        <h2 className="text-2xl font-bold">CUDA Memory Leak Analyzer</h2>
      </div>
      
      <div className="mb-8 bg-purple-50 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-2 text-purple-800">About Memory Leak Analysis</h3>
        <p className="text-gray-700 mb-3">
          This tool analyzes CUDA applications for memory leaks by tracking memory allocations and deallocations.
          It identifies memory blocks that were allocated but never freed, helping you optimize memory usage and
          prevent resource exhaustion in long-running applications.
        </p>
        <div className="flex items-center">
          <AlertCircle className="h-5 w-5 text-purple-500 mr-2" />
          <p className="text-sm text-purple-700">
            Memory leaks in CUDA applications can lead to reduced performance, out-of-memory errors, and increased
            energy consumption due to unnecessary memory transfers.
          </p>
        </div>
      </div>
      
      <div className="mb-6">
        <button 
          onClick={runAnalysis}
          disabled={isAnalyzing}
          className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 transition-colors disabled:bg-purple-300"
        >
          {isAnalyzing ? 'Analyzing Memory Usage...' : 'Run Memory Leak Analysis'}
        </button>
      </div>
      
      {analysisComplete && (
        <>
          <div className="mb-8">
            <h3 className="text-xl font-semibold mb-4">Memory Usage Timeline</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={memoryUsageHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp" 
                  label={{ value: 'Time (ms)', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  label={{ value: 'Memory (bytes)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip formatter={(value) => formatBytes(value as number)} />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="totalAllocated" 
                  name="Total Allocated Memory" 
                  stroke="#8884d8" 
                  activeDot={{ r: 8 }} 
                />
                <Line 
                  type="monotone" 
                  dataKey="potentialLeaks" 
                  name="Potential Leaks" 
                  stroke="#ff7300" 
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          
          <div className="mb-8">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold">Detected Memory Leaks</h3>
              <div className="bg-red-100 text-red-800 px-3 py-1 rounded-full text-sm font-medium">
                Total Leaked: {formatBytes(getTotalLeakedMemory())}
              </div>
            </div>
            
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Address</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Size</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Kernel</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Allocated At</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {memoryLeaks.map((leak) => (
                    <tr key={leak.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-mono">{leak.address}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">{formatBytes(leak.size)}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">{leak.kernelName}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">{leak.allocatedAt}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        <button 
                          onClick={() => setSelectedLeak(leak)}
                          className="text-indigo-600 hover:text-indigo-900 font-medium"
                        >
                          View Details
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          
          {selectedLeak && (
            <div className="bg-gray-50 p-4 rounded-lg mb-4">
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-lg font-semibold">Memory Leak Details</h3>
                <button 
                  onClick={() => setSelectedLeak(null)}
                  className="text-gray-500 hover:text-gray-700"
                >
                  ✕
                </button>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <p className="text-sm text-gray-500">Address</p>
                  <p className="font-mono">{selectedLeak.address}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Size</p>
                  <p>{formatBytes(selectedLeak.size)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Kernel</p>
                  <p>{selectedLeak.kernelName}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Allocated At</p>
                  <p>{selectedLeak.allocatedAt}</p>
                </div>
              </div>
              
              <div className="mb-4">
                <h4 className="text-md font-medium mb-2">Stack Trace</h4>
                <div className="bg-gray-100 p-3 rounded font-mono text-sm">
                  {selectedLeak.stackTrace.map((line, index) => (
                    <div key={index} className="mb-1">{line}</div>
                  ))}
                </div>
              </div>
              
              <div className="bg-yellow-50 border-l-4 border-yellow-400 p-3">
                <div className="flex">
                  <AlertCircle className="h-5 w-5 text-yellow-400 mr-2" />
                  <div>
                    <h4 className="text-sm font-medium text-yellow-800">Recommendation</h4>
                    <p className="text-sm text-yellow-700">
                      Add a corresponding <code>cudaFree()</code> call after the kernel execution is complete.
                      Consider using RAII patterns or smart pointers to manage CUDA memory allocations.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <div className="bg-indigo-50 p-4 rounded-lg">
            <h3 className="text-lg font-semibold mb-3 text-indigo-800">Optimization Recommendations</h3>
            <ul className="space-y-2">
              <li className="flex items-start">
                <ArrowRight className="h-5 w-5 text-indigo-500 mt-0.5 mr-2 flex-shrink-0" />
                <span>Implement RAII (Resource Acquisition Is Initialization) patterns for CUDA memory management</span>
              </li>
              <li className="flex items-start">
                <ArrowRight className="h-5 w-5 text-indigo-500 mt-0.5 mr-2 flex-shrink-0" />
                <span>Use smart pointer wrappers for CUDA allocations to ensure proper cleanup</span>
              </li>
              <li className="flex items-start">
                <ArrowRight className="h-5 w-5 text-indigo-500 mt-0.5 mr-2 flex-shrink-0" />
                <span>Consider using Unified Memory with managed allocations for simpler memory management</span>
              </li>
              <li className="flex items-start">
                <ArrowRight className="h-5 w-5 text-indigo-500 mt-0.5 mr-2 flex-shrink-0" />
                <span>Add explicit error checking after each CUDA memory operation</span>
              </li>
            </ul>
          </div>
        </>
      )}
    </div>
  );
};

export default ValgrindMemoryAnalyzer;