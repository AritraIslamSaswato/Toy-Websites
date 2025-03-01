import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface MemoryAnalysisData {
  kernelName: string;
  globalMemoryAccess: number;
  sharedMemoryUsage: number;
  pageFaults: number;
  efficiency: number;
}

const MemoryAnalyzer: React.FC = () => {
  const [analysisData, setAnalysisData] = useState<MemoryAnalysisData[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  // Sample data - would be replaced with real analysis
  const sampleData: MemoryAnalysisData[] = [
    { kernelName: 'MatrixMul', globalMemoryAccess: 1240, sharedMemoryUsage: 48, pageFaults: 32, efficiency: 76 },
    { kernelName: 'Convolution', globalMemoryAccess: 980, sharedMemoryUsage: 64, pageFaults: 18, efficiency: 82 },
    { kernelName: 'FFT', globalMemoryAccess: 1560, sharedMemoryUsage: 32, pageFaults: 45, efficiency: 68 },
    { kernelName: 'Reduction', globalMemoryAccess: 420, sharedMemoryUsage: 16, pageFaults: 8, efficiency: 94 },
  ];
  
  const runAnalysis = () => {
    setIsAnalyzing(true);
    // Simulate analysis process
    setTimeout(() => {
      setAnalysisData(sampleData);
      setIsAnalyzing(false);
    }, 2000);
  };
  
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold mb-6">CUDA Memory Access Analyzer</h2>
      
      <div className="mb-6">
        <button 
          onClick={runAnalysis}
          disabled={isAnalyzing}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:bg-blue-300"
        >
          {isAnalyzing ? 'Analyzing...' : 'Run Memory Analysis'}
        </button>
      </div>
      
      {analysisData.length > 0 && (
        <>
          <div className="mb-8">
            <h3 className="text-xl font-semibold mb-4">Memory Access Patterns</h3>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={analysisData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="kernelName" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="globalMemoryAccess" name="Global Memory Access (MB)" fill="#3b82f6" />
                <Bar dataKey="sharedMemoryUsage" name="Shared Memory Usage (KB)" fill="#10b981" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          <div className="mb-8">
            <h3 className="text-xl font-semibold mb-4">Page Fault Analysis</h3>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={analysisData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="kernelName" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="pageFaults" name="Page Faults" fill="#ef4444" />
                <Bar dataKey="efficiency" name="Memory Efficiency (%)" fill="#8b5cf6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          <div className="bg-blue-50 p-4 rounded-md">
            <h3 className="text-lg font-semibold mb-2">Optimization Recommendations</h3>
            <ul className="list-disc pl-6 space-y-2">
              <li>Use shared memory for the FFT kernel to reduce global memory accesses</li>
              <li>Implement memory prefetching for MatrixMul to reduce page faults</li>
              <li>Optimize memory coalescing in the Convolution kernel</li>
              <li>Consider using Unified Memory with memory advising for the FFT kernel</li>
            </ul>
          </div>
        </>
      )}
    </div>
  );
};

export default MemoryAnalyzer;