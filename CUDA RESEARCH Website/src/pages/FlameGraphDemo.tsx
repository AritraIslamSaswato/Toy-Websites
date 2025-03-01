import React from 'react';
import FlameGraph from '../components/FlameGraph';

const FlameGraphDemo: React.FC = () => {
  // Sample CUDA kernel execution data
  const flameGraphData = {
    name: "CUDA Kernel Execution",
    value: 100,
    children: [
      {
        name: "Matrix Multiplication",
        value: 45,
        color: "#3b82f6",
        children: [
          { name: "Memory Access", value: 20, color: "#60a5fa" },
          { name: "Computation", value: 15, color: "#2563eb" },
          { name: "Synchronization", value: 10, color: "#1d4ed8" }
        ]
      },
      {
        name: "Memory Operations",
        value: 35,
        color: "#10b981",
        children: [
          { name: "Data Transfer (H2D)", value: 20, color: "#34d399" },
          { name: "Data Transfer (D2H)", value: 15, color: "#059669" }
        ]
      },
      {
        name: "Kernel Launch",
        value: 20,
        color: "#f59e0b",
        children: [
          { name: "Grid Setup", value: 12, color: "#fbbf24" },
          { name: "Launch Overhead", value: 8, color: "#d97706" }
        ]
      }
    ]
  };

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">CUDA Performance Flame Graph</h1>
        <p className="mb-8 text-gray-600">
          Visualizing the execution time breakdown of CUDA operations
        </p>
        
        <div className="bg-white p-6 rounded-lg shadow-md">
          <FlameGraph data={flameGraphData} width={900} height={400} />
        </div>
        
        <div className="mt-8 bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">How to Read This Flame Graph</h2>
          <ul className="list-disc pl-6 space-y-2 text-gray-700">
            <li>Each block represents a CUDA operation or sub-operation</li>
            <li>The width of each block indicates the time spent in that operation</li>
            <li>Blocks are stacked to show parent-child relationships</li>
            <li>Hover over blocks to see detailed timing information</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default FlameGraphDemo;