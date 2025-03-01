import React, { useState } from 'react';
import { Cpu, MemoryStick as Memory, HardDrive, Activity } from 'lucide-react';

interface MemoryBlock {
  id: string;
  size: number;
  type: 'allocated' | 'free' | 'leaked';
  address: string;
  function?: string;
  line?: number;
}

interface MemoryProfilerProps {
  initialBlocks?: MemoryBlock[];
}

const MemoryProfiler: React.FC<MemoryProfilerProps> = ({ 
  initialBlocks = [] 
}) => {
  const [blocks] = useState<MemoryBlock[]>(initialBlocks.length > 0 ? initialBlocks : [
    { id: '1', size: 1024, type: 'allocated', address: '0x7f8a4c3b2d1e', function: 'cudaMalloc', line: 42 },
    { id: '2', size: 512, type: 'allocated', address: '0x7f8a4c3b3d1e', function: 'cudaMalloc', line: 43 },
    { id: '3', size: 2048, type: 'free', address: '0x7f8a4c3b4d1e', function: 'cudaFree', line: 65 },
    { id: '4', size: 256, type: 'leaked', address: '0x7f8a4c3b5d1e', function: 'cudaMalloc', line: 78 },
    { id: '5', size: 4096, type: 'allocated', address: '0x7f8a4c3b6d1e', function: 'cudaMalloc', line: 92 },
    { id: '6', size: 128, type: 'leaked', address: '0x7f8a4c3b7d1e', function: 'cudaMalloc', line: 103 },
  ]);

  // Optimize memory calculations with useMemo
  const memoryStats = React.useMemo(() => {
    const totalAllocated = blocks.filter(b => b.type === 'allocated').reduce((sum, b) => sum + b.size, 0);
    const totalLeaked = blocks.filter(b => b.type === 'leaked').reduce((sum, b) => sum + b.size, 0);
    const totalFree = blocks.filter(b => b.type === 'free').reduce((sum, b) => sum + b.size, 0);
    const totalMemory = totalAllocated + totalLeaked + totalFree;
    
    return {
      totalAllocated,
      totalLeaked,
      totalFree,
      totalMemory
    };
  }, [blocks]);
  
  // Use the memoized values
  const { totalAllocated, totalLeaked, totalFree, totalMemory } = memoryStats;

  const getBlockColor = (type: string) => {
    switch (type) {
      case 'allocated': return 'bg-blue-500';
      case 'free': return 'bg-green-500';
      case 'leaked': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getBlockWidth = (size: number) => {
    return `${(size / totalMemory) * 100}%`;
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-xl font-bold mb-4 flex items-center">
        <Memory className="mr-2" /> Memory Profiler
      </h3>
      
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-blue-100 p-4 rounded-lg">
          <div className="flex items-center text-blue-700 mb-1">
            <Cpu className="mr-2" size={18} />
            <span className="font-semibold">Allocated</span>
          </div>
          <p className="text-2xl font-bold">{(totalAllocated / 1024).toFixed(2)} KB</p>
        </div>
        
        <div className="bg-red-100 p-4 rounded-lg">
          <div className="flex items-center text-red-700 mb-1">
            <HardDrive className="mr-2" size={18} />
            <span className="font-semibold">Leaked</span>
          </div>
          <p className="text-2xl font-bold">{(totalLeaked / 1024).toFixed(2)} KB</p>
        </div>
        
        <div className="bg-green-100 p-4 rounded-lg">
          <div className="flex items-center text-green-700 mb-1">
            <Activity className="mr-2" size={18} />
            <span className="font-semibold">Freed</span>
          </div>
          <p className="text-2xl font-bold">{(totalFree / 1024).toFixed(2)} KB</p>
        </div>
      </div>
      
      <div className="mb-6">
        <h4 className="font-semibold mb-2">Memory Allocation Visualization</h4>
        <div className="h-8 w-full flex rounded-md overflow-hidden">
          {blocks.map(block => (
            <div 
              key={block.id}
              className={`${getBlockColor(block.type)} hover:opacity-80 transition-opacity relative group`}
              style={{ width: getBlockWidth(block.size) }}
            >
              <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block bg-gray-900 text-white text-xs rounded p-2 z-10 whitespace-nowrap">
                <p><strong>Size:</strong> {(block.size / 1024).toFixed(2)} KB</p>
                <p><strong>Address:</strong> {block.address}</p>
                {block.function && <p><strong>Function:</strong> {block.function}</p>}
                {block.line && <p><strong>Line:</strong> {block.line}</p>}
              </div>
            </div>
          ))}
        </div>
        <div className="flex justify-between mt-2 text-xs text-gray-600">
          <span>0 KB</span>
          <span>{(totalMemory / 1024 / 2).toFixed(2)} KB</span>
          <span>{(totalMemory / 1024).toFixed(2)} KB</span>
        </div>
      </div>
      
      <div>
        <h4 className="font-semibold mb-2">Memory Allocation Details</h4>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Address</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Size (KB)</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Function</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Line</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {blocks.map(block => (
                <tr key={block.id}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-mono">{block.address}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">{(block.size / 1024).toFixed(2)}</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                      ${block.type === 'allocated' ? 'bg-blue-100 text-blue-800' : 
                        block.type === 'free' ? 'bg-green-100 text-green-800' : 
                        'bg-red-100 text-red-800'}`}>
                      {block.type}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">{block.function || '-'}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">{block.line || '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default MemoryProfiler;