import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { Database, Layers, Zap, AlertTriangle } from 'lucide-react';

interface MemoryAccessEvent {
  address: string;
  type: 'cache_hit' | 'cache_miss' | 'similarity_hit';
  similarityScore?: number;
  kernelName: string;
  timestamp: number;
}

interface SimilarityAnalysisResult {
  kernelName: string;
  cacheHits: number;
  cacheMisses: number;
  similarityHits: number;
  energySavings: number;
  averageSimilarityScore: number;
}

const SimilarityHitAnalyzer: React.FC = () => {
  const [analysisResults, setAnalysisResults] = useState<SimilarityAnalysisResult[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedKernel, setSelectedKernel] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.75);
  
  // Sample data - would be replaced with real analysis
  const sampleAccessEvents: MemoryAccessEvent[] = [
    // Matrix Multiplication kernel events
    { address: '0x7f8a4c3b2d1e', type: 'cache_hit', kernelName: 'MatrixMul', timestamp: 1000 },
    { address: '0x7f8a4c3b2d2e', type: 'similarity_hit', similarityScore: 0.92, kernelName: 'MatrixMul', timestamp: 1001 },
    { address: '0x7f8a4c3b2d3e', type: 'cache_miss', kernelName: 'MatrixMul', timestamp: 1002 },
    { address: '0x7f8a4c3b2d4e', type: 'similarity_hit', similarityScore: 0.85, kernelName: 'MatrixMul', timestamp: 1003 },
    { address: '0x7f8a4c3b2d5e', type: 'similarity_hit', similarityScore: 0.78, kernelName: 'MatrixMul', timestamp: 1004 },
    { address: '0x7f8a4c3b2d6e', type: 'cache_hit', kernelName: 'MatrixMul', timestamp: 1005 },
    
    // Convolution kernel events
    { address: '0x7f8a4c3b3d1e', type: 'cache_miss', kernelName: 'Convolution', timestamp: 2000 },
    { address: '0x7f8a4c3b3d2e', type: 'similarity_hit', similarityScore: 0.95, kernelName: 'Convolution', timestamp: 2001 },
    { address: '0x7f8a4c3b3d3e', type: 'similarity_hit', similarityScore: 0.88, kernelName: 'Convolution', timestamp: 2002 },
    { address: '0x7f8a4c3b3d4e', type: 'cache_hit', kernelName: 'Convolution', timestamp: 2003 },
    { address: '0x7f8a4c3b3d5e', type: 'similarity_hit', similarityScore: 0.91, kernelName: 'Convolution', timestamp: 2004 },
    
    // FFT kernel events
    { address: '0x7f8a4c3b4d1e', type: 'cache_miss', kernelName: 'FFT', timestamp: 3000 },
    { address: '0x7f8a4c3b4d2e', type: 'cache_miss', kernelName: 'FFT', timestamp: 3001 },
    { address: '0x7f8a4c3b4d3e', type: 'similarity_hit', similarityScore: 0.76, kernelName: 'FFT', timestamp: 3002 },
    { address: '0x7f8a4c3b4d4e', type: 'cache_hit', kernelName: 'FFT', timestamp: 3003 },
  ];
  
  const runAnalysis = () => {
    setIsAnalyzing(true);
    
    // Simulate analysis process
    setTimeout(() => {
      // Process the sample data to generate analysis results
      const kernelMap = new Map<string, {
        cacheHits: number,
        cacheMisses: number,
        similarityHits: number,
        similarityScores: number[],
      }>();
      
      // Count events by kernel
      sampleAccessEvents.forEach(event => {
        if (!kernelMap.has(event.kernelName)) {
          kernelMap.set(event.kernelName, {
            cacheHits: 0,
            cacheMisses: 0,
            similarityHits: 0,
            similarityScores: [],
          });
        }
        
        const kernelStats = kernelMap.get(event.kernelName)!;
        
        if (event.type === 'cache_hit') {
          kernelStats.cacheHits++;
        } else if (event.type === 'cache_miss') {
          kernelStats.cacheMisses++;
        } else if (event.type === 'similarity_hit') {
          kernelStats.similarityHits++;
          if (event.similarityScore) {
            kernelStats.similarityScores.push(event.similarityScore);
          }
        }
      });
      
      // Convert to results array
      const results: SimilarityAnalysisResult[] = [];
      kernelMap.forEach((stats, kernelName) => {
        const totalAccesses = stats.cacheHits + stats.cacheMisses + stats.similarityHits;
        const avgSimilarityScore = stats.similarityScores.length > 0 
          ? stats.similarityScores.reduce((sum, score) => sum + score, 0) / stats.similarityScores.length
          : 0;
        
        // Calculate estimated energy savings based on similarity hits
        // Assuming cache hits save 100% energy compared to misses, and similarity hits save proportional to their score
        const energySavings = (stats.cacheHits + stats.similarityHits * avgSimilarityScore) / totalAccesses * 100;
        
        results.push({
          kernelName,
          cacheHits: stats.cacheHits,
          cacheMisses: stats.cacheMisses,
          similarityHits: stats.similarityHits,
          energySavings: parseFloat(energySavings.toFixed(2)),
          averageSimilarityScore: parseFloat(avgSimilarityScore.toFixed(2)),
        });
      });
      
      setAnalysisResults(results);
      setIsAnalyzing(false);
    }, 2000);
  };
  
  const getKernelDetails = (kernelName: string) => {
    return analysisResults.find(result => result.kernelName === kernelName);
  };
  
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28'];
  
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center mb-6">
        <Database className="h-6 w-6 text-blue-600 mr-2" />
        <h2 className="text-2xl font-bold">Similarity Hit Analyzer</h2>
      </div>
      
      <div className="mb-8 bg-blue-50 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-2 text-blue-800">About Similarity Hits</h3>
        <p className="text-gray-700 mb-3">
          Similarity hits occur when a memory access doesn't match exactly with cached data (cache miss) 
          but has a high similarity with existing cached data. By leveraging approximate computing principles, 
          we can use similar data instead of fetching from main memory, saving energy while maintaining acceptable accuracy.
        </p>
        <div className="flex items-center">
          <AlertTriangle className="h-5 w-5 text-amber-500 mr-2" />
          <p className="text-sm text-amber-700">
            Similarity hits are particularly effective for applications that can tolerate approximate results, 
            such as machine learning, image processing, and scientific simulations.
          </p>
        </div>
      </div>
      
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Similarity Threshold: {threshold}
        </label>
        <input
          type="range"
          min="0.5"
          max="0.95"
          step="0.05"
          value={threshold}
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>0.5 (More hits, less accuracy)</span>
          <span>0.95 (Fewer hits, higher accuracy)</span>
        </div>
      </div>
      
      <div className="mb-6">
        <button 
          onClick={runAnalysis}
          disabled={isAnalyzing}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:bg-blue-300"
        >
          {isAnalyzing ? 'Analyzing...' : 'Run Similarity Analysis'}
        </button>
      </div>
      
      {analysisResults.length > 0 && (
        <>
          <div className="mb-8">
            <h3 className="text-xl font-semibold mb-4">Memory Access Distribution</h3>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={analysisResults}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="kernelName" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="cacheHits" name="Cache Hits" fill="#22c55e" />
                <Bar dataKey="similarityHits" name="Similarity Hits" fill="#3b82f6" />
                <Bar dataKey="cacheMisses" name="Cache Misses" fill="#ef4444" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          <div className="mb-8">
            <h3 className="text-xl font-semibold mb-4">Energy Savings Analysis</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={analysisResults}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="kernelName" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="energySavings" name="Energy Savings (%)" fill="#10b981" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              
              <div>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={analysisResults}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="kernelName" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="averageSimilarityScore" name="Avg. Similarity Score" fill="#8b5cf6" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            {analysisResults.map(result => (
              <div 
                key={result.kernelName}
                className="bg-gray-50 p-4 rounded-lg hover:shadow-md transition-shadow cursor-pointer"
                onClick={() => setSelectedKernel(result.kernelName)}
              >
                <h4 className="font-semibold text-lg mb-2">{result.kernelName}</h4>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>
                    <p className="text-gray-500">Cache Hits:</p>
                    <p className="font-medium">{result.cacheHits}</p>
                  </div>
                  <div>
                    <p className="text-gray-500">Cache Misses:</p>
                    <p className="font-medium">{result.cacheMisses}</p>
                  </div>
                  <div>
                    <p className="text-gray-500">Similarity Hits:</p>
                    <p className="font-medium">{result.similarityHits}</p>
                  </div>
                  <div>
                    <p className="text-gray-500">Energy Savings:</p>
                    <p className="font-medium text-green-600">{result.energySavings}%</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          {selectedKernel && (
            <div className="bg-blue-50 p-4 rounded-lg mb-4">
              <h3 className="text-lg font-semibold mb-3">Optimization Recommendations for {selectedKernel}</h3>
              <ul className="list-disc pl-6 space-y-2">
                <li>Implement approximate computing techniques to leverage similarity hits</li>
                <li>Adjust memory layout to increase spatial locality and similarity patterns</li>
                <li>Consider using specialized data structures that promote similarity between adjacent elements</li>
                <li>Implement custom caching policies that consider data similarity</li>
              </ul>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default SimilarityHitAnalyzer;