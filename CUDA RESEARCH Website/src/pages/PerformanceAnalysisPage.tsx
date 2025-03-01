import React from 'react';
import { Link } from 'react-router-dom';
import { Zap, BarChart3, TrendingUp, Clock } from 'lucide-react';
import PerformanceChart from '../components/PerformanceChart';
import SimilarityHitAnalyzer from '../components/SimilarityHitAnalyzer';
import { energyEfficiencyData, executionTimeData, memoryUsageData, throughputData } from '../data/performanceData';

// Only keep the timeline data since it's not imported
const timelineData = [
  { name: 'Jan', baseline: 100, optimized: 95 },
  { name: 'Feb', baseline: 100, optimized: 90 },
  { name: 'Mar', baseline: 100, optimized: 82 },
  { name: 'Apr', baseline: 100, optimized: 75 },
  { name: 'May', baseline: 100, optimized: 68 },
  { name: 'Jun', baseline: 100, optimized: 60 }
];

const PerformanceAnalysisPage: React.FC = () => {
  return (
    <div className="min-h-screen">
      {/* Header Section */}
      <section className="bg-gradient-to-r from-blue-900 to-indigo-800 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-4xl font-bold mb-6">Performance Analysis</h1>
          <p className="text-xl max-w-3xl">
            Comprehensive analysis of CUDA optimization techniques and their impact on performance, energy efficiency, and resource utilization.
          </p>
        </div>
      </section>

      {/* Energy Efficiency Tracking */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center mb-12">
            <Zap className="h-8 w-8 text-yellow-500 mr-3" />
            <h2 className="text-3xl font-bold text-gray-800">Energy Efficiency Tracking</h2>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
            <PerformanceChart 
              data={energyEfficiencyData}
              type="bar"
              dataKeys={[
                { key: 'unoptimized', color: '#ef4444', name: 'Unoptimized (Joules)' },
                { key: 'optimized', color: '#22c55e', name: 'Optimized (Joules)' }
              ]}
              title="Energy Consumption Comparison"
              yAxisLabel="Energy (Joules)"
              xAxisLabel="Algorithm"
            />
            
            <PerformanceChart 
              data={timelineData}
              type="line"
              dataKeys={[
                { key: 'baseline', color: '#ef4444', name: 'Baseline Energy Usage (%)' },
                { key: 'optimized', color: '#22c55e', name: 'Optimized Energy Usage (%)' }
              ]}
              title="Energy Efficiency Timeline (2025)"
              yAxisLabel="Relative Energy Usage (%)"
              xAxisLabel="Month"
            />
          </div>
          
          <div className="bg-white rounded-lg shadow-md p-6 mb-8">
            <h3 className="text-xl font-bold mb-4">Energy Efficiency Insights</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-green-50 p-4 rounded-md">
                <div className="flex items-center mb-2">
                  <Zap className="h-5 w-5 text-green-600 mr-2" />
                  <h4 className="font-semibold text-green-800">Average Energy Reduction</h4>
                </div>
                <p className="text-4xl font-bold text-green-700 mb-2">64.8%</p>
                <p className="text-gray-600 text-sm">Average energy savings across all benchmarked algorithms</p>
              </div>
              
              <div className="bg-blue-50 p-4 rounded-md">
                <div className="flex items-center mb-2">
                  <TrendingUp className="h-5 w-5 text-blue-600 mr-2" />
                  <h4 className="font-semibold text-blue-800">Performance per Watt</h4>
                </div>
                <p className="text-4xl font-bold text-blue-700 mb-2">2.8x</p>
                <p className="text-gray-600 text-sm">Average improvement in performance per watt</p>
              </div>
              
              <div className="bg-purple-50 p-4 rounded-md">
                <div className="flex items-center mb-2">
                  <Clock className="h-5 w-5 text-purple-600 mr-2" />
                  <h4 className="font-semibold text-purple-800">Idle Time Reduction</h4>
                </div>
                <p className="text-4xl font-bold text-purple-700 mb-2">42.3%</p>
                <p className="text-gray-600 text-sm">Reduction in GPU idle time with optimized code</p>
              </div>
            </div>
          </div>
          
          <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded-md">
            <h3 className="text-lg font-semibold text-yellow-800 mb-2">CPPJoules Integration</h3>
            <p className="text-gray-700">
              Energy efficiency data is collected using CPPJoules, a C++ library for energy profiling. It provides accurate measurements of energy consumption at the function, kernel, and application levels, enabling precise optimization of CUDA code for energy efficiency.
            </p>
          </div>
        </div>
      </section>

      {/* Performance Graphs & Charts */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center mb-12">
            <BarChart3 className="h-8 w-8 text-blue-500 mr-3" />
            <h2 className="text-3xl font-bold text-gray-800">Performance Metrics</h2>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <PerformanceChart 
              data={executionTimeData}
              type="bar"
              dataKeys={[
                { key: 'unoptimized', color: '#ef4444', name: 'Unoptimized (ms)' },
                { key: 'optimized', color: '#22c55e', name: 'Optimized (ms)' }
              ]}
              title="Execution Time Comparison"
              yAxisLabel="Time (ms)"
              xAxisLabel="Algorithm"
            />
            
            <PerformanceChart 
              data={throughputData}
              type="bar"
              dataKeys={[
                { key: 'unoptimized', color: '#ef4444', name: 'Unoptimized (GFLOPS)' },
                { key: 'optimized', color: '#22c55e', name: 'Optimized (GFLOPS)' }
              ]}
              title="Computational Throughput"
              yAxisLabel="Throughput (GFLOPS)"
              xAxisLabel="Algorithm"
            />
            
            <PerformanceChart 
              data={memoryUsageData}
              type="bar"
              dataKeys={[
                { key: 'unoptimized', color: '#ef4444', name: 'Unoptimized (MB)' },
                { key: 'optimized', color: '#22c55e', name: 'Optimized (MB)' }
              ]}
              title="Memory Usage Comparison"
              yAxisLabel="Memory (MB)"
              xAxisLabel="Algorithm"
            />
            
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-xl font-bold mb-4">Performance Improvement Summary</h3>
              
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Metric</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Average Improvement</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Best Case</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Worst Case</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Execution Time</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">3.2x faster</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">4.5x faster (Stencil)</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">2.1x faster (FFT)</td>
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Energy Consumption</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">3.8x reduction</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">4.4x reduction (Matrix Mult.)</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">3.0x reduction (Particle Sim.)</td>
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Memory Usage</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">18% reduction</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">25% reduction (Matrix Mult.)</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">12% reduction (Particle Sim.)</td>
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Throughput</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">3.5x increase</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">3.9x increase (Stencil)</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">2.7x increase (Graph BFS)</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Detailed Performance Analysis */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold mb-12 text-center text-gray-800">
            Detailed Performance Analysis
          </h2>
          
          <div className="bg-white rounded-lg shadow-md overflow-hidden mb-12">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Algorithm</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Dataset</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Execution Time (ms)</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Energy (J)</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Memory (MB)</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Throughput (GFLOPS)</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Techniques</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {performanceDatasets.map((dataset) => (
                    <React.Fragment key={dataset.id}>
                      <tr className="bg-gray-50">
                        <td rowSpan={2} className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {dataset.algorithm}
                        </td>
                        <td rowSpan={2} className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {dataset.dataset}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <span className="font-semibold text-red-600">{dataset.metrics.executionTime.unoptimized}</span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <span className="font-semibold text-red-600">{dataset.metrics.energyConsumption.unoptimized}</span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <span className="font-semibold text-red-600">{dataset.metrics.memoryUsage.unoptimized}</span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <span className="font-semibold text-red-600">{dataset.metrics.throughput.unoptimized}</span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                            Unoptimized
                          </span>
                        </td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <span className="font-semibold text-green-600">{dataset.metrics.executionTime.optimized}</span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <span className="font-semibold text-green-600">{dataset.metrics.energyConsumption.optimized}</span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <span className="font-semibold text-green-600">{dataset.metrics.memoryUsage.optimized}</span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <span className="font-semibold text-green-600">{dataset.metrics.throughput.optimized}</span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <div className="flex flex-wrap gap-1">
                            {dataset.optimizationTechniques.map((technique, index) => (
                              <span 
                                key={index} 
                                className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800"
                              >
                                {technique}
                              </span>
                            ))}
                          </div>
                        </td>
                      </tr>
                    </React.Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          
          <div className="bg-blue-50 border-l-4 border-blue-500 p-6 rounded-md">
            <h3 className="text-xl font-semibold text-blue-800 mb-4">Key Performance Insights</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-semibold text-gray-800 mb-2">Memory Optimization Impact</h4>
                <ul className="list-disc pl-6 text-gray-600 space-y-1">
                  <li>Shared memory utilization reduces global memory access by up to 78%</li>
                  <li>Memory coalescing improves memory throughput by 2.5-3.5x</li>
                  <li>Texture memory for read-only data improves cache hit rate by 45%</li>
                  <li>Proper memory alignment reduces transaction overhead by 15-25%</li>
                </ul>
              </div>
              
              <div>
                <h4 className="font-semibold text-gray-800 mb-2">Execution Efficiency</h4>
                <ul className="list-disc pl-6 text-gray-600 space-y-1">
                  <li>Warp divergence minimization improves execution efficiency by 25-35%</li>
                  <li>Dynamic parallelism reduces kernel launch overhead by 40-60%</li>
                  <li>Register usage optimization increases occupancy by 15-30%</li>
                  <li>Loop unrolling and instruction-level parallelism boost throughput by 20-40%</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="py-16 bg-gradient-to-r from-indigo-800 to-purple-800 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold mb-6">Apply These Optimizations to Your Projects</h2>
          <p className="text-xl mb-8 max-w-3xl mx-auto">
            Ready to achieve similar performance improvements and energy savings in your CUDA applications? Explore our optimization toolkit and research methods.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <Link 
              to="/optimization" 
              className="px-6 py-3 bg-white text-indigo-800 font-semibold rounded-md hover:bg-blue-50 transition-colors shadow-md"
            >
              Optimization Toolkit
            </Link>
            <Link 
              to="/research" 
              className="px-6 py-3 bg-transparent border-2 border-white text-white font-semibold rounded-md hover:bg-white/10 transition-colors"
            >
              Research Methods
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
};

export default PerformanceAnalysisPage;