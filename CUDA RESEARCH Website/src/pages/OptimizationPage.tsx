import React from 'react';
import { Link } from 'react-router-dom';
import { Cpu, Zap, Code } from 'lucide-react';
import CodeSnippet from '../components/CodeSnippet';
import { OptimizationTechnique } from '../data/optimizationTechniques';
import Tutorial from '../components/Tutorial';

const OptimizationPage: React.FC = () => {
  return (
    <div className="min-h-screen">
      {/* Header Section */}
      <section className="bg-gradient-to-r from-blue-900 to-indigo-800 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h1 className="text-4xl font-bold mb-6">CUDA Optimization Toolkit</h1>
          <p className="text-xl max-w-3xl">
            Explore our comprehensive collection of CUDA optimization techniques and tools to enhance your GPU applications' performance and energy efficiency.
          </p>
        </div>
      </section>

      {/* Optimization Tools */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold mb-12 text-center text-gray-800">
            Optimization Tools
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="flex items-center mb-4">
                <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center mr-4">
                  <Cpu className="h-5 w-5 text-blue-600" />
                </div>
                <h3 className="text-xl font-bold text-gray-800">Memory Analyzer</h3>
              </div>
              <p className="text-gray-600 mb-4">
                Analyze memory access patterns and identify optimization opportunities in your CUDA kernels.
              </p>
              <Link 
                to="/tools/memory-analyzer"
                className="inline-block px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
              >
                Launch Tool
              </Link>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="flex items-center mb-4">
                <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center mr-4">
                  <Zap className="h-5 w-5 text-green-600" />
                </div>
                <h3 className="text-xl font-bold text-gray-800">Power Profiler</h3>
              </div>
              <p className="text-gray-600 mb-4">
                Profile power consumption and identify energy optimization opportunities in your CUDA applications.
              </p>
              <Link 
                to="/tools/power-profiler"
                className="inline-block px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
              >
                Launch Tool
              </Link>
            </div>

            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="flex items-center mb-4">
                <div className="w-10 h-10 bg-purple-100 rounded-full flex items-center justify-center mr-4">
                  <Code className="h-5 w-5 text-purple-600" />
                </div>
                <h3 className="text-xl font-bold text-gray-800">Code Generator</h3>
              </div>
              <p className="text-gray-600 mb-4">
                Generate optimized CUDA code templates and patterns for common GPU computing scenarios.
              </p>
              <Link 
                to="/tools/code-generator"
                className="inline-block px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
              >
                Launch Tool
              </Link>
            </div>
          </div>
        </div>
      </section>
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold mb-12 text-center text-gray-800">
            Tutorials
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {/* Tutorial Components */}
        <Tutorial 
          level="beginner"
          topic="Memory Access Patterns"
          interactive={true}
          videos={[
                {
                  title: "Introduction to CUDA Memory Patterns",
                  url: "https://www.youtube.com/embed/dQw4w9WgXcQ",
                  source: "youtube",
                  duration: "10:25"
                },
                {
                  title: "Advanced Memory Access Optimization",
                  url: "https://www.youtube.com/embed/dQw4w9WgXcQ",
                  source: "youtube",
                  duration: "15:30"
                }
              ]}
            />
            <Tutorial 
              level="intermediate"
              topic="UVM Optimization"
              interactive={true}
              videos={[
                {
                  title: "Understanding UVM in CUDA",
                  url: "https://www.youtube.com/embed/dQw4w9WgXcQ",
                  source: "youtube",
                  duration: "12:45"
                }
              ]}
            />
            <Tutorial 
              level="advanced"
              topic="PTX Parallelization"
              interactive={true}
              videos={[
                {
                  title: "PTX Optimization Techniques",
                  url: "https://www.youtube.com/embed/dQw4w9WgXcQ",
                  source: "youtube",
                  duration: "20:15"
                }
              ]}
            />
          </div>
        </div>
      </section>
    </div>
  );
};

export default OptimizationPage;