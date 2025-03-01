import React from 'react';
import { Link } from 'react-router-dom';
import { Cpu, Zap, BarChart3, BookOpen, ChevronRight, Code, Database, Layers } from 'lucide-react';
import ArticleCard from '../components/ArticleCard';
import CudaVisualization from '../components/CudaVisualization';
import { articles } from '../data/articles';

const HomePage: React.FC = () => {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="bg-gradient-to-r from-blue-900 via-indigo-800 to-purple-900 text-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
            <div>
              <h1 className="text-4xl md:text-5xl font-extrabold mb-6 leading-tight">
                Energy-Efficient CUDA Programming Research
              </h1>
              <p className="text-xl mb-8 text-blue-100">
                Advancing the state of the art in GPU computing optimization techniques and energy-efficient programming practices.
              </p>
              <div className="flex flex-wrap gap-4">
                <Link 
                  to="/research" 
                  className="px-6 py-3 bg-white text-blue-900 font-semibold rounded-md hover:bg-blue-50 transition-colors shadow-md"
                >
                  Explore Research
                </Link>
                <Link 
                  to="/optimization" 
                  className="px-6 py-3 bg-transparent border-2 border-white text-white font-semibold rounded-md hover:bg-white/10 transition-colors"
                >
                  Optimization Toolkit
                </Link>
              </div>
            </div>
            <div className="hidden md:block">
              <img 
                src="https://images.unsplash.com/photo-1639322537504-6427a16b0a28?ixlib=rb-1.2.1&auto=format&fit=crop&w=1200&q=80" 
                alt="GPU Computing" 
                className="rounded-lg shadow-2xl"
              />
            </div>
          </div>
        </div>
      </section>

      {/* Enhanced Key Features Section */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center mb-4 text-gray-800">
            Advancing CUDA Optimization Research
          </h2>
          <p className="text-lg text-center text-gray-600 mb-12 max-w-3xl mx-auto">
            Our research focuses on cutting-edge techniques to maximize GPU performance while minimizing energy consumption through innovative programming approaches.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-16">
            {/* UVM Research Area - Enhanced Card */}
            <div className="bg-white rounded-xl shadow-lg overflow-hidden hover:shadow-xl transition-shadow duration-300">
              <div className="bg-gradient-to-r from-blue-500 to-blue-700 h-3"></div>
              <div className="p-6">
                <div className="flex items-start mb-4">
                  <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mr-4 flex-shrink-0">
                    <Cpu className="h-6 w-6 text-blue-600" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold mb-2 text-gray-800">Unified Virtual Memory Optimization</h3>
                    <p className="text-gray-600 mb-4">
                      Our research explores advanced UVM techniques that significantly reduce page faults and optimize memory transfers between CPU and GPU.
                    </p>
                  </div>
                </div>
                
                <div className="bg-blue-50 p-4 rounded-lg mb-4">
                  <h4 className="font-semibold text-blue-800 mb-2">Key Findings</h4>
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-start">
                      <ChevronRight className="h-4 w-4 text-blue-500 mt-0.5 mr-1 flex-shrink-0" />
                      <span>Reduced page faults by 78% using prefetching and memory advising techniques</span>
                    </li>
                    <li className="flex items-start">
                      <ChevronRight className="h-4 w-4 text-blue-500 mt-0.5 mr-1 flex-shrink-0" />
                      <span>Developed adaptive memory migration policies based on access patterns</span>
                    </li>
                    <li className="flex items-start">
                      <ChevronRight className="h-4 w-4 text-blue-500 mt-0.5 mr-1 flex-shrink-0" />
                      <span>Implemented concurrent data transfers with computation overlap</span>
                    </li>
                  </ul>
                </div>
                
                <Link to="/research/uvm" className="inline-flex items-center text-blue-600 hover:text-blue-800 font-medium">
                  Explore UVM Research
                  <ChevronRight className="h-4 w-4 ml-1" />
                </Link>
              </div>
            </div>
            
            {/* Energy Efficiency Research Area - Enhanced Card */}
            <div className="bg-white rounded-xl shadow-lg overflow-hidden hover:shadow-xl transition-shadow duration-300">
              <div className="bg-gradient-to-r from-green-500 to-green-700 h-3"></div>
              <div className="p-6">
                <div className="flex items-start mb-4">
                  <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mr-4 flex-shrink-0">
                    <Zap className="h-6 w-6 text-green-600" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold mb-2 text-gray-800">Energy-Efficient Computing</h3>
                    <p className="text-gray-600 mb-4">
                      We develop innovative techniques to reduce power consumption while maintaining high computational throughput on CUDA GPUs.
                    </p>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-3 mb-4">
                  <div className="bg-green-50 p-3 rounded-lg text-center">
                    <p className="text-3xl font-bold text-green-700">42%</p>
                    <p className="text-xs text-green-800">Average Energy Reduction</p>
                  </div>
                  <div className="bg-green-50 p-3 rounded-lg text-center">
                    <p className="text-3xl font-bold text-green-700">2.8x</p>
                    <p className="text-xs text-green-800">Performance per Watt</p>
                  </div>
                </div>
                
                <div className="bg-green-50 p-4 rounded-lg mb-4">
                  <h4 className="font-semibold text-green-800 mb-2">Techniques</h4>
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-start">
                      <ChevronRight className="h-4 w-4 text-green-500 mt-0.5 mr-1 flex-shrink-0" />
                      <span>Dynamic voltage and frequency scaling based on workload characteristics</span>
                    </li>
                    <li className="flex items-start">
                      <ChevronRight className="h-4 w-4 text-green-500 mt-0.5 mr-1 flex-shrink-0" />
                      <span>Kernel fusion to reduce launch overhead and memory transfers</span>
                    </li>
                  </ul>
                </div>
                
                <Link to="/performance" className="inline-flex items-center text-green-600 hover:text-green-800 font-medium">
                  View Energy Efficiency Data
                  <ChevronRight className="h-4 w-4 ml-1" />
                </Link>
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {/* Performance Analysis - Enhanced Card */}
            <div className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow border-t-4 border-purple-500">
              <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mb-4">
                <BarChart3 className="h-6 w-6 text-purple-600" />
              </div>
              <h3 className="text-xl font-semibold mb-2 text-gray-800">Performance Analysis</h3>
              <p className="text-gray-600 mb-4">
                Advanced profiling and analysis tools to identify bottlenecks and optimization opportunities.
              </p>
              <div className="flex flex-wrap gap-2 mb-4">
                <span className="px-2 py-1 bg-purple-100 text-purple-800 text-xs rounded-full">Flame Graphs</span>
                <span className="px-2 py-1 bg-purple-100 text-purple-800 text-xs rounded-full">Memory Profiling</span>
                <span className="px-2 py-1 bg-purple-100 text-purple-800 text-xs rounded-full">Kernel Analysis</span>
              </div>
              <Link to="/performance" className="text-purple-600 hover:text-purple-800 font-medium inline-flex items-center">
                Explore Tools
                <ChevronRight className="h-4 w-4 ml-1" />
              </Link>
            </div>
            
            {/* PTX Optimization - Enhanced Card */}
            <div className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow border-t-4 border-orange-500">
              <div className="w-12 h-12 bg-orange-100 rounded-full flex items-center justify-center mb-4">
                <Code className="h-6 w-6 text-orange-600" />
              </div>
              <h3 className="text-xl font-semibold mb-2 text-gray-800">PTX Optimization</h3>
              <p className="text-gray-600 mb-4">
                Low-level PTX optimizations for maximum efficiency and performance in compute-intensive applications.
              </p>
              <div className="bg-gray-100 p-3 rounded text-xs font-mono mb-4 overflow-x-auto">
                <code className="text-gray-800">
                  mul.wide.u32 %rd3, %r2, 4;<br/>
                  add.u64 %rd4, %rd1, %rd3;<br/>
                  ld.global.f32 %f1, [%rd4];
                </code>
              </div>
              <Link to="/research/ptx" className="text-orange-600 hover:text-orange-800 font-medium inline-flex items-center">
                Learn More
                <ChevronRight className="h-4 w-4 ml-1" />
              </Link>
            </div>
            
            {/* Memory Optimization - New Card */}
            <div className="bg-white p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow border-t-4 border-blue-500">
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mb-4">
                <Database className="h-6 w-6 text-blue-600" />
              </div>
              <h3 className="text-xl font-semibold mb-2 text-gray-800">Memory Access Patterns</h3>
              <p className="text-gray-600 mb-4">
                Optimizing memory access patterns for coalescing and minimizing bank conflicts in shared memory.
              </p>
              <div className="grid grid-cols-4 gap-1 mb-4">
                {[...Array(16)].map((_, i) => (
                  <div 
                    key={i} 
                    className={`h-6 rounded ${i % 4 === 0 ? 'bg-blue-500' : 'bg-blue-200'}`}
                    title="Memory access visualization"
                  ></div>
                ))}
              </div>
              <Link to="/optimization" className="text-blue-600 hover:text-blue-800 font-medium inline-flex items-center">
                View Techniques
                <ChevronRight className="h-4 w-4 ml-1" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Recent Research Articles */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold mb-12 text-center text-gray-800">
            Recent Research Articles
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {articles.slice(0, 3).map(article => (
              <ArticleCard key={article.id} article={article} />
            ))}
          </div>
          
          <div className="mt-12 text-center">
            <Link 
              to="/research" 
              className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-md hover:bg-blue-700 transition-colors shadow-md"
            >
              View All Research
            </Link>
          </div>
        </div>
      </section>

      {/* Interactive Visualization */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold mb-12 text-center text-gray-800">
            CUDA Architecture Visualization
          </h2>
          
          <div className="flex justify-center">
            <CudaVisualization width={800} height={500} />
          </div>
          
          <div className="mt-12 max-w-3xl mx-auto text-center">
            <p className="text-gray-600 mb-6">
              This interactive visualization demonstrates the CUDA GPU architecture, showing streaming multiprocessors (SMs) and memory transfers. The animation illustrates how data moves between SMs and global memory during kernel execution.
            </p>
            <Link 
              to="/optimization" 
              className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-md hover:bg-blue-700 transition-colors shadow-md"
            >
              Explore Optimization Tools
            </Link>
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="py-16 bg-gradient-to-r from-indigo-800 to-purple-800 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold mb-6">Ready to Optimize Your CUDA Applications?</h2>
          <p className="text-xl mb-8 max-w-3xl mx-auto">
            Explore our research, optimization techniques, and performance analysis tools to make your CUDA applications more energy-efficient and performant.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <Link 
              to="/research" 
              className="px-6 py-3 bg-white text-indigo-800 font-semibold rounded-md hover:bg-blue-50 transition-colors shadow-md"
            >
              Research Methods
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

export default HomePage;