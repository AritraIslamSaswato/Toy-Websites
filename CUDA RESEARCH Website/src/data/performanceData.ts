export interface PerformanceData {
  id: number;
  algorithm: string;
  dataset: string;
  metrics: {
    executionTime: {
      unoptimized: number;
      optimized: number;
    };
    energyConsumption: {
      unoptimized: number;
      optimized: number;
    };
    memoryUsage: {
      unoptimized: number;
      optimized: number;
    };
    throughput: {
      unoptimized: number;
      optimized: number;
    };
  };
  optimizationTechniques: string[];
}

export const performanceDatasets: PerformanceData[] = [
  {
    id: 1,
    algorithm: "Matrix Multiplication",
    dataset: "4096x4096 Single Precision",
    metrics: {
      executionTime: {
        unoptimized: 245.3,
        optimized: 68.7
      },
      energyConsumption: {
        unoptimized: 187.6,
        optimized: 42.3
      },
      memoryUsage: {
        unoptimized: 512,
        optimized: 384
      },
      throughput: {
        unoptimized: 138.2,
        optimized: 493.5
      }
    },
    optimizationTechniques: ["Shared Memory Tiling", "Memory Coalescing", "Loop Unrolling"]
  },
  {
    id: 2,
    algorithm: "FFT",
    dataset: "16M Complex Points",
    metrics: {
      executionTime: {
        unoptimized: 128.4,
        optimized: 42.1
      },
      energyConsumption: {
        unoptimized: 98.3,
        optimized: 29.7
      },
      memoryUsage: {
        unoptimized: 768,
        optimized: 640
      },
      throughput: {
        unoptimized: 124.6,
        optimized: 380.0
      }
    },
    optimizationTechniques: ["Shared Memory Usage", "Bank Conflict Avoidance", "Batch Processing"]
  },
  {
    id: 3,
    algorithm: "Graph BFS",
    dataset: "RMAT Scale 22",
    metrics: {
      executionTime: {
        unoptimized: 187.2,
        optimized: 73.5
      },
      energyConsumption: {
        unoptimized: 143.8,
        optimized: 48.2
      },
      memoryUsage: {
        unoptimized: 1024,
        optimized: 896
      },
      throughput: {
        unoptimized: 53.4,
        optimized: 136.1
      }
    },
    optimizationTechniques: ["Work Efficient Parallelism", "Warp Aggregation", "Dynamic Load Balancing"]
  },
  {
    id: 4,
    algorithm: "Stencil Computation",
    dataset: "3D Grid 512³",
    metrics: {
      executionTime: {
        unoptimized: 312.7,
        optimized: 89.4
      },
      energyConsumption: {
        unoptimized: 239.5,
        optimized: 56.8
      },
      memoryUsage: {
        unoptimized: 1536,
        optimized: 1280
      },
      throughput: {
        unoptimized: 83.8,
        optimized: 292.6
      }
    },
    optimizationTechniques: ["Temporal Blocking", "Register Blocking", "Shared Memory Caching"]
  },
  {
    id: 5,
    algorithm: "Particle Simulation",
    dataset: "10M Particles",
    metrics: {
      executionTime: {
        unoptimized: 423.6,
        optimized: 156.2
      },
      energyConsumption: {
        unoptimized: 324.7,
        optimized: 103.5
      },
      memoryUsage: {
        unoptimized: 2048,
        optimized: 1792
      },
      throughput: {
        unoptimized: 23.6,
        optimized: 64.0
      }
    },
    optimizationTechniques: ["Spatial Partitioning", "Warp Shuffle", "Atomic Operation Reduction"]
  }
];

export const energyEfficiencyData = [
  { name: 'Matrix Mult.', unoptimized: 187.6, optimized: 42.3 },
  { name: 'FFT', unoptimized: 98.3, optimized: 29.7 },
  { name: 'Graph BFS', unoptimized: 143.8, optimized: 48.2 },
  { name: 'Stencil', unoptimized: 239.5, optimized: 56.8 },
  { name: 'Particle Sim.', unoptimized: 324.7, optimized: 103.5 }
];

export const executionTimeData = [
  { name: 'Matrix Mult.', unoptimized: 245.3, optimized: 68.7 },
  { name: 'FFT', unoptimized: 128.4, optimized: 42.1 },
  { name: 'Graph BFS', unoptimized: 187.2, optimized: 73.5 },
  { name: 'Stencil', unoptimized: 312.7, optimized: 89.4 },
  { name: 'Particle Sim.', unoptimized: 423.6, optimized: 156.2 }
];

export const memoryUsageData = [
  { name: 'Matrix Mult.', unoptimized: 512, optimized: 384 },
  { name: 'FFT', unoptimized: 768, optimized: 640 },
  { name: 'Graph BFS', unoptimized: 256, optimized: 192 },
  { name: 'Stencil', unoptimized: 1024, optimized: 768 },
  { name: 'Particle Sim.', unoptimized: 2048, optimized: 1536 }
];

export const throughputData = [
  { name: 'Matrix Mult.', unoptimized: 1200, optimized: 4800 },
  { name: 'FFT', unoptimized: 800, optimized: 2400 },
  { name: 'Graph BFS', unoptimized: 600, optimized: 1800 },
  { name: 'Stencil', unoptimized: 1500, optimized: 6000 },
  { name: 'Particle Sim.', unoptimized: 900, optimized: 3600 }
];

export const timelineData = [
  { name: 'Jan', baseline: 100, optimized: 95 },
  { name: 'Feb', baseline: 100, optimized: 90 },
  { name: 'Mar', baseline: 100, optimized: 82 },
  { name: 'Apr', baseline: 100, optimized: 75 },
  { name: 'May', baseline: 100, optimized: 68 },
  { name: 'Jun', baseline: 100, optimized: 60 }
];