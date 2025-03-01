import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface CudaVisualizationProps {
  width?: number;
  height?: number;
}

const CudaVisualization: React.FC<CudaVisualizationProps> = ({ 
  width = 800, 
  height = 500 
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  
  useEffect(() => {
    if (!svgRef.current) return;
    
    // Clear previous visualization
    d3.select(svgRef.current).selectAll('*').remove();
    
    const svg = d3.select(svgRef.current);
    
    // Define grid dimensions
    const gridWidth = 16;
    const gridHeight = 8;
    const cellSize = Math.min(width / (gridWidth + 2), height / (gridHeight + 4));
    
    // Create a group for the visualization
    const g = svg.append('g')
      .attr('transform', `translate(${width / 2 - (gridWidth * cellSize) / 2}, ${height / 2 - (gridHeight * cellSize) / 2})`);
    
    // Draw GPU grid
    const grid = g.append('g').attr('class', 'grid');
    
    // Draw streaming multiprocessors (SMs)
    for (let i = 0; i < gridHeight; i++) {
      for (let j = 0; j < gridWidth; j++) {
        const sm = grid.append('rect')
          .attr('x', j * cellSize)
          .attr('y', i * cellSize)
          .attr('width', cellSize - 2)
          .attr('height', cellSize - 2)
          .attr('rx', 3)
          .attr('ry', 3)
          .attr('fill', '#4a5568')
          .attr('stroke', '#2d3748')
          .attr('stroke-width', 1);
          
        // Add animation to simulate processing
        const delay = Math.random() * 2000;
        const duration = 1000 + Math.random() * 2000;
        
        sm.transition()
          .delay(delay)
          .duration(duration)
          .attr('fill', '#4299e1')
          .transition()
          .duration(duration)
          .attr('fill', '#4a5568')
          .on('end', function repeat() {
            d3.select(this)
              .transition()
              .delay(Math.random() * 2000)
              .duration(1000 + Math.random() * 2000)
              .attr('fill', '#4299e1')
              .transition()
              .duration(1000 + Math.random() * 2000)
              .attr('fill', '#4a5568')
              .on('end', repeat);
          });
      }
    }
    
    // Add memory transfers
    const memoryTransfers = g.append('g').attr('class', 'memory-transfers');
    
    // Global memory at the bottom
    memoryTransfers.append('rect')
      .attr('x', 0)
      .attr('y', (gridHeight + 1) * cellSize)
      .attr('width', gridWidth * cellSize)
      .attr('height', cellSize)
      .attr('fill', '#ed8936')
      .attr('stroke', '#dd6b20')
      .attr('stroke-width', 1)
      .attr('rx', 3)
      .attr('ry', 3);
      
    // Add label for global memory
    memoryTransfers.append('text')
      .attr('x', gridWidth * cellSize / 2)
      .attr('y', (gridHeight + 1.5) * cellSize)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('fill', 'white')
      .attr('font-size', cellSize / 3)
      .text('Global Memory');
    
    // Add memory transfer animations
    function createMemoryTransfer() {
      const startCol = Math.floor(Math.random() * gridWidth);
      const startRow = Math.floor(Math.random() * gridHeight);
      
      const dataPacket = memoryTransfers.append('circle')
        .attr('cx', startCol * cellSize + cellSize / 2)
        .attr('cy', startRow * cellSize + cellSize / 2)
        .attr('r', cellSize / 6)
        .attr('fill', '#4299e1')
        .attr('opacity', 0.8);
        
      // Animate to global memory
      dataPacket.transition()
        .duration(1000)
        .attr('cy', (gridHeight + 1) * cellSize + cellSize / 2)
        .transition()
        .duration(500)
        .attr('r', cellSize / 8)
        .attr('fill', '#ed8936')
        .transition()
        .duration(1000)
        .attr('cy', Math.floor(Math.random() * gridHeight) * cellSize + cellSize / 2)
        .attr('cx', Math.floor(Math.random() * gridWidth) * cellSize + cellSize / 2)
        .attr('fill', '#4299e1')
        .transition()
        .duration(300)
        .attr('opacity', 0)
        .remove();
    }
    
    // Start memory transfer animations
    const transferInterval = setInterval(createMemoryTransfer, 800);
    
    // Add labels
    const labels = g.append('g').attr('class', 'labels');
    
    labels.append('text')
      .attr('x', gridWidth * cellSize / 2)
      .attr('y', -cellSize / 2)
      .attr('text-anchor', 'middle')
      .attr('fill', '#2d3748')
      .attr('font-size', cellSize / 2)
      .attr('font-weight', 'bold')
      .text('CUDA GPU Architecture');
      
    // Add legend
    const legend = svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${width - 150}, 20)`);
      
    // SM legend
    legend.append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', 15)
      .attr('height', 15)
      .attr('fill', '#4a5568');
      
    legend.append('text')
      .attr('x', 20)
      .attr('y', 12)
      .attr('fill', '#2d3748')
      .attr('font-size', 12)
      .text('Streaming Multiprocessor');
      
    // Active SM legend
    legend.append('rect')
      .attr('x', 0)
      .attr('y', 25)
      .attr('width', 15)
      .attr('height', 15)
      .attr('fill', '#4299e1');
      
    legend.append('text')
      .attr('x', 20)
      .attr('y', 37)
      .attr('fill', '#2d3748')
      .attr('font-size', 12)
      .text('Active Processing');
      
    // Global memory legend
    legend.append('rect')
      .attr('x', 0)
      .attr('y', 50)
      .attr('width', 15)
      .attr('height', 15)
      .attr('fill', '#ed8936');
      
    legend.append('text')
      .attr('x', 20)
      .attr('y', 62)
      .attr('fill', '#2d3748')
      .attr('font-size', 12)
      .text('Global Memory');
    
    // Cleanup on unmount
    return () => {
      clearInterval(transferInterval);
    };
  }, [width, height]);
  
  return (
    <div className="bg-white rounded-lg shadow-md p-6 overflow-hidden">
      <h3 className="text-xl font-bold mb-4">CUDA GPU Architecture Visualization</h3>
      <div className="flex justify-center">
        <svg 
          ref={svgRef} 
          width={width} 
          height={height} 
          className="bg-gray-50 rounded-lg"
        ></svg>
      </div>
    </div>
  );
};

export default CudaVisualization;