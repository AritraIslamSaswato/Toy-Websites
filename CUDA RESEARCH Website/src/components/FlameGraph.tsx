import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface FlameGraphNode {
  name: string;
  value: number;
  children?: FlameGraphNode[];
  color?: string;
}

interface FlameGraphProps {
  data: FlameGraphNode;
  width?: number;
  height?: number;
}

const FlameGraph: React.FC<FlameGraphProps> = ({ data, width = 960, height = 500 }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  
  useEffect(() => {
    if (!svgRef.current || !data) return;
    
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    
    const margin = { top: 10, right: 10, bottom: 10, left: 10 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);
    
    // Add error handling for data validation
    try {
      const root = d3.hierarchy(data)
        .sum(d => d.value || 0) // Handle potential undefined values
        .sort((a, b) => (b.value || 0) - (a.value || 0));
      
      const partition = d3.partition()
        .size([innerWidth, innerHeight])
        .padding(1)
        .round(true);
      
      partition(root);
      
      // Verify partition was successful
      if (!root.x0 && !root.x1 && !root.y0 && !root.y1) {
        console.error('Partition layout failed to compute coordinates');
        return;
      }
      
      const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
      
      // Use a more efficient selection pattern
      const cell = g.selectAll("g.cell")
        .data(root.descendants())
        .join("g")
        .attr("class", "cell")
        .attr("transform", d => `translate(${d.x0},${d.y0})`);
      
      // Add event listeners more efficiently
      cell.append("rect")
        .attr("width", d => Math.max(0, d.x1 - d.x0)) // Prevent negative width
        .attr("height", d => Math.max(0, d.y1 - d.y0)) // Prevent negative height
        .attr("fill", d => d.data.color || colorScale(d.depth.toString()))
        .attr("opacity", 0.8)
        .on("mouseover", function() {
          d3.select(this).attr("opacity", 1);
        })
        .on("mouseout", function() {
          d3.select(this).attr("opacity", 0.8);
        });
      
      // Only add text if there's enough space
      const text = cell.append("text")
        .attr("x", 4)
        .attr("y", 13)
        .attr("fill", "white")
        .attr("pointer-events", "none"); // Prevent text from interfering with mouse events
      
      text.append("tspan")
        .text(d => {
          const width = d.x1 - d.x0;
          return width > 30 ? d.data.name : '';
        });
      
      text.append("tspan")
        .attr("x", 4)
        .attr("dy", "1.2em")
        .text(d => {
          const width = d.x1 - d.x0;
          return width > 30 ? `${(d.value || 0).toFixed(2)}ms` : '';
        });
      
      cell.append("title")
        .text(d => `${d.ancestors().map(d => d.data.name).reverse().join("/")}\n${(d.value || 0).toFixed(2)}ms`);
    } catch (error) {
      console.error('Error rendering flame graph:', error);
      g.append("text")
        .attr("x", innerWidth / 2)
        .attr("y", innerHeight / 2)
        .attr("text-anchor", "middle")
        .attr("fill", "red")
        .text("Error rendering flame graph");
    }
  }, [data, width, height]);
  
  return (
    <svg ref={svgRef} width={width} height={height}></svg>
  );
};

export default FlameGraph;