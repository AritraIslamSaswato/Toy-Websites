import React from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  LineChart,
  Line
} from 'recharts';

interface DataPoint {
  name: string;
  [key: string]: string | number; // More specific than 'any'
}

// Add support for flame graphs
// Removed unused import FlameGraph

// Add new visualization options
interface KernelProfileData {
  kernelName: string;
  executionTime: number;
  memoryUsage: number;
  threadOccupancy: number;
}

interface PerformanceChartProps {
  data: DataPoint[];
  type: 'bar' | 'line';
  dataKeys: {
    key: string;
    color: string;
    name?: string;
  }[];
  title: string;
  yAxisLabel?: string;
  xAxisLabel?: string;
  height?: number;
  showFlameGraph?: boolean;
  kernelProfile?: KernelProfileData;
}

const PerformanceChart: React.FC<PerformanceChartProps> = ({
  data,
  type,
  dataKeys,
  title,
  yAxisLabel,
  xAxisLabel,
  height = 400
}) => {
  if (!Array.isArray(dataKeys) || dataKeys.length === 0) {
    console.error('PerformanceChart: dataKeys must be a non-empty array');
    return null;
  }

  if (!Array.isArray(data)) {
    console.error('PerformanceChart: data must be an array');
    return null;
  }

  if (!data || data.length === 0) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-md mb-8 flex items-center justify-center" style={{ height }}>
        <p className="text-gray-500">No data available</p>
      </div>
    );
  }

  // Memoize chart rendering for better performance
  const renderChart = React.useMemo(() => {
    if (type === 'bar') {
      return (
        <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 30 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" label={{ value: xAxisLabel, position: 'insideBottom', offset: -10 }} />
          <YAxis label={{ value: yAxisLabel, angle: -90, position: 'insideLeft' }} />
          <Tooltip 
            formatter={(value: number) => [value.toFixed(2), '']}
            labelFormatter={(label) => `Algorithm: ${label}`}
          />
          <Legend />
          {dataKeys.map((dataKey) => (
            <Bar 
              key={dataKey.key} 
              dataKey={dataKey.key} 
              fill={dataKey.color} 
              name={dataKey.name || dataKey.key} 
            />
          ))}
        </BarChart>
      );
    } else {
      return (
        <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 30 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" label={{ value: xAxisLabel, position: 'insideBottom', offset: -10 }} />
          <YAxis label={{ value: yAxisLabel, angle: -90, position: 'insideLeft' }} />
          <Tooltip 
            formatter={(value: number) => {
              if (typeof value === 'number') {
                return [`${value.toFixed(2)}`, ''];
              }
              return [value, ''];
            }}
            labelFormatter={(label) => `${title}: ${label}`}
            contentStyle={{
              backgroundColor: 'rgba(255, 255, 255, 0.95)',
              border: '1px solid #ccc',
              borderRadius: '4px',
              padding: '8px'
            }}
          />
          <Legend />
          {dataKeys.map((dataKey) => (
            <Line 
              key={dataKey.key} 
              type="monotone" 
              dataKey={dataKey.key} 
              stroke={dataKey.color} 
              name={dataKey.name || dataKey.key} 
              strokeWidth={2}
              activeDot={{ r: 8 }}
            />
          ))}
        </LineChart>
      );
    }
  }, [data, type, dataKeys, xAxisLabel, yAxisLabel, title]); // Added title to dependency array

  return (
    <div 
      className="bg-white p-6 rounded-lg shadow-md mb-8"
      role="region"
      aria-label={`${title} Chart`}
    >
      <h3 className="text-xl font-bold mb-4 text-gray-800" id={`chart-title-${title.toLowerCase().replace(/\s+/g, '-')}`}>
        {title}
      </h3>
      <div 
        className="w-full" 
        style={{ height, minWidth: '300px' }}
        aria-labelledby={`chart-title-${title.toLowerCase().replace(/\s+/g, '-')}`}
      >
        <ResponsiveContainer width="100%" height="100%" minWidth={300}>
          {renderChart()}
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default PerformanceChart;