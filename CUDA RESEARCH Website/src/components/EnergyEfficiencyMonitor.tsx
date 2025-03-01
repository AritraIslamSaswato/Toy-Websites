import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Zap, Battery, Settings } from 'lucide-react';

interface PowerReading {
  timestamp: string;
  gpuPower: number;
  cpuPower: number;
  totalPower: number;
  temperature: number;
}

const EnergyEfficiencyMonitor: React.FC = () => {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [samplingRate, setSamplingRate] = useState(1000); // ms
  const [powerThreshold, setPowerThreshold] = useState(150); // watts
  const [powerReadings, setPowerReadings] = useState<PowerReading[]>([]);
  const [intervalId, setIntervalId] = useState<number | null>(null); // Declare intervalId only once
  
  // Sample data - would be replaced with real monitoring
  const generateSampleData = () => {
    const now = new Date();
    const readings: PowerReading[] = [];
    
    for (let i = 0; i < 30; i++) {
      const time = new Date(now.getTime() - (30 - i) * samplingRate);
      readings.push({
        timestamp: time.toLocaleTimeString(),
        gpuPower: 80 + Math.random() * 40,
        cpuPower: 30 + Math.random() * 20,
        totalPower: 110 + Math.random() * 60,
        temperature: 65 + Math.random() * 15
      });
    }
    
    return readings;
  };
  
  const startMonitoring = () => {
    setIsMonitoring(true);
    setPowerReadings(generateSampleData());
    
    const interval = setInterval(() => {
      setPowerReadings(prev => {
        const newReading: PowerReading = {
          timestamp: new Date().toLocaleTimeString(),
          gpuPower: 80 + Math.random() * 40,
          cpuPower: 30 + Math.random() * 20,
          totalPower: 110 + Math.random() * 60,
          temperature: 65 + Math.random() * 15
        };
        return [...prev.slice(1), newReading];
      });
    }, samplingRate);
    
    // Store interval ID for cleanup
    setIntervalId(interval);
  };

  const stopMonitoring = () => {
    if (intervalId) {
      clearInterval(intervalId);
    }
    setIsMonitoring(false);
    setIntervalId(null);
  };

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [intervalId]);

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold">Energy Efficiency Monitor</h2>
        <div className="flex items-center space-x-4">
          <button
            onClick={isMonitoring ? stopMonitoring : startMonitoring}
            className={`px-4 py-2 rounded-md ${
              isMonitoring 
                ? 'bg-red-600 hover:bg-red-700' 
                : 'bg-blue-600 hover:bg-blue-700'
            } text-white`}
          >
            {isMonitoring ? 'Stop Monitoring' : 'Start Monitoring'}
          </button>
        </div>
      </div>

      <div className="h-96">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={powerReadings}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="gpuPower" stroke="#8884d8" name="GPU Power (W)" />
            <Line type="monotone" dataKey="cpuPower" stroke="#82ca9d" name="CPU Power (W)" />
            <Line type="monotone" dataKey="totalPower" stroke="#ff7300" name="Total Power (W)" />
            <Line type="monotone" dataKey="temperature" stroke="#ff0000" name="Temperature (°C)" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default EnergyEfficiencyMonitor;