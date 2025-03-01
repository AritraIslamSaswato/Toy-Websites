import React, { useState } from 'react';
import { Link, NavLink } from 'react-router-dom'; // Add NavLink import
import { Cpu } from 'lucide-react';

const Navbar: React.FC = () => {
  return (
    <nav className="bg-blue-900 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="flex items-center">
            <Cpu className="h-8 w-8 mr-2" />
            <span className="font-bold text-xl">CUDA Research</span>
          </Link>
          <div className="flex space-x-4">
            <NavLink to="/" className={({isActive}) => 
              `px-3 py-2 rounded-md ${isActive ? 'bg-blue-700' : 'hover:bg-blue-800'}`
            }>
              Home
            </NavLink>
            <NavLink to="/research" className={({isActive}) => 
              `px-3 py-2 rounded-md ${isActive ? 'bg-blue-700' : 'hover:bg-blue-800'}`
            }>
              Research
            </NavLink>
            <NavLink to="/optimization" className={({isActive}) => 
              `px-3 py-2 rounded-md ${isActive ? 'bg-blue-700' : 'hover:bg-blue-800'}`
            }>
              Optimization
            </NavLink>
            <NavLink to="/performance-analysis" className={({isActive}) => 
              `px-3 py-2 rounded-md ${isActive ? 'bg-blue-700' : 'hover:bg-blue-800'}`
            }>
              Performance
            </NavLink>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;