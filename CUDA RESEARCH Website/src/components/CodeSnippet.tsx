import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';

interface CodeSnippetProps {
  code: string;
  language?: string;
  title?: string;
}

const CodeSnippet: React.FC<CodeSnippetProps> = ({ 
  code, 
  language = 'cpp', 
  title = 'CUDA Code Snippet' 
}) => {
  const [copied, setCopied] = useState(false);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="rounded-lg overflow-hidden bg-gray-900 shadow-lg mb-6">
      <div className="flex justify-between items-center px-4 py-2 bg-gray-800 text-gray-200">
        <span className="font-mono text-sm">{title}</span>
        <button
          onClick={copyToClipboard}
          className="p-1 rounded hover:bg-gray-700 transition-colors"
          aria-label="Copy code"
        >
          {copied ? <Check size={18} className="text-green-400" /> : <Copy size={18} />}
        </button>
      </div>
      <pre className="p-4 overflow-x-auto text-gray-300 font-mono text-sm">
        <code>{code}</code>
      </pre>
    </div>
  );
};

export default CodeSnippet;