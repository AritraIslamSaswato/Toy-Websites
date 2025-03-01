import React, { useState } from 'react';
import { BookOpen, ChevronRight, Code, Star, Video, ExternalLink } from 'lucide-react';

interface TutorialVideo {
  title: string;
  url: string;
  source: 'youtube' | 'vimeo' | 'custom';
  duration: string;
}

interface TutorialProps {
  level: 'beginner' | 'intermediate' | 'advanced';
  topic: string;
  interactive: boolean;
  videos?: TutorialVideo[];
}

const Tutorial: React.FC<TutorialProps> = ({ level, topic, interactive, videos }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [showVideo, setShowVideo] = useState(false);
  const [selectedVideo, setSelectedVideo] = useState<TutorialVideo | null>(null);

  // Update the getLevelColor function
  const getLevelColor = () => {
    switch (level) {
      case 'beginner': return 'text-green-600 bg-green-50';
      case 'intermediate': return 'text-blue-600 bg-blue-50';
      case 'advanced': return 'text-purple-600 bg-purple-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };
  const getLevelIcon = () => {
    switch (level) {
      case 'beginner': return <Star className="h-5 w-5 text-green-600" />;
      case 'intermediate': return <Code className="h-5 w-5 text-blue-600" />;
      case 'advanced': return <BookOpen className="h-5 w-5 text-purple-600" />;
      default: return null;
    }
  };
  // Update the video URLs with relevant CUDA content
  const getRelevantVideoUrl = (topic: string, title: string): string => {
    // Map of relevant CUDA videos based on topics
    const videoMap: Record<string, string> = {
      // Memory Access Patterns videos
      "Introduction to CUDA Memory Patterns": "https://www.youtube.com/embed/CZgM3DEBplE", // CUDA Memory Model
      "Advanced Memory Access Optimization": "https://www.youtube.com/embed/tGu5RKr6X6U", // Memory Coalescing in CUDA
      
      // UVM Optimization videos
      "Understanding UVM in CUDA": "https://www.youtube.com/embed/57RcVkDgwVE", // Unified Memory in CUDA
      
      // PTX Parallelization videos
      "PTX Optimization Techniques": "https://www.youtube.com/embed/8pTiLYBdZLM", // PTX Assembly and Optimization
    };
    
    return videoMap[title] || "https://www.youtube.com/embed/CZgM3DEBplE"; // Default to CUDA memory video
  };
  
  // Update the handleVideoClick function to ensure it works properly
  const handleVideoClick = (video: TutorialVideo) => {
    // Create a copy of the video with the updated URL
    const updatedVideo = {
      ...video,
      url: getRelevantVideoUrl(topic, video.title)
    };
    setSelectedVideo(updatedVideo);
    setShowVideo(true);
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center mb-4">
        <div className={`w-8 h-8 rounded-full ${getLevelColor()} flex items-center justify-center mr-3`}>
          {getLevelIcon()}
        </div>
        <h3 className="text-xl font-bold text-gray-800">{topic}</h3>
      </div>

      <div className="mb-4">
        <span className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${getLevelColor()}`}>
          {level.charAt(0).toUpperCase() + level.slice(1)}
        </span>
      </div>

      {/* Video section */}
      {videos && videos.length > 0 && (
        <div className="mt-4">
          <h4 className="text-lg font-semibold mb-3">Tutorial Videos</h4>
          <div className="space-y-3">
            {videos.map((video, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center">
                  <Video className="h-5 w-5 text-gray-500 mr-2" />
                  <div>
                    <p className="font-medium">{video.title}</p>
                    <p className="text-sm text-gray-500">Duration: {video.duration}</p>
                  </div>
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={() => handleVideoClick(video)}
                    className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
                  >
                    View Here
                  </button>
                  <a
                    href={getRelevantVideoUrl(topic, video.title)}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="px-3 py-1 text-sm border border-blue-600 text-blue-600 rounded hover:bg-blue-50 flex items-center"
                  >
                    Open <ExternalLink className="h-3 w-3 ml-1" />
                  </a>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Video modal - ensure it works properly */}
      {showVideo && selectedVideo && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-white p-4 rounded-lg max-w-4xl w-full mx-4 relative">
            <button
              onClick={() => setShowVideo(false)}
              className="absolute top-2 right-2 text-gray-500 hover:text-gray-700 text-2xl font-bold z-10"
              aria-label="Close video"
            >
              ×
            </button>
            <h3 className="text-xl font-bold mb-4">{selectedVideo.title}</h3>
            <div className="relative pt-[56.25%]">
              {selectedVideo.url ? (
                <iframe
                  src={selectedVideo.url}
                  className="absolute inset-0 w-full h-full"
                  allowFullScreen
                  title={selectedVideo.title}
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                />
              ) : (
                <div className="absolute inset-0 w-full h-full flex items-center justify-center bg-gray-200">
                  <p>Video unavailable</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Existing interactive section */}
      {interactive && (
        <div className="mt-6">
          <button
            onClick={() => setCurrentStep(prev => prev + 1)}
            className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
          >
            Next Step
            <ChevronRight className="ml-2 h-4 w-4" />
          </button>
        </div>
      )}
    </div>
  );
};

export default Tutorial;