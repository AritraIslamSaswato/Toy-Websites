import React from 'react';
import { Article } from '../data/articles';
import { Calendar, User, Tag } from 'lucide-react';

interface ArticleCardProps {
  article: Article;
}

const ArticleCard: React.FC<ArticleCardProps> = ({ article }) => {
  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden transition-transform duration-300 hover:shadow-xl hover:-translate-y-1">
      <img 
        src={article.imageUrl} 
        alt={article.title} 
        className="w-full h-48 object-cover"
      />
      <div className="p-5">
        <h3 className="text-xl font-bold text-gray-800 mb-2">{article.title}</h3>
        
        <div className="flex items-center text-gray-600 text-sm mb-3">
          <User size={16} className="mr-1" />
          <span className="mr-4">{article.author}</span>
          <Calendar size={16} className="mr-1" />
          <span>{article.date}</span>
        </div>
        
        <p className="text-gray-600 mb-4">{article.summary}</p>
        
        <div className="flex flex-wrap gap-2 mb-4">
          {article.tags.map((tag, index) => (
            <span 
              key={index} 
              className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
            >
              <Tag size={12} className="mr-1" />
              {tag}
            </span>
          ))}
        </div>
        
        <a 
          href={article.url} 
          className="inline-block px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
        >
          Read More
        </a>
      </div>
    </div>
  );
};

export default ArticleCard;