export interface Article {
  id: number;
  title: string;
  author: string;
  date: string;
  summary: string;
  imageUrl: string;
  tags: string[];
  url: string;
}

export const articles: Article[] = [
  {
    id: 1,
    title: "Optimizing Memory Access Patterns in CUDA for Energy Efficiency",
    author: "Dr. Sarah Chen",
    date: "2025-03-15",
    summary: "This research explores novel techniques for optimizing memory access patterns in CUDA applications, resulting in up to 35% energy savings without compromising performance.",
    imageUrl: "https://images.unsplash.com/photo-1558494949-ef010cbdcc31?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80",
    tags: ["Memory Optimization", "Energy Efficiency", "CUDA"],
    url: "/research/memory-access-patterns"
  },
  {
    id: 2,
    title: "Unified Virtual Memory: Performance Analysis and Best Practices",
    author: "Prof. Michael Rodriguez",
    date: "2025-02-28",
    summary: "A comprehensive study of Unified Virtual Memory (UVM) in CUDA, analyzing its impact on performance and energy consumption across various workloads.",
    imageUrl: "https://images.unsplash.com/photo-1544197150-b99a580bb7a8?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80",
    tags: ["UVM", "Performance Analysis", "Best Practices"],
    url: "/research/uvm-analysis"
  },
  {
    id: 3,
    title: "PTX-Level Parallelization Strategies for Scientific Computing",
    author: "Dr. James Wilson",
    date: "2025-01-10",
    summary: "This paper presents novel PTX-level parallelization strategies that achieve significant performance improvements for scientific computing applications.",
    imageUrl: "https://images.unsplash.com/photo-1551033406-611cf9a28f67?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80",
    tags: ["PTX", "Parallelization", "Scientific Computing"],
    url: "/research/ptx-parallelization"
  },
  {
    id: 4,
    title: "Dynamic Kernel Fusion for Energy-Efficient GPU Computing",
    author: "Dr. Emily Zhang",
    date: "2024-12-05",
    summary: "A novel approach to dynamically fuse CUDA kernels at runtime, reducing kernel launch overhead and improving energy efficiency by up to 28%.",
    imageUrl: "https://images.unsplash.com/photo-1580584126903-c17d41830450?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80",
    tags: ["Kernel Fusion", "Dynamic Optimization", "Energy Efficiency"],
    url: "/research/dynamic-kernel-fusion"
  }
];