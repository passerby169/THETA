/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    // unoptimized: true 已移除 — 启用 Next.js 图片优化（WebP/AVIF 自动转换 + lazy loading）
    formats: ['image/webp', 'image/avif'],
  },
  // 禁用严格模式以避免双重渲染导致的状态问题
  reactStrictMode: false,
  // Turbopack 配置 (Next.js 16+)
  turbopack: {
    resolveAlias: {},
  },
  // 开发服务器配置
  devIndicators: {
    buildActivity: false,
  },
  // 环境变量配置
  // Vercel 会自动读取环境变量，无需在此处设置默认值
  // 本地开发时使用 .env.local 文件
  // 输出配置
  // Docker 部署需要 standalone 模式
  output: 'standalone',
  // 按需导入 recharts 组件，减小初始 bundle 大小
  modularizeImports: {
    rechart: {
      transform: 'recharts/esm/components/{{member}}',
      skipDefaultConversion: true,
    },
  },
  async redirects() {
    return [
      { source: '/training', destination: '/dashboard', permanent: true },
      { source: '/login', destination: '/', permanent: true },
      { source: '/register', destination: '/', permanent: true },
      { source: '/results', destination: '/dashboard', permanent: true },
      { source: '/visualizations', destination: '/dashboard', permanent: true },
    ]
  },
}

export default nextConfig
