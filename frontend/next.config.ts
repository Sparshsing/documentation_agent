import type { NextConfig } from "next";

const isDev = process.env.NODE_ENV === 'development';
const basePath = "/projects/documentation-agent";

const nextConfig: NextConfig = {
  output: "export",
  basePath: isDev ? "" : basePath,
  assetPrefix: isDev ? "" : basePath,
  /* config options here */
  // distDir: 'out/projects/documentation-agent'
};

export default nextConfig;
