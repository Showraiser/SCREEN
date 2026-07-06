/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    // Override at build time: NEXT_PUBLIC_API_URL=https://your-api.com npm run build
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  },
};

module.exports = nextConfig;
