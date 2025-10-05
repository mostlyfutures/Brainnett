FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package.json ./
COPY tsconfig.json ./

# Install dependencies
RUN npm install -g pnpm && \
    pnpm install

# Copy source code
COPY src/ ./src/

# Build TypeScript
RUN pnpm build

# Expose port
EXPOSE 3001

# Start service
CMD ["node", "dist/index.js"]
