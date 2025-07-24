# Documentation Agent

Create a .env.local file
NEXT_PUBLIC_API_URL="http://localhost:8000"

to run locally
npm run dev

# to create a static build
npm run build

# to run the static build
move contents of "out" folder to out/projects/documentation-agent

then run:
    npx http-server out -p 4000

