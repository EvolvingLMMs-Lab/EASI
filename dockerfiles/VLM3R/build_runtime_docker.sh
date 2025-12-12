 #！ /bin/bash
echo "Building the VLM3R runtime docker image..."
docker build -t vlmevalkit_vlm3r:latest -f dockerfiles/VLM3R/Dockerfile --progress=plain .
echo "Successfully built the VLM3R runtime docker image: vlmevalkit_vlm3r:latest"
