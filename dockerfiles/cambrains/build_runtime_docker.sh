 #！ /bin/bash
echo "Building the cambrains runtime docker image..."
docker build -t vlmevalkit_cambrains:latest -f dockerfiles/cambrains/Dockerfile --progress=plain .
echo "Successfully built the cambrains runtime docker image: vlmevalkit_cambrains:latest"
