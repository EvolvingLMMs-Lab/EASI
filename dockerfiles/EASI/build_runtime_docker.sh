 #！ /bin/bash
echo "Building the EASI runtime docker image..."
docker build -t vlmevalkit_EASI:latest -f dockerfiles/EASI/Dockerfile --progress=plain .
echo "Successfully built the EASI runtime docker image: vlmevalkit_EASI:latest"
