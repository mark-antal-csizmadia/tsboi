MLGWHOST=$(shell docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mlflow_server)
# Localhost is used to access mlflow outside the docker container.
MLFLOW_PORT=5050


mlflowquickcheck:
	# Simple access check to mlflow server on host port; just lists experiments.
	echo "MLFLOW_TRACKING_URI=http://${MLGWHOST}:${MLFLOW_PORT}"
	docker exec                                                      \
	    -e MLFLOW_TRACKING_URI=http://${MLGWHOST}:${MLFLOW_PORT}     \
	    mlflow_server                                                \
	    mlflow experiments search  # (for mlflow v2 switch 'list' to 'search')