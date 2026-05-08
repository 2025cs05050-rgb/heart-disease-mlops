# Kubernetes deployment

Manifests for deploying the Heart Disease API on a local cluster
(Minikube or Docker Desktop's built-in Kubernetes) and on managed
clusters (GKE / EKS / AKS).

## Files

| File              | Purpose                                                  |
|-------------------|----------------------------------------------------------|
| `namespace.yaml`  | Creates `heart-disease` namespace                        |
| `configmap.yaml`  | Non-secret runtime config (MODEL_PATH, version, etc.)    |
| `deployment.yaml` | 2 replicas, probes, resource limits, non-root container  |
| `service.yaml`    | LoadBalancer (external) + ClusterIP (internal scraping)  |
| `ingress.yaml`    | Optional NGINX ingress at `heart-disease.local`          |
| `hpa.yaml`        | Horizontal Pod Autoscaler (2-5 replicas, CPU/memory)     |
| `kustomization.yaml` | Bundles all of the above for `kubectl apply -k .`     |

## Deploy on Minikube

```bash
# 1. Start cluster (with metrics server for HPA)
minikube start --cpus=2 --memory=4g
minikube addons enable metrics-server
minikube addons enable ingress         # optional, for ingress.yaml

# 2. Build image *inside* the Minikube docker daemon (so no registry needed)
minikube docker-env | Invoke-Expression          # PowerShell
# bash:  eval $(minikube docker-env)
docker build -t heart-disease-api:latest -f ../docker/Dockerfile ..

# 3. Apply manifests
kubectl apply -k .

# 4. Wait until the rollout is healthy
kubectl rollout status deployment/heart-disease-api -n heart-disease

# 5. Access the service
minikube service heart-disease-api -n heart-disease --url
# or, with ingress enabled, add to hosts:
#   <minikube ip>   heart-disease.local
# then open http://heart-disease.local/docs
```

## Deploy on Docker Desktop Kubernetes

```bash
# Image already lives in the local docker daemon — no extra step needed.
docker build -t heart-disease-api:latest -f docker/Dockerfile .
kubectl apply -k k8s/
kubectl rollout status deployment/heart-disease-api -n heart-disease

# LoadBalancer is exposed at localhost:80 on Docker Desktop
curl http://localhost/health
```

## Smoke test the deployed API

```bash
curl http://<EXTERNAL-IP>/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,
       "restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,
       "slope":3,"ca":0,"thal":6}'
```

Expected response:

```json
{
  "prediction": 0,
  "label": "No disease",
  "probability": 0.1718,
  "confidence": 0.8282,
  "model_version": "v1.0.0"
}
```

## Useful kubectl commands

```bash
kubectl get all          -n heart-disease
kubectl describe deploy  -n heart-disease heart-disease-api
kubectl logs -f deploy/heart-disease-api -n heart-disease
kubectl get hpa          -n heart-disease

# Tear down
kubectl delete -k .
```
