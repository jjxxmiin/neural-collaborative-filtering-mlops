
# Neural Collaborative Filtering

- OS: Window
- Dataset: MovieLens
- Framework: Pytorch

## Tools for MLOPS 

- Model Versioning: DVC
- Experiments: MLFlow
- Config: Hydra
- Deployment: BentoML

## MLFlow

```
mlflow run . --env-manager=local --entry-point gmf --experiment-name gmf
mlflow run . --env-manager=local --entry-point mlp --experiment-name mlp
mlflow run . --env-manager=local --entry-point neumf --experiment-name neumf
```

```
mlflow ui
```

- model training
- experiments
- logging

## DVC 

```
dvc init 
dvc remote add -d checkpoints {LOCAL, S3, etc.}
dvc add
dvc push
dvc pull

git commit -m "commit"
git tag -a "v1.0" -m "model, data v1.0"
```

- model & data versioning

## Reference 
- [https://github.com/yihong-chen/neural-collaborative-filtering](https://github.com/yihong-chen/neural-collaborative-filtering)
- [https://velog.io/@moey920/Window%EC%97%90%EC%84%9C-Minikube-Kubeflow-%EA%B5%AC%EC%B6%95%ED%95%98%EA%B8%B0](https://velog.io/@moey920/Window%EC%97%90%EC%84%9C-Minikube-Kubeflow-%EA%B5%AC%EC%B6%95%ED%95%98%EA%B8%B0)