from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(relative_path):
    with (ROOT / relative_path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_deployment_operational_settings():
    deployment = load_yaml("k8s/deployment.yaml")

    assert deployment["metadata"]["name"] == "mp034-api"
    assert deployment["spec"]["replicas"] == 2

    pod_spec = deployment["spec"]["template"]["spec"]
    pod_labels = deployment["spec"]["template"]["metadata"]["labels"]
    assert deployment["spec"]["selector"]["matchLabels"] == pod_labels

    container = pod_spec["containers"][0]
    assert container["image"] == "mp034-api:1.0"
    assert container["ports"][0]["containerPort"] == 8000

    readiness = container["readinessProbe"]
    liveness = container["livenessProbe"]
    assert readiness["httpGet"]["path"] == "/health"
    assert liveness["httpGet"]["path"] == "/health"
    assert liveness["initialDelaySeconds"] > readiness["initialDelaySeconds"]


def test_service_targets_deployment():
    deployment = load_yaml("k8s/deployment.yaml")
    service = load_yaml("k8s/service.yaml")

    assert service["metadata"]["name"] == "mp034-svc"
    assert service["spec"]["selector"] == deployment["spec"]["template"]["metadata"]["labels"]

    service_port = service["spec"]["ports"][0]
    assert service_port["port"] == 80
    assert service_port["targetPort"] == 8000