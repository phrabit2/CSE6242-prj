"""
CSE6242 Team26 — Baseball CPD Dashboard
AWS Cloud Architecture & CI/CD Pipeline Diagram

Usage:
    python infra/architecture_diagram.py
    → Generates: infra/cse6242_team26_architecture.png
"""

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.compute import EC2
from diagrams.aws.network import Endpoint
from diagrams.aws.storage import S3
from diagrams.aws.security import IAM, IAMRole
from diagrams.onprem.iac import Terraform
from diagrams.onprem.vcs import Github
from diagrams.onprem.ci import GithubActions
from diagrams.onprem.client import Users, User

graph_attr = {
    "fontsize": "22",
    "fontname": "Helvetica Bold",
    "bgcolor": "white",
    "pad": "0.6",
    "nodesep": "0.6",
    "ranksep": "1.0",
    "splines": "spline",
    "compound": "true",
}

node_attr = {
    "fontsize": "10",
    "fontname": "Helvetica",
}

edge_attr = {
    "fontsize": "9",
    "fontname": "Helvetica",
}

with Diagram(
    "CSE6242 Team26 — Baseball CPD Dashboard\nAWS Cloud Architecture & CI/CD Pipeline",
    filename="infra/cse6242_team26_architecture",
    show=False,
    direction="LR",
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr,
    outformat="png",
):

    # ══════════════════════════════════
    # Left: Actors
    # ══════════════════════════════════
    team = Users("Developers\n(Eric, Irene,\nClara, Ethan)")
    end_user = User("End Users")

    # ══════════════════════════════════
    # Center: GitHub + CI/CD
    # ══════════════════════════════════
    with Cluster("GitHub Repository", graph_attr={
        "bgcolor": "#f5f5f5", "style": "rounded", "fontsize": "13",
    }):
        repo = Github("main branch")

    with Cluster("GitHub Actions CI/CD", graph_attr={
        "bgcolor": "#e8f5e9", "style": "rounded", "fontsize": "13",
    }):
        job_ec2 = GithubActions("Deploy EC2\n(SSH)")
        job_s3 = GithubActions("Sync S3\n(AWS CLI)")

    tf = Terraform("Terraform IaC")

    # ══════════════════════════════════
    # Right: AWS Cloud
    # ══════════════════════════════════
    with Cluster("AWS Cloud  (ap-northeast-2 / Seoul)", graph_attr={
        "bgcolor": "#fff8e1", "style": "rounded", "fontsize": "14",
    }):
        eip = Endpoint("Elastic IP\n15.165.52.135")

        with Cluster("VPC / Security Group\nPorts: 22, 8050, 8501", graph_attr={
            "bgcolor": "#e3f2fd", "style": "rounded", "fontsize": "11",
        }):
            ec2 = EC2("EC2 (t3.micro)\nUbuntu 22.04\nStreamlit :8501")

        s3 = S3("S3 Bucket\nteam26-cpd-data\nraw/ + processed/")

        iam_group = IAM("IAM Group\ns3-uploaders\n(4 members)")

    # ══════════════════════════════════
    # Connections
    # ══════════════════════════════════

    # Developer → GitHub → CI/CD
    team >> Edge(label="git push", color="#2e7d32", style="bold") >> repo
    repo >> Edge(label="on push\nto main", color="#e65100", style="bold") >> job_ec2
    repo >> Edge(color="#e65100", style="bold") >> job_s3

    # CI/CD → AWS
    job_ec2 >> Edge(
        label="SSH → git pull\npip install\nrestart streamlit",
        color="#1565c0",
    ) >> ec2

    job_s3 >> Edge(label="aws s3 sync", color="#2e7d32") >> s3

    # Terraform → AWS
    tf >> Edge(label="provisions", color="#7b1fa2", style="bold") >> ec2
    tf >> Edge(color="#7b1fa2", style="bold") >> s3
    tf >> Edge(color="#7b1fa2", style="bold") >> eip
    tf >> Edge(color="#7b1fa2", style="bold") >> iam_group

    # Elastic IP → EC2
    eip - Edge(label="attached", color="#795548", style="dashed") - ec2

    # IAM: team → S3
    team >> Edge(label="boto3\nS3 access", color="#7b1fa2") >> iam_group
    iam_group >> Edge(label="CRUD", color="#7b1fa2", style="dashed") >> s3

    # End User → Dashboard
    end_user >> Edge(label="HTTP :8501", color="#c62828", style="bold") >> eip
