terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ─────────────────────────────────────────────
# S3 Bucket: Statcast 데이터 저장용
# ─────────────────────────────────────────────
resource "aws_s3_bucket" "data_bucket" {
  bucket = "${var.project_name}-data-${data.aws_caller_identity.current.account_id}"

  tags = {
    Project = var.project_name
  }
}

resource "aws_s3_bucket_public_access_block" "data_bucket_block" {
  bucket = aws_s3_bucket.data_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

data "aws_caller_identity" "current" {}

# ─────────────────────────────────────────────
# Security Group: SSH + Dash 대시보드 포트
# ─────────────────────────────────────────────
resource "aws_security_group" "team26_sg" {
  name        = "${var.project_name}-sg"
  description = "Allow SSH and Dash dashboard access"

  # SSH access (open to all — secured by key-based auth)
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Plotly Dash 기본 포트
  ingress {
    description = "Dash Dashboard"
    from_port   = 8050
    to_port     = 8050
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Streamlit 기본 포트
  ingress {
    description = "Streamlit Dashboard"
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # 아웃바운드 전체 허용
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Project = var.project_name
  }
}

# ─────────────────────────────────────────────
# EC2 Instance: t3.micro (Free Tier)
# ─────────────────────────────────────────────
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_instance" "dev_server" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = "t3.micro"
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.team26_sg.id]

  user_data = file("userdata.sh")

  root_block_device {
    volume_size = 20    # Free Tier: 최대 30GB
    volume_type = "gp2"
  }

  tags = {
    Name    = "${var.project_name}-server"
    Project = var.project_name
  }
}

# ─────────────────────────────────────────────
# IAM: 팀원 S3 업로드 권한 관리
# ─────────────────────────────────────────────

# S3 접근 전용 IAM 그룹
resource "aws_iam_group" "s3_uploaders" {
  name = "${var.project_name}-s3-uploaders"
}

# 버킷에 대한 최소 권한 IAM 정책 (업로드/다운로드/목록 조회)
resource "aws_iam_policy" "s3_upload_policy" {
  name        = "${var.project_name}-s3-upload-policy"
  description = "팀원이 S3 데이터 버킷에 업로드/다운로드할 수 있는 최소 권한"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowBucketList"
        Effect = "Allow"
        Action = ["s3:ListBucket"]
        Resource = [
          aws_s3_bucket.data_bucket.arn
        ]
      },
      {
        Sid    = "AllowObjectOperations"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
        ]
        Resource = [
          "${aws_s3_bucket.data_bucket.arn}/*"
        ]
      }
    ]
  })
}

# 정책을 그룹에 연결
resource "aws_iam_group_policy_attachment" "s3_upload_attach" {
  group      = aws_iam_group.s3_uploaders.name
  policy_arn = aws_iam_policy.s3_upload_policy.arn
}

# 팀원별 IAM 사용자 생성
resource "aws_iam_user" "team_members" {
  for_each = toset(var.team_members)

  name = each.key
  tags = {
    Project = var.project_name
    Role    = "data-uploader"
  }
}

# 팀원 사용자를 그룹에 추가
resource "aws_iam_group_membership" "s3_uploaders_membership" {
  name  = "${var.project_name}-s3-uploaders-membership"
  group = aws_iam_group.s3_uploaders.name
  users = [for u in aws_iam_user.team_members : u.name]
}

# 팀원별 Access Key 생성 (terraform output으로 확인)
resource "aws_iam_access_key" "team_member_keys" {
  for_each = toset(var.team_members)
  user     = aws_iam_user.team_members[each.key].name
}

# ─────────────────────────────────────────────
# Elastic IP: 고정 퍼블릭 IP (EC2 재시작해도 유지)
# ─────────────────────────────────────────────
resource "aws_eip" "dev_server_eip" {
  instance = aws_instance.dev_server.id
  domain   = "vpc"

  tags = {
    Name    = "${var.project_name}-eip"
    Project = var.project_name
  }
}