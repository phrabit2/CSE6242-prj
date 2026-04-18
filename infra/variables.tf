variable "aws_region" {
  description = "AWS region"
  default     = "ap-northeast-2"
}

variable "project_name" {
  description = "Project name for tagging"
  default     = "team26-cpd"
}

variable "key_name" {
  description = "EC2 SSH key pair name"
  type        = string
}

variable "my_ip" {
  description = "Your IP address for SSH access (CIDR format, e.g. 203.0.113.0/32)"
  type        = string
}

variable "team_members" {
  description = "S3 업로드 권한을 부여할 팀원 IAM 사용자명 목록"
  type        = list(string)
  default     = []
}