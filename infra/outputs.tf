output "ec2_public_ip" {
  description = "EC2 Elastic IP (고정)"
  value       = aws_eip.dev_server_eip.public_ip
}

output "ec2_public_dns" {
  description = "EC2 instance public DNS"
  value       = aws_instance.dev_server.public_dns
}

output "s3_bucket_name" {
  description = "S3 bucket name for data"
  value       = aws_s3_bucket.data_bucket.id
}

output "dashboard_url" {
  description = "Dash dashboard URL"
  value       = "http://${aws_eip.dev_server_eip.public_ip}:8050"
}

output "elastic_ip" {
  description = "Elastic IP address (GitHub Secrets EC2_HOST에 등록)"
  value       = aws_eip.dev_server_eip.public_ip
}

output "team_member_access_keys" {
  description = "팀원 IAM Access Key ID 목록"
  value       = { for k, v in aws_iam_access_key.team_member_keys : k => v.id }
}

output "team_member_secret_keys" {
  description = "팀원 IAM Secret Access Key (terraform output -json으로 확인, 최초 1회만 노출)"
  value       = { for k, v in aws_iam_access_key.team_member_keys : k => v.secret }
  sensitive   = true
}