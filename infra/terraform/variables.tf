variable "aws_region" {
  description = "AWS region used by the reference deployment."
  type        = string
  default     = "eu-central-1"
}

variable "project_name" {
  description = "Resource-name prefix."
  type        = string
  default     = "struct-xai"
}

variable "db_username" {
  description = "PostgreSQL application username."
  type        = string
  default     = "structxai"
}

variable "db_password" {
  description = "PostgreSQL password. Supply via TF_VAR_db_password or a secrets manager in real deployments."
  type        = string
  sensitive   = true
}
