variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "Cluster region"
  type        = string
  default     = "us-central1-c"
}

variable "cluster_name" {
  type    = string
  default = "my-gke"
}

variable "network" {
  type    = string
  default = "default"
}

variable "subnetwork" {
  type    = string
  default = "default"
}

variable "machine_type" {
  default = "e2-standard-4"
}

variable "node_count" {
  default = 1
}
