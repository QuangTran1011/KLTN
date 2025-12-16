terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# -----------------------------
#  GKE CLUSTER
# -----------------------------
resource "google_container_cluster" "gke" {
  name     = var.cluster_name
  location = var.region

  network    = var.network
  subnetwork = var.subnetwork

  release_channel {
    channel = "REGULAR"
  }

  remove_default_node_pool = true
  initial_node_count       = 1

  ip_allocation_policy {}  

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
}

# -----------------------------
#  NODE POOL
# -----------------------------
resource "google_container_node_pool" "general" {
  name       = "general-pool"
  location   = var.region
  cluster    = google_container_cluster.gke.name

  node_count = var.node_count

  node_config {
    machine_type = var.machine_type

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      role = "general"
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }

    advanced_machine_features {
        threads_per_core = 2
    }

    tags = ["gke-node"]
  }
}
