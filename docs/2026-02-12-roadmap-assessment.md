# Infrastructure Roadmap Assessment

**Date:** 2026-02-12
**Context:** Career transition from software engineering to AI infrastructure. Dual goals: (1) operational base for business ideas, (2) LinkedIn portfolio demonstrating end-to-end AI infrastructure management.

## Where the Project Stands Today

The platform is already a strong portfolio piece. It demonstrates:

- **Full-stack IaC**: VPC, EKS, Karpenter, KubeRay, Kueue, cert-manager — all Terraform/Terragrunt
- **Production-grade observability**: Prometheus + Grafana + ADOT + AMP + Loki with cross-correlation
- **Deep testing discipline**: Terratest suite with 4-level assertion pyramid (pod exists → data flowing)
- **Battle-tested teardown**: LIFO destruction, finalizer cleanup, orphan resource handling
- **Real engineering decisions**: DNS hardening, Pod Identity migration, ADOT label parity — not toy problems

This puts you ahead of most candidates who show "I deployed an EKS cluster." You've shown you can debug the hard problems (conntrack, DNS amplification, webhook race conditions) and write infrastructure that actually tears down cleanly.

## What to Build Next (Ranked by Signal-to-Effort)

### Tier 1: High Signal, Moderate Effort

These are the features that hiring managers and LinkedIn audiences immediately recognize as "this person can run production AI workloads."

#### 1. GPU Workloads on Karpenter

**Why:** This is the single most relevant addition for AI infrastructure. Every AI infra role involves GPU scheduling. Running actual GPU inference or training on your cluster — with Karpenter provisioning GPU nodes on-demand — is the strongest possible signal.

**What to build:**
- Karpenter NodePool for `g5.xlarge` or `g6.xlarge` (NVIDIA L4/A10G)
- RayJob that runs a real PyTorch inference task (not just hello-world)
- Kueue ResourceFlavor for GPU quota governance
- Terratest assertion that the RayJob completes on a GPU node

**Cost control:** Spot instances for g5.xlarge are ~$0.40/hr. A test run takes ~10 minutes. Budget: ~$0.10 per test cycle.

**LinkedIn angle:** "Built GPU-aware autoscaling for Ray ML workloads on EKS — Karpenter provisions GPU nodes JIT, Kueue prevents resource stampedes."

#### 2. Ingress + External Access (ALB + External DNS)

**Why:** Right now everything is `kubectl port-forward`. Adding an ALB Ingress Controller with External DNS makes your cluster actually usable as a platform — for yourself and for demos.

**What to build:**
- AWS Load Balancer Controller (Helm)
- External DNS for Route53 automation
- Ingress resources for Grafana and Ray Dashboard
- ACM certificate for HTTPS (free via Let's Encrypt or ACM)
- Terratest: HTTP GET to `grafana.yourdomain.com` returns 200

**LinkedIn angle:** "End-to-end platform: deploy a Ray job via API, monitor it on Grafana, all behind TLS ingress with automated DNS."

#### 3. CI/CD Pipeline (GitHub Actions)

**Why:** Proves the infrastructure isn't just "works on my laptop." A CI pipeline that runs `make test-up` → verify → `make test-down` on every PR shows operational maturity.

**What to build:**
- GitHub Actions workflow triggered on PR
- IAM role with OIDC federation for GitHub (no static credentials)
- Workflow: `make test-all` with 60-minute timeout
- Cost guard: only runs on `[ci-full]` label or manual trigger
- Badge in README showing last successful run

**LinkedIn angle:** "Full GitOps pipeline — every PR runs end-to-end infrastructure tests against a real EKS cluster, then tears it down."

### Tier 2: Strong Signal, Lower Effort

These round out the portfolio and show breadth.

#### 4. Prometheus Agent Mode + AMP Rulers

**Why:** Your README already mentions this as Phase 2. Doing it shows you understand the operational trajectory of monitoring at scale — local Prometheus doesn't scale, you need to offload to managed backends.

**What to build:**
- Toggle `enable_prometheus_agent_mode` variable
- Prometheus config switch to agent mode (WAL-only, no local TSDB)
- AMP recording rules and alert rules via Terraform
- Terratest: verify AMP workspace has active rule groups

**LinkedIn angle:** "Migrated from full Prometheus to Agent Mode + AMP Rulers — reduced in-cluster memory footprint while maintaining full alerting capability."

#### 5. Velero Backup + Disaster Recovery

**Why:** Every production cluster needs backup/restore. This is table stakes for senior roles but rarely demonstrated in portfolios.

**What to build:**
- Velero with S3 backend
- Scheduled backup of critical namespaces (kuberay, prometheus)
- Terratest: create a ConfigMap, backup, delete it, restore, verify it exists

**LinkedIn angle:** "Automated disaster recovery with Velero — scheduled backups to S3 with verified restore capability."

#### 6. Network Policies + Pod Security Standards

**Why:** Security posture. Shows you think about blast radius, not just functionality.

**What to build:**
- Default-deny NetworkPolicy per namespace
- Allow-list for known traffic patterns (Prometheus scraping, Loki shipping)
- Pod Security Admission (PSA) labels on namespaces (restricted/baseline)
- Terratest: verify a pod in namespace A cannot reach namespace B

**LinkedIn angle:** "Zero-trust networking — default-deny with explicit allow-lists, Pod Security Standards enforced at namespace level."

### Tier 3: Business-Enabling (For Your Own Use)

These make the platform actually useful for running your business ideas.

#### 7. Shared Storage (EFS + S3 Gateway)

**Why:** Ray workers need shared storage for model weights, datasets, and checkpoints. EFS CSI driver is already wired (IAM role exists), just needs the FileSystem + StorageClass.

**What to build:**
- EFS FileSystem in the VPC
- StorageClass for dynamic PV provisioning
- S3 Gateway VPC Endpoint for fast model/data access
- RayJob that reads a model from S3, runs inference, writes results to EFS

#### 8. ArgoCD or Flux for GitOps Deployments

**Why:** Once you have ingress and CI/CD, the natural next step is GitOps for workload deployments. This lets you deploy Ray workloads by pushing YAML to a repo instead of running kubectl.

#### 9. Cost Observability (Kubecost or OpenCost)

**Why:** You're already tracking AWS costs manually via CLI. Kubecost gives per-namespace, per-workload cost attribution inside Grafana. Useful for your own cost management and a strong portfolio signal.

## Recommended Execution Order

```
Phase 1 (Next 2 weeks):
  1. GPU Workloads on Karpenter  ← highest signal for AI infra roles
  2. CI/CD Pipeline              ← proves operational maturity

Phase 2 (Weeks 3-4):
  3. Ingress + External Access   ← makes the platform demo-able
  4. Network Policies            ← security posture (quick win)

Phase 3 (Weeks 5-6):
  5. Prometheus Agent Mode       ← shows scaling knowledge
  6. EFS + S3 Gateway            ← enables real workloads

Phase 4 (As needed):
  7. Velero Backup
  8. ArgoCD
  9. Kubecost
```

## What NOT to Build

- **Service mesh (Istio/Linkerd):** Over-engineering for a batch workload platform. Adds complexity without clear value for Ray jobs.
- **Multi-cluster federation:** Impressive but not practical at this stage. One well-built cluster beats two poorly-integrated ones.
- **Custom operators:** The existing KubeRay + Kueue combination covers your use case. Writing a custom operator is a distraction.
- **Terraform Cloud/Spacelift:** Your Makefile + Terratest workflow is already more rigorous than most Terraform Cloud setups. Don't add managed tooling just for the resume line.

## LinkedIn Content Strategy

Each feature above maps to a post. The format that performs well for infrastructure content:

1. **The problem** (1-2 sentences): "GPU scheduling on Kubernetes is broken by default — nodes take 5 minutes to provision."
2. **The approach** (2-3 sentences): "I built Karpenter NodePools with GPU-aware scheduling..."
3. **The proof** (screenshot or code snippet): Terratest output showing GPU RayJob succeeded in 45 seconds.
4. **The learning** (1 sentence): "The key insight was..."

Ship one post per feature. Consistency > volume. One well-documented feature per week for 6 weeks builds a compelling narrative of "I'm building production AI infrastructure from scratch."
