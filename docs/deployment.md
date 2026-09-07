# Deployment

## Local (Docker Compose)
```
cp .env.example .env        # fill in ANTHROPIC_API_KEY and OPS_PERFORMANCE_API_URL
docker compose up --build
python -m scripts.index_documents   # first run only, or after a document change
```
The container health check (`Dockerfile`) hits `GET /health`, which itself verifies both
`operations-performance`'s API and the vector store are reachable -- `docker compose ps` will show
`unhealthy` if either dependency can't be reached, not just "container running."

## Connecting to operations-performance locally
`docker-compose.yml` defaults `OPS_PERFORMANCE_API_URL` to `http://host.docker.internal:8000`,
with `extra_hosts: host.docker.internal:host-gateway` so that resolves on Linux too (it's automatic
on Mac/Windows Docker Desktop, not on Linux without this). This assumes `operations-performance`
is running directly on the host (`python -m scripts.run_analysis` there), since that project
doesn't have its own Dockerfile yet. Once it does, the natural improvement is a shared external
Docker network so both containers reach each other by service name instead of the host-gateway
workaround.

## Cloud Run (matches operations-performance's GCP direction)
Not yet executed, but the path is the same shape as `operations-performance`'s BigQuery migration
-- documented before being built, not built blind:

```
docker build -t operations-assistant .
docker tag operations-assistant gcr.io/$GCP_PROJECT_ID/operations-assistant
docker push gcr.io/$GCP_PROJECT_ID/operations-assistant
gcloud run deploy operations-assistant \
  --image gcr.io/$GCP_PROJECT_ID/operations-assistant \
  --set-env-vars OPS_PERFORMANCE_API_URL=<deployed operations-performance URL> \
  --set-secrets ANTHROPIC_API_KEY=anthropic-api-key:latest
```

Real considerations for a real deployment, not yet resolved here:
- **`ANTHROPIC_API_KEY` via GCP Secret Manager**, not an env var in the deploy command as shown
  above (that example is illustrative, not what you'd actually run) -- `--set-secrets` is the
  correct pattern, referencing a secret already created in Secret Manager.
- **Chroma's local persistence doesn't survive Cloud Run's stateless, scale-to-zero model** --
  a Cloud Run deployment would need either a mounted persistent volume (Cloud Run supports this
  as of recent GCP versions) or a migration to a hosted vector store. This is the one piece of
  `docs/rag-design.md`'s "local is sufficient at this scale" reasoning that would actually need
  revisiting for a real cloud deployment, and it's called out explicitly there for that reason.
- **`operations-performance` needs to be reachable from wherever this deploys** -- either both
  services deploy to Cloud Run and reach each other over the internet (with real auth between
  them, not the open API this project currently has), or both deploy to the same VPC.

## What's NOT done
No CI/CD pipeline building and pushing the image automatically (Phase 21 stops at "the Dockerfile
and compose file work locally," matching FEATURES.md's Tier 1 bar for this phase -- automated
deployment pipelines are a reasonable Tier 3 addition, not attempted here).
