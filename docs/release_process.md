# Release Process

## Build and tag
- On every push to `main`, CI builds and pushes `ghcr.io/<repo>/api:sha-<commit>` and `:latest`.
- On tag `v*`, CI also pushes `ghcr.io/<repo>/api:<tag>`.

## Staging smoke
- CD workflow runs a compose-based smoke check:
  - `/dev/reset`
  - upload a tiny document
  - `/answer` must return a grounded answer with citations

## Deploy
- Staging deploy: use `:latest`.
- Prod/demo deploy: use the semver tag (e.g., `:v0.1.0`).

## Rollback
- Re-deploy the previous known-good tag (or `sha-<commit>`).
- Re-run smoke checks after rollback.

## Nightly eval
- Nightly workflow rebuilds the index from `data/seed`, runs full eval, and uploads artifacts.

