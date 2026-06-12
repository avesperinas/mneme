# Deployment entrypoints. Dev tooling (fmt/lint/test/chat) lives in the justfile.

.PHONY: run run-prod stop down

# Detect GPU, pick the serving profile, and bring up the base stack.
run:
	@bash scripts/detect.sh

# Stop the running stack but keep the containers; resume with `make run`.
# Both profiles are listed so it targets whichever serving engine is up.
stop:
	@docker compose --profile gpu --profile cpu stop

# Stop and remove containers and the network. Named volumes (qdrant data,
# model caches) are preserved; this intentionally never passes -v.
down:
	@docker compose --profile gpu --profile cpu down

# Production overlay (Langfuse, auth, Cloudflare Tunnel) is introduced in phase 8.
run-prod:
	@echo "make run-prod: the production overlay is introduced in phase 8."
	@exit 1
