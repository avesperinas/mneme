# Deployment entrypoints. Dev tooling (fmt/lint/test/chat) lives in the justfile.

.PHONY: run run-prod

# Detect GPU, pick the serving profile, and bring up the base stack.
run:
	@bash scripts/detect.sh

# Production overlay (Langfuse, auth, Cloudflare Tunnel) is introduced in phase 8.
run-prod:
	@echo "make run-prod: the production overlay is introduced in phase 8."
	@exit 1
