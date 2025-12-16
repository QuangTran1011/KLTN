.PHONY:
.ONESHELL:

include .env
export

ml-platform-up:
	docker compose -f compose.yaml up -d

ml-platform-logs:
# For make command that follows logs, if not add prefix '-' then when interrupet the command, it will complain with Error 130
	- docker compose -f compose.yaml logs -f