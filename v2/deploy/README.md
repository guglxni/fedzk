# FEDzk v2 deploy notes

Local demo (requires Docker):

```bash
cd v2
docker compose -f deploy/docker-compose.yml up --build
# fedzk-zk: http://127.0.0.1:8787/healthz
# coordinator: http://127.0.0.1:8000/docs
```

Day-to-day without Docker:

```bash
cd v2/rust && cargo build -p fedzk-zk --release
./target/release/fedzk-zk serve --bind 127.0.0.1:8787
# other terminal:
cd v2 && .venv/bin/uvicorn fedzk.coordinator.api:app --port 8000
```
