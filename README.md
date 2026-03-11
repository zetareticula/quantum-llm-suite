# Quantum LLM-as-a-Service Suite

Professional open-source platform for quantum-accelerated LLM serving with exposed quantum quantization aspects.

## Setup
- Backend: Python 3.10+, FastAPI.
- Frontend: React with TypeScript.
- Deployment: Docker Compose, Gunicorn for prod.
- Quantum: Qiskit, Cirq, PyQuil, Braket (real/sim).

## Running Locally
- Backend: `uvicorn backend.main:app --reload`
- Frontend: `cd frontend && npm start`
- Docker: `docker-compose up`

## Testing
- Install pytest: `pip install pytest`
- Run: `pytest tests/`

## Deployment
- Use `deploy.sh` for prod setup (requires NGINX, Gunicorn).
- Set env vars in `.env` for quantum tokens.

## Features
- Quantum providers: IBM, Rigetti, Google, IonQ, Local, AWS Braket.
- Cloud: AWS, GCP, Azure, Oracle, IBM.
- Models: Top 5 HF LLMs with quantization levels/modes (classical, quantum_distillation, quantum_embedding).
- API: OpenAPI at /docs.
- Scheduler: APScheduler for annealing.
- Concurrent: Asyncio for GPU/Quantum dispatch.

MIT License.