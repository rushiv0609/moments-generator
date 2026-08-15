# Local AI Moments Generator

An offline, local AI-powered moments video generator for Apple Silicon (M5 / M5 Pro / M-series). Ingests photos and videos (up to 20 GB), understands visual semantics using SigLIP 2 embeddings, and stitches the best highlights matching a natural-language prompt without generative video synthesis.

## Quick Start

```bash
# 1. Start the system (installs dependencies, starts Qdrant, runs FastAPI)
./start.sh

# 2. Open your browser
open http://localhost:8000/ui/
```

## Documentation

- [Product Requirements](docs/product-requirements.md)
- [Technical Architecture](docs/technical-architecture.md)
- [Project Setup & Structure](docs/project-setup.md)
