# Technical Documentation

## Entry Surface
Everything in the Vision Archive AI stack is controlled from `main.py`. Running `python main.py` prints the current list of supported commands (setup, processing, interfaces, utilities) and it performs a quick system status check so operators always have a single reference point for scripts they might need.

## Core Workflows
- **Setup**: Download the CLIP models and prepare the LFW dataset before ingesting images. These helpers are listed under the "SETUP" section of `main.py` and are the supported entry points for provisioning dependencies.
- **Processing**: Use `production_pipeline.py` to extract embeddings from `test_images/`, and run `tune_clustering.py` + `gallery.py` if you need to review the resulting clusters. The same `main.py` listing highlights those commands under "PROCESSING".
- **Interfaces**: Launch `streamlit run app.py`, `python search_app.py`, or `python server.py` exactly as shown under the "INTERFACES" header in `main.py`. (Production deployments should prefer `server.py` behind a proxy and supply `VISION_API_KEY` to lock down the FastAPI surface.)
- **Utilities**: Repair tools such as `rename_person.py`, `watcher.py`, and `align_database.py` are the supported helpers; they appear under "UTILITIES" in `main.py` and are the only officially supported admin commands going forward.

## Logging & Configuration
- Logs are emitted to the `logs/` directory (`vision_config.py` manages `VISION_LOG_DIR`/`VISION_LOG_FILE`) so that runtime output is isolated from the project root. The default file is `logs/vision_archive.log`, and a console handler remains active for real-time visibility.
- When exposing `server.py` on any network, set `VISION_API_KEY` to enforce a simple `X-API-Key` header check before serving requests.

## Operational Notes
- The FAISS models and CLIP weights are intentionally kept outside of Git; use `python download_models.py` to bootstrap the `/models` cache before running the pipelines.
- There are no longer any `test_pipeline*` scripts in the repo; those workflows have been consolidated into `production_pipeline.py`, and `main.py` is the only documentation source for how to invoke them.
