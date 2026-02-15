"""
Vision Archive AI - Main Entry Point
Run this file to see available commands and get started.
"""
import os
import sys

def main():
    print("=" * 50)
    print("  👁️  VISION ARCHIVE AI")
    print("=" * 50)
    print()
    print("Available Commands:")
    print()
    print("  SETUP:")
    print("    python download_models.py      Download CLIP models (~500MB)")
    print("    python prepare_lfw.py          Copy LFW dataset to test_images/")
    print()
    print("  PROCESSING:")
    print("    python production_pipeline.py  Extract face embeddings (fast)")
    print("    python tune_clustering.py      Cluster faces & tune parameters")
    print("    python reindex_search.py       Build semantic search index")
    print()
    print("  INTERFACES:")
    print("    streamlit run app.py           Launch Streamlit Web UI")
    print("    python gallery.py              Generate & serve HTML gallery")
    print("    python search_app.py           CLI text-to-image search")
    print("    python server.py               Start FastAPI server")
    print()
    print("  UTILITIES:")
    print("    python rename_person.py ID Name  Rename a person in DB")
    print("    python watcher.py [dir]        Watch folder for new images")
    print("    python align_database.py       Repair vector database")
    print()
    
    # Quick status check
    print("-" * 50)
    print("System Status:")
    models_ok = os.path.exists("models/buffalo_s/det_500m.onnx")
    clip_ok = os.path.exists("models/clip_image.onnx")
    db_ok = os.path.exists("models/people_db.json")
    index_ok = os.path.exists("models/faiss_index.bin")
    
    print(f"  Face Models:   {'✅ Ready' if models_ok else '❌ Missing (run download_models.py)'}")
    print(f"  CLIP Models:   {'✅ Ready' if clip_ok else '❌ Missing (run download_models.py)'}")
    print(f"  People DB:     {'✅ Ready' if db_ok else '⚠️  Not built yet'}")
    print(f"  Search Index:  {'✅ Ready' if index_ok else '⚠️  Not built yet'}")

if __name__ == "__main__":
    main()
