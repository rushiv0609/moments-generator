"""
CLI Test Script for LangGraph Director Agent.
Allows testing the state machine with live Ollama or Mock mode against an indexed workspace.

Usage:
    python scripts/test_director.py --workspace /Users/rushivyas/Desktop/pinbhabha-full --prompt "Scenic nature and mountains" --duration 30
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.workspace import get_workspace_manager
from app.core.embedder import MLXEmbedder, PyTorchMPSEmbedder, HAS_MLX
from app.core.director import DirectorAgent, OllamaDirectorLLM, MockDirectorLLM


def main():
    parser = argparse.ArgumentParser(description="Test LangGraph Director Agent")
    parser.add_argument("--workspace", type=str, default="/Users/rushivyas/Desktop/pinbhabha-full", help="Workspace folder")
    parser.add_argument("--prompt", type=str, default="Scenic outdoor moments and activities", help="User creative prompt")
    parser.add_argument("--duration", type=int, default=30, help="Target video duration (seconds)")
    parser.add_argument("--mode", type=str, default="dual", choices=["scene", "frame", "dual"], help="Retrieval mode")
    parser.add_argument("--model", type=str, default="qwen2.5:7b", help="Ollama model name (e.g. qwen2.5:7b, gemma:7b)")
    parser.add_argument("--mock-llm", action="store_true", help="Force Mock LLM instead of live Ollama")
    parser.add_argument("--alternatives", action="store_true", help="Generate all 3 alternatives (Alt-A, Alt-B, Alt-C)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n" + "=" * 70)
    print("🎬 LANGGRAPH DIRECTOR AGENT CLI TESTER")
    print("=" * 70)
    print(f"📁 Workspace:       {args.workspace}")
    print(f"💡 Prompt:          '{args.prompt}'")
    print(f"⏱️ Target Duration:  {args.duration}s")
    print(f"🔍 Retrieval Mode:  {args.mode}")
    print(f"🤖 LLM Backend:     {'MockDirectorLLM' if args.mock_llm else f'Ollama ({args.model})'}")
    print("=" * 70 + "\n")

    # 1. Connect Workspace
    workspace_mgr = get_workspace_manager()
    workspace_mgr.set_workspace(args.workspace)
    manifest = workspace_mgr.get_manifest_db()
    qdrant = workspace_mgr.get_qdrant_db()
    collection_name = workspace_mgr.collection_name

    # 2. Initialize Embedder
    embedder = MLXEmbedder() if HAS_MLX else PyTorchMPSEmbedder()

    # 3. Initialize LLM
    if args.mock_llm:
        llm = MockDirectorLLM()
    else:
        llm = OllamaDirectorLLM(model_name=args.model, fallback_to_mock=False)

    agent = DirectorAgent(
        embedder=embedder,
        qdrant=qdrant,
        collection_name=collection_name,
        llm=llm,
        manifest=manifest,
        model_name=args.model,
    )

    if args.alternatives:
        print("🔄 Generating 3 Parallel Alternatives (Alt-A Scene, Alt-B Frame, Alt-C Dual)...\n")
        alts = agent.generate_alternatives(
            prompt=args.prompt,
            target_duration=args.duration,
            job_id_prefix="cli_test",
        )
        for name, state in alts.items():
            print(f"\n--- {name.upper()} ---")
            print(f"Storyboard Segments: {len(state['storyboard'])}")
            tot_dur = sum(s.get('duration', 0) for s in state['storyboard'])
            print(f"Total Duration: {tot_dur:.1f}s (Target: {args.duration}s)")
            print(f"Approved: {state['approved']} (Iterations: {state['iteration_count']})")
    else:
        print("🚀 Executing LangGraph State Machine...\n")
        final_state = agent.run(
            prompt=args.prompt,
            target_duration=args.duration,
            retrieval_mode=args.mode,
            job_id="cli_test_single",
        )

        print("\n" + "-" * 70)
        print("📊 EXECUTION SUMMARY")
        print("-" * 70)
        print(f"🎯 Narrative Arc:      {final_state.get('narrative_arc')}")
        print(f"🔍 Sub-Queries ({len(final_state.get('search_queries', []))}):")
        for q in final_state.get("search_queries", []):
            print(f"   • \"{q}\"")

        candidates = final_state.get("retrieved_candidates", [])
        print(f"\n📦 Retrieved Candidates: {len(candidates)} total")
        for c in candidates[:5]:
            print(f"   - [{c.get('granularity')}] {Path(c.get('file_path')).name} (score: {c.get('score', 0):.2f})")

        storyboard = final_state.get("storyboard", [])
        total_dur = sum(s.get("duration", 0) for s in storyboard)
        print(f"\n🎞️ Final Storyboard ({len(storyboard)} segments | {total_dur:.1f}s total):")
        for idx, seg in enumerate(storyboard):
            fname = Path(seg.get("file_path", "")).name
            stype = seg.get("segment_type")
            dur = seg.get("duration", 0)
            just = seg.get("justification", "")
            print(f"   [{idx + 1}] {fname} ({stype}, {dur}s) -> {just}")

        print(f"\n✅ Editor Approved:     {final_state.get('approved')}")
        print(f"🔁 Iterations:          {final_state.get('iteration_count')}")
        if final_state.get("editor_feedback"):
            print(f"💬 Editor Feedback:")
            for fb in final_state.get("editor_feedback", []):
                print(f"   • {fb}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
