from celebvision.media import classify_source, ingest, extract_keyframe
from celebvision.stages.scenes import detect_scenes, build_scene_windows
from celebvision.stages.faces import match_faces_in_scene
from celebvision.stages.mentions import scene_transcript_text, extract_scene_mentions
from celebvision.stages.aggregate import build_scene_report, build_report
from celebvision.interfaces import InferenceClient, LLMClient, WatchlistIndex
from celebvision.models import Report


def run_pipeline(locator: str, watchlist: WatchlistIndex, keywords: list[str],
                 inference: InferenceClient, llm: LLMClient, workdir: str,
                 job_id: str, completed_at: str, face_threshold: float = 0.35,
                 frames_per_scene: int = 1) -> Report:
    source = classify_source(locator)
    video_path, audio_path = ingest(source, workdir)
    transcript = inference.transcribe(audio_path)
    boundaries = detect_scenes(video_path)
    windows = build_scene_windows(video_path, boundaries, workdir,
                                  keyframe_fn=extract_keyframe,
                                  frames_per_scene=frames_per_scene)

    scene_reports = []
    for scene in windows:
        faces = match_faces_in_scene(scene, inference, watchlist, face_threshold)
        text = scene_transcript_text(transcript, scene)
        mentions, hits = extract_scene_mentions(text, watchlist, keywords, llm)
        scene_reports.append(build_scene_report(scene, text, mentions, hits, faces))

    return build_report(job_id, source, watchlist.watchlist_id, completed_at,
                        transcript.duration_s, scene_reports)
