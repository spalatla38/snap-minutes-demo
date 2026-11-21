"""Offline tests for speaker identification (speakers.py) and how it feeds owners.

The voice and OCR models are not needed: these test the logic around them with made-up
observations, and the processing flow with the models faked.
    python tests/test_speakers.py      (or)      python -m pytest tests -q
"""
import sys
import types
from collections import Counter
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import jobs as J  # noqa: E402
import pipeline as P  # noqa: E402
import speakers as S  # noqa: E402
from segments import Segment, said, transcript_txt  # noqa: E402
from test_pipeline import FakeResp, FakeStream, bag_of_words  # noqa: E402

MAIN, THUMB1, THUMB2, THUMB3 = (0, 5), (4, 1), (4, 3), (4, 5)
PEOPLE = ["Precious Mensah", "Aarya Kulkarni", "Niharika Iyer", "Sai Reddy"]


def test_clean_name_fixes_ocr_slips_and_rejects_ui_text():
    assert S.clean_name("PreciousMensah") == "Precious Mensah"
    assert S.clean_name("Niharika lyer") == "Niharika Iyer"          # l read for I
    assert S.clean_name("Sai Reddy (Guest)") == "Sai Reddy"
    assert S.clean_name("Aarya Kulkarni | Organizer") == "Aarya Kulkarni"
    for ui_text in ["Recording", "Weekly sync 00:40 Recording", "Mute", "You", "12:04", "a", "lowercase name"]:
        assert S.clean_name(ui_text) is None, ui_text


def test_spelling_variants_fold_into_the_most_common():
    canon = S._fold_spellings(Counter({"aaryakulkarni": 9, "aaryakuikarni": 3, "aaryakukarni": 2, "sairedy": 4}))
    assert canon["aaryakuikarni"] == canon["aaryakukarni"] == "aaryakulkarni"
    assert canon["sairedy"] == "sairedy"


def speaker_view(talkers: dict, misread: dict = None, frames=5):
    """Fake OCR of a speaker-view recording. talkers: voice label -> person index. The talking
    person is on the big tile, everyone else on small tiles. misread: name -> wrong spelling
    used on the small tiles in some frames."""
    observed, frames_of, f = [], Counter(), 0
    for label, person in talkers.items():
        for k in range(frames):
            frames_of[label] += 1
            observed.append((f, label, PEOPLE[person], MAIN))
            others = [p for p in range(4) if p != person]
            for p, cell in zip(others, (THUMB1, THUMB2, THUMB3)):
                name = PEOPLE[p]
                if misread and name in misread and k % 2 == 0:
                    name = misread[name]
                observed.append((f, label, name, cell))
            observed.append((f, label, "Weekly sync", (0, 0)))  # top bar, filtered in real OCR
            f += 1
    return observed, frames_of


def test_big_tile_names_each_voice():
    obs, fr = speaker_view({"Speaker 1": 0, "Speaker 2": 1, "Speaker 3": 2, "Speaker 4": 3})
    named = S._match_names(obs, fr)
    assert {k: v["name"] for k, v in named.items()} == {
        "Speaker 1": "Precious Mensah", "Speaker 2": "Aarya Kulkarni",
        "Speaker 3": "Niharika Iyer", "Speaker 4": "Sai Reddy"}


def test_one_person_split_into_two_voices_gets_one_name_on_both():
    obs, fr = speaker_view({"Speaker 1": 0, "Speaker 2": 0, "Speaker 3": 1, "Speaker 4": 2, "Speaker 5": 3})
    named = S._match_names(obs, fr)
    assert named["Speaker 1"]["name"] == named["Speaker 2"]["name"] == "Precious Mensah"
    assert named["Speaker 5"]["name"] == "Sai Reddy"


def test_small_tile_misreadings_do_not_name_the_wrong_person():
    """The real bug found while testing: "Kuikarni" (a misread small-tile label, seen only while
    others talk) was taken as Precious's name."""
    obs, fr = speaker_view({"Speaker 1": 0, "Speaker 2": 1, "Speaker 3": 2, "Speaker 4": 3},
                           misread={"Aarya Kulkarni": "Aarya Kuikarni", "Sai Reddy": "Sai Redy"})
    named = {k: v["name"] for k, v in S._match_names(obs, fr).items()}
    assert named["Speaker 1"] == "Precious Mensah"
    assert named["Speaker 2"] == "Aarya Kulkarni" and named["Speaker 4"] == "Sai Reddy"  # big-tile spelling


def test_gallery_view_with_every_name_always_visible_names_nobody():
    """When every name is always on screen, the screen says nothing about who talks: better no
    name (the user types it) than a wrong one."""
    obs, fr, f = [], Counter(), 0
    for label in ["Speaker 1", "Speaker 2", "Speaker 3"]:
        for _ in range(5):
            fr[label] += 1
            obs += [(f, label, PEOPLE[p], cell) for p, cell in zip(range(3), (THUMB1, THUMB2, THUMB3))]
            f += 1
    assert S._match_names(obs, fr) == {}


def test_lines_get_the_speaker_who_talks_most_during_them():
    turns = [S.Turn(0, 5, "Speaker 1"), S.Turn(5.2, 9, "Speaker 2"), S.Turn(9.5, 12, "Speaker 1")]
    # line 3 falls in the pause between turns: it goes to the nearest turn (Speaker 1 at 9.5 s)
    segs = [Segment(1, 0.2, 4.8, "a"), Segment(2, 4.6, 8.5, "b"), Segment(3, 9.4, 9.45, "c"), Segment(4, None, None, "d")]
    S.label_segments(segs, turns)
    assert [s.speaker for s in segs] == ["Speaker 1", "Speaker 2", "Speaker 1", None]


def test_names_can_be_changed_again():
    segs = [Segment(1, 0, 1, "I'll do it.", speaker="Speaker 1")]
    S.apply_names(segs, {"Speaker 1": "Precious Mensah"})
    assert segs[0].speaker == "Precious Mensah" and segs[0].speaker_id == "Speaker 1"
    S.apply_names(segs, {"Speaker 1": "Precious M."})
    assert segs[0].speaker == "Precious M."
    S.apply_names(segs, {"Speaker 1": ""})            # cleared: back to the voice label
    assert segs[0].speaker == "Speaker 1"


def test_speaker_reaches_the_model_and_the_downloads():
    segs = [Segment(9, 84.0, 86.0, "Will do.", speaker="Sai Reddy")]
    assert P.lines_for_prompt(segs) == ["[9] (01:24) Sai Reddy: Will do."]
    assert said(segs[0]) == "Sai Reddy: Will do." and "Sai Reddy: Will do." in transcript_txt(segs)
    assert "the speaker of that line" in P.SYSTEM_PROMPT


def test_pronoun_owners_become_unassigned():
    """Seen with the sample transcript (no speaker names): "I'll prepare..." came back with
    owner "I"."""
    segs = [Segment(1, 0, 1, "I'll prepare the summary.")]
    raw = {"title": "t", "summary": "s", "key_points": [], "action_items": [
        {"owner": o, "task": "x", "due": "Monday", "segments": [1]} for o in ["I", " we ", "", "Precious"]]}
    assert [a["owner"] for a in P.validate_notes(raw, segs)["action_items"]] == \
        ["Unassigned", "Unassigned", "Unassigned", "Precious"]


def test_prompt_examples_contain_no_real_looking_names():
    """A surname in the prompt's examples ("Sai Reddy") leaked into notes for a transcript that
    only said "Sai". Examples must use placeholders."""
    for name in ["Reddy", "Mensah", "Kulkarni", "Iyer"]:
        assert name not in P.SYSTEM_PROMPT


def test_segments_survive_json_with_speakers():
    segs = [Segment(1, 0.0, 1.0, "Hi.", "Sai Reddy", "Speaker 2")]
    back = J.segments_from_json(J.result_to_json({"segments": segs, "notes": {}, "source": "", "timings": {},
                                                  "stats": {}, "speakers": []})["segments"])
    assert back == segs and J.segments_from_json([[1, 0.0, 1.0, "Hi."]])[0].speaker is None


# ---------------------------------------------------------------- the processing flow
def fake_ollama(notes):
    def post(url, json=None, timeout=None, stream=False):  # noqa: A002
        if url.endswith("/api/embed"):
            return FakeResp({"embeddings": [bag_of_words(t) for t in json["input"]]})
        body = __import__("json").dumps(notes)
        return FakeStream([body[i:i + 9] for i in range(0, len(body), 9)])
    return post


def fake_whisper():
    seg = lambda a, b, t: types.SimpleNamespace(start=a, end=b, text=t)  # noqa: E731

    class FakeModel:
        def __init__(self, *a, **kw):
            pass

        def transcribe(self, audio, **kw):
            return iter([seg(0, 3, "Can you send the summary?"), seg(3.5, 5, "Yes, I'll send it Monday.")]), \
                types.SimpleNamespace(language="en", duration=5.0)
    return types.SimpleNamespace(WhisperModel=FakeModel)


def test_video_run_labels_speakers_before_writing_notes():
    seen_prompt = {}
    notes = {"title": "t", "summary": "s", "key_points": [],
             "action_items": [{"owner": "Precious Mensah", "task": "Send the summary", "due": "Monday", "segments": [2]}]}
    post = fake_ollama(notes)

    def spy(url, json=None, timeout=None, stream=False):  # noqa: A002
        if url.endswith("/api/chat"):
            seen_prompt["user"] = json["messages"][1]["content"]
        return post(url, json=json, timeout=timeout, stream=stream)

    def fake_identify(video, segments, on_progress=None):
        segments[0].speaker, segments[1].speaker = "Speaker 1", "Speaker 2"
        S.apply_names(segments, {"Speaker 1": "Sai Reddy", "Speaker 2": "Precious Mensah"})
        on_progress(1.0, "done")
        return {"speakers": [{"label": "Speaker 1", "name": "Sai Reddy", "source": "screen", "lines": 1},
                             {"label": "Speaker 2", "name": "Precious Mensah", "source": "screen", "lines": 1}],
                "turns": 2}

    rates = J.new_rates()
    with mock.patch.dict(sys.modules, {"faster_whisper": fake_whisper()}), \
            mock.patch.object(S, "available", return_value=(True, "")), \
            mock.patch.object(S, "identify", side_effect=fake_identify), \
            mock.patch.object(P.requests, "post", side_effect=spy), \
            mock.patch.object(P.requests, "get", return_value=FakeResp({"models": []})):
        tracker = J.make_tracker("video", 10, rates)
        res = J.process({"kind": "video", "name": "m.mp4", "data": b"x"}, tracker, rates, P.DEFAULT_LLM, "small")
    assert [s["key"] for s in tracker.snapshot()["stages"]] == ["input", "speakers", "index", "notes"]
    assert "Precious Mensah: Yes, I'll send it Monday." in seen_prompt["user"]   # the model sees who spoke
    assert res["speakers"][1]["name"] == "Precious Mensah" and "speakers" in res["timings"]
    assert res["notes"]["action_items"][0]["owner"] == "Precious Mensah"


def test_video_run_still_works_without_speaker_models():
    notes = {"title": "t", "summary": "s", "key_points": [], "action_items": []}
    rates = J.new_rates()
    with mock.patch.dict(sys.modules, {"faster_whisper": fake_whisper()}), \
            mock.patch.object(S, "available", return_value=(False, "Speaker models not downloaded")), \
            mock.patch.object(P.requests, "post", side_effect=fake_ollama(notes)), \
            mock.patch.object(P.requests, "get", return_value=FakeResp({"models": []})):
        tracker = J.make_tracker("video", 10, rates)
        res = J.process({"kind": "video", "name": "m.mp4", "data": b"x"}, tracker, rates, P.DEFAULT_LLM, "small")
    assert res["speakers"] is None and "not downloaded" in J.notices(res["stats"])["speakers_off"]
    assert "speakers" not in [s["key"] for s in tracker.snapshot()["stages"]]


def test_identification_failure_never_loses_the_meeting():
    notes = {"title": "t", "summary": "s", "key_points": [], "action_items": []}
    rates = J.new_rates()
    with mock.patch.dict(sys.modules, {"faster_whisper": fake_whisper()}), \
            mock.patch.object(S, "available", return_value=(True, "")), \
            mock.patch.object(S, "identify", side_effect=RuntimeError("bad frame")), \
            mock.patch.object(P.requests, "post", side_effect=fake_ollama(notes)), \
            mock.patch.object(P.requests, "get", return_value=FakeResp({"models": []})):
        res = J.process({"kind": "video", "name": "m.mp4", "data": b"x"}, J.make_tracker("video", 10, rates),
                        rates, P.DEFAULT_LLM, "small")
    assert res["speakers"] is None and "bad frame" in res["stats"]["speakers_off"]
    assert res["segments"] and all(s.speaker is None for s in res["segments"])


def test_renaming_rewrites_owners_with_the_new_names():
    segs = [Segment(1, 0, 3, "Can you send it?", "Speaker 1", "Speaker 1"),
            Segment(2, 3, 5, "I'll send it Monday.", "Speaker 2", "Speaker 2")]
    prompts = []
    notes = {"title": "t", "summary": "s", "key_points": [],
             "action_items": [{"owner": "Precious", "task": "Send it", "due": "Monday", "segments": [2]}]}
    post = fake_ollama(notes)

    def spy(url, json=None, timeout=None, stream=False):  # noqa: A002
        if url.endswith("/api/chat"):
            prompts.append(json["messages"][1]["content"])
        return post(url, json=json, timeout=timeout, stream=stream)

    from progress import Stage, Tracker
    tracker = Tracker([Stage("index", "i", 1), Stage("notes", "n", 1)])
    with mock.patch.object(P.requests, "post", side_effect=spy):
        res = J.rewrite(segs, {"Speaker 2": "Precious"}, tracker, J.new_rates(), P.DEFAULT_LLM)
    assert "Precious: I'll send it Monday." in prompts[0] and "Speaker 1: Can you send it?" in prompts[0]
    assert res["notes"]["action_items"][0]["owner"] == "Precious" and tracker.snapshot()["done"]


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
