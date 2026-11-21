# Demo-day checklist

## A few days before
- [ ] Run the setup script on the demo laptop (Sai's Windows laptop) and on Precious's Mac.
- [ ] Record the sample meeting (`sample/sample_meeting_script.md`) as a 1.5–2.5 minute `.mp4`.
      The backup inputs are already in `sample/`: `sample_meeting.vtt` (transcript) and `sample_notes.md` (notes).
- [ ] Check the sidebar says **NVIDIA GPU available**, and that a video run shows "Whisper on GPU".
      If it says CPU, rerun `setup_windows.ps1` or update the NVIDIA driver; the demo still works on CPU.
- [ ] Run the full demo offline once with each input and **time it**:
      Video: ____ s   Transcript file: ____ s   One question: ____ s
- [ ] If a run is over 90 seconds, switch to Whisper `base` and/or `qwen3:1.7b` and time it again.
- [ ] Compare the bullet points with the expected ones in the script.
- [ ] Ask these 5 questions and check the answers cite the right lines:
      1. "When does the Riverbend pilot start?"
      2. "Which model did we choose, and why?"
      3. "Who is building the progress bar, and by when?"
      4. "What did Riverbend ask for?"
      5. "What is the budget for the pilot?" (should say **Not in the transcript.**)
- [ ] Screen-record one complete successful run as a backup video.

## 30 minutes before
- [ ] Plug the laptop in (laptops slow down on battery). Close other heavy apps.
- [ ] Open Ollama, start the app, and click **Warm up model**.
- [ ] Turn on airplane mode, then click **Check: is this laptop offline?** It should show green.
- [ ] Open Task Manager → Performance (Windows) or Activity Monitor (Mac), ready to show.
- [ ] Open a terminal with the network check ready: `netstat -ano | findstr 11434` (Windows)
      or `lsof -i -P | grep -iE "ollama|python"` (Mac).
- [ ] Set the browser zoom to about 125% so the room can read it.

## Live script (about 3.5 minutes)
| Time | Presenter | Action | What to say |
|---|---|---|---|
| 0:00 | Precious | Show the app | "This is a confidential meeting. Most firms won't let it near cloud AI." |
| 0:20 | Sai | Turn on airplane mode and click the offline check | "No internet. Everything you see runs on this laptop." |
| 0:35 | Aarya | Upload the video and click **Process meeting** | "Whisper turns speech into text right here. Watch the CPU." |
| 1:30 | Niharika | Show **Bullet points**; click a timestamp so the video jumps | "Each point links to the exact moment it came from." |
| 2:00 | Niharika | **Ask the transcript**: "Who is building the progress bar?" then "What is the budget?" | "It answers with its sources, and says when the answer isn't there." |
| 2:45 | Aarya | **Transcript** tab: click **Download** | "The transcript is optional and stays on your machine." |
| 3:00 | Sai | Show the privacy receipt and the terminal | "Zero external connections. Nothing left this laptop." |
| 3:20 | Precious | Close | "Private you can prove, and built for follow-through." |

## If something goes wrong
- **Transcription is slow:** upload `sample/sample_meeting.vtt` instead. All three outputs still work.
- **The model errors out:** check that Ollama is running, then click **Warm up model** again.
- **The laptop fails completely:** switch to Precious's Mac, or play the backup screen recording.
