# Demo-day checklist

## A few days before
- [ ] Run the setup script on the demo laptop **and** a backup laptop.
- [ ] Record the sample meeting (`sample/sample_meeting_script.md`) as a 1.5–2.5 minute `.mp4`. Copy it to both laptops.
- [ ] Run the full demo offline once and **time it**. Write the times down:
      Transcription: ____ s   Key points: ____ s
- [ ] If the total is over 90 seconds, switch to Whisper `base` and/or `llama3.2:1b` and retime.
- [ ] Compare the output with the expected key points in the script. Note anything it misses so it isn't a surprise.
- [ ] Screen-record one complete successful run as a backup video.

## 30 minutes before
- [ ] Plug the laptop in (laptops slow down on battery). Close other heavy apps.
- [ ] Open Ollama, start the app, and click **Warm up model**.
- [ ] Turn on airplane mode, then click **Check: is this laptop offline?** It should show green.
- [ ] Open Activity Monitor (Mac) or Task Manager → Performance (Windows), ready to show.
- [ ] Open a terminal with the network command ready: `lsof -i -P | grep -iE "ollama|python"`.
- [ ] Set the browser zoom to about 125% so the room can read it.

## Live script (about 3 minutes)
| Time | Presenter | Action | What to say |
|---|---|---|---|
| 0:00 | Precious | Show the app | "This is a confidential meeting. Most companies won't let it near cloud AI." |
| 0:20 | Sai | Turn on airplane mode and click the offline check | "No internet. Everything you're about to see runs on this laptop." |
| 0:35 | Aarya | Upload the video and click **Transcribe and summarize** | "Whisper is turning speech into text right here; watch the CPU." |
| 1:30 | Niharika | Show key points and action items | "Llama 3.2 found the decisions and who owns each task." |
| 2:15 | Niharika | Click a timestamp, and the video jumps | "Every point links to the exact moment, so you can check it." |
| 2:35 | Sai | Show the privacy receipt and the terminal | "Zero external connections. Nothing left this laptop." |
| 2:50 | Precious | Close | "Private you can prove, and built for follow-through." |

## If something goes wrong
- **Transcription is slow:** click **Use sample transcript** and keep talking. The key-point step is the impressive part.
- **The model errors out:** check that the Ollama icon is running, then click **Warm up model** again.
- **The laptop fails completely:** switch to the backup laptop, or play the backup screen recording.
