# Sample meeting script: record this as your demo video

A made-up 2-minute team meeting. Record it on a phone or laptop webcam with all four
team members (or one person reading every part). Use only this fictional content:
never demo with real confidential data.

**Tips:** speak clearly, one person at a time, in a quiet room. Aim for 1.5 to 2.5 minutes.
Save as `.mp4` or `.mov`.

---

**Precious (Product Manager):** Okay, let's get started. This is our weekly SNAP Assist sync. The main topic today is the pilot with Riverbend Legal, which starts on December fifteenth.

**Aarya (Data Engineer):** On the data side, the ingestion pipeline now handles scanned PDFs with OCR. The open problem is that some contract scans are tilted, and accuracy drops to about eighty percent on those.

**Precious:** That matters, because Riverbend mostly sends scanned contracts. Aarya, can you add automatic page straightening before OCR?

**Aarya:** Yes. I'll have it done by December fifth.

**Niharika (AI/ML Engineer):** On the model side, we tested Llama 3.2 3B against the 1B model on our twenty sample contracts. The 3B model got the right owner and deadline in eighteen of twenty. The 1B model only got thirteen. So I recommend we ship the 3B model.

**Precious:** Agreed. Decision: we go with the 3B model for the pilot.

**Sai (Application Developer):** One concern: the 3B model takes about twenty seconds per document on older laptops. I'd like to add a progress bar and let people keep working while it runs.

**Precious:** Good idea. Sai, please build the progress bar and background processing before the pilot starts.

**Sai:** Will do. I'll target December tenth so we have time to test.

**Niharika:** I'll also write a short test set of tricky contracts so we can catch mistakes before Riverbend does. I can finish that this Friday.

**Precious:** Great. Last item: Riverbend asked for written proof that no data leaves the device. I'll prepare a one-page privacy summary with our network test results and send it to their IT lead next Monday.

**Precious:** So to recap: Aarya does page straightening by December fifth, Sai does the progress bar by December tenth, Niharika writes the test set by Friday, and I send the privacy summary on Monday. Thanks, everyone.

---

## Expected output (use this to check the demo)

**Key points**
- The Riverbend Legal pilot starts on December 15.
- OCR accuracy drops to about 80% on tilted contract scans.
- Llama 3.2 3B found the right owner and deadline in 18 of 20 sample contracts, versus 13 for the 1B model.
- Decision: the pilot will use the 3B model.
- The 3B model takes about 20 seconds per document on older laptops.
- Riverbend wants written proof that no data leaves the device.

**Action items**
| Owner | Task | Due |
|---|---|---|
| Aarya | Add automatic page straightening before OCR | December 5 |
| Sai | Build a progress bar and background processing | December 10 |
| Niharika | Write a test set of tricky contracts | This Friday |
| Precious | Send a one-page privacy summary to Riverbend's IT lead | Next Monday |
