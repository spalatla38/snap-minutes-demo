# Weekly SNAP Assist sync (notes)

Attendees: Precious, Aarya, Niharika, Sai

## Riverbend Legal pilot
- Pilot with Riverbend Legal starts December 15.
- Riverbend mostly sends scanned contracts; OCR accuracy drops to about 80% on tilted scans.

## Model choice
- Tested the 4B model against the 1.7B model on 20 sample contracts: 18 of 20 correct versus 13 of 20.
- Decision: ship the 4B model for the pilot.
- The 4B model takes about 20 seconds per document on older laptops.

## Action items
- [ ] Aarya: add automatic page straightening before OCR by December 5.
- [ ] Sai: build a progress bar and background processing by December 10.
- [ ] Niharika: write a test set of tricky contracts by Friday.
- [ ] Precious: send a one-page privacy summary to Riverbend's IT lead next Monday.
