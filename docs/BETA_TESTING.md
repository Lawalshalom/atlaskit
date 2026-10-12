# Beta testing guide

The challenge requires **at least 2 external beta testers** for Developer Infrastructure submissions. External means developers outside the Halo Digital team. This guide is what you send each tester, plus how to collect the evidence.

## Message to send each tester

> Hi! We've built **AtlasKit**, an SDK and API for N-ATLaS (Nigeria's Yoruba/Hausa/Igbo/English LLM). Could you spend about 30 minutes trying it today and fill a short form?
>
> - Playground: `https://<POD_ID>-8080.proxy.runpod.net` (paste your API key at the top)
> - Your API key: `<KEY>`
> - Code: https://github.com/Lawalshalom/atlaskit (see README → Quick start)
>
> Please do these three tasks:
> 1. **Playground:** chat in at least two Nigerian languages, try "Compare languages", and rate a few answers 👍/👎.
> 2. **SDK:** install the Python or JS SDK and write a small script that calls `chat` (or `stream`) and `translate`. If you have a voice note, try `transcribe`.
> 3. **Drop-in:** if you already use the OpenAI SDK anywhere, point `base_url` at the gateway and run it with `model="n-atlas"`.
>
> Then fill the form: `<Google Form link>`. Thank you!

Give each tester **their own API key**, so usage in `/v1/atlas/stats` can be counted per tester (keys are hashed in the log).

## Feedback form questions (Google Forms / Typeform)

`docs/beta_feedback_form.gs` builds this form, plus a linked responses sheet, in one click. Paste it into https://script.new and click Run. The script turns off Google Workspace's organisation-only restriction, so external testers can respond. It also asks for a tester ID, which links each response to that tester's API usage, and asks whether the respondent is on the Halo Digital team.

1. Name, role/company, and GitHub or LinkedIn (needed as evidence they are real, external developers)
2. Which did you use? (Playground / Python SDK / JS SDK / OpenAI drop-in / CLI)
3. How long did it take from reading the README to your first successful N-ATLaS response? (<5 min / 5–15 / 15–30 / >30 / didn't succeed)
4. Which languages did you try? (Yoruba / Hausa / Igbo / Nigerian English)
5. For the languages you speak: how good were N-ATLaS's answers? (1–5)
6. How easy was the SDK to use compared with calling the model yourself? (1–5)
7. What broke or confused you?
8. What would you build with this?
9. Would you use AtlasKit in a real project? (Yes / Maybe / No)
10. May we quote your feedback in our NAIC submission? (Yes / No)

## Evidence to collect for the submission

- [ ] Form responses exported (CSV or screenshots), from at least 2 external testers
- [ ] `curl -H "Authorization: Bearer $KEY" $URL/v1/atlas/stats`, saved as JSON after testing: request counts, unique clients, languages, 👍/👎
- [ ] A screenshot of each tester's script or terminal output, if they will share one
- [ ] Issues they raised, and what you fixed (shows iteration)
- [ ] The eval report: `python eval/run_eval.py --flores yo,ha,ig --limit 100 ...`, which counts as a live benchmark on real data
