# Contributing to AtlasKit

Thank you for helping make N-ATLaS easier to build on. AtlasKit welcomes contributions from developers, linguists, researchers and anyone who uses Yoruba, Hausa, Igbo or Nigerian English in their work.

By taking part you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report a bug** or something confusing: open an issue with the *Bug report* template.
- **Report a language-quality problem:** an answer, translation or transcript that is wrong in Yoruba, Hausa, Igbo or Nigerian English. Use the *Language quality* template. These reports guide our evaluation sets and the next fine-tuned adapters.
- **Improve the docs**, including translations of the docs into Yoruba, Hausa and Igbo (on our roadmap).
- **Add features** to the gateway, the SDKs, the CLI or the playground.
- **Add evaluation data or fine-tuning recipes:** new test sets, tasks or adapters, with their benchmark reports.

For anything larger than a small fix, please open an issue first so we can agree on the approach before you write the code.

## Ground rules

These keep AtlasKit trustworthy for the people who build on it:

1. **N-ATLaS only.** The gateway serves N-ATLaS and LoRA adapters trained on it, and nothing else. Pull requests that route requests to another foundation model will not be accepted.
2. **No secrets or weights in the repository.** Never commit API keys, `.env` files, Hugging Face tokens or model weights. Each deployer downloads N-ATLaS under their own accepted licence.
3. **Privacy by default.** The usage log must never store prompts, completions or audio. Keep test fixtures free of personal data.
4. **Respect the model licence.** N-ATLaS and its ASR models are released by NCAIR and Awarri under the Awarri Open-Source Research and Innovation License. Keep attribution intact and don't remove licence notices.

## Development setup

You don't need a GPU for most work. The test suite and the local dev server use a mock model.

```bash
git clone https://github.com/Lawalshalom/atlaskit && cd atlaskit
python -m venv .venv && source .venv/bin/activate          # Python 3.10+
pip install -e gateway -e sdk/python pytest

python examples/dev_server.py     # gateway + playground on http://localhost:8080 against a mock model
pytest tests -q                   # the full suite (28 tests); includes the JS SDK tests when Node 18+ is installed
```

The JavaScript SDK has its own test command: `cd sdk/js && npm test`.

For work that needs the real model (fine-tuning, benchmarks, ASR quality), follow [docs/DEPLOY_RUNPOD.md](docs/DEPLOY_RUNPOD.md), [finetune/README.md](finetune/README.md) and [eval/README.md](eval/README.md).

## Where things live

| Area | Files | Notes |
|---|---|---|
| Request shaping and validation | `gateway/atlaskit_gateway/core.py` | Pure Python, no I/O. Most logic changes belong here, with unit tests in `tests/test_core.py`. |
| HTTP routes | `gateway/atlaskit_gateway/app.py` | Keep it thin. End-to-end tests run against the mock model in `tests/test_integration.py`. |
| Speech | `gateway/atlaskit_gateway/asr.py` | Audio is decoded to 16 kHz mono before the N-ATLaS ASR models see it. |
| Python SDK and CLI | `sdk/python/atlaskit/` | Standard library only. |
| JavaScript SDK | `sdk/js/src/index.js` | `fetch` only; runs in Node 18+, Deno, Bun and browsers. |
| Playground | `playground/index.html` | One file, no build step. |
| Fine-tuning | `finetune/` | Training must render examples exactly as the gateway does at inference time. |
| Evaluation | `eval/run_eval.py` | Reports go in `eval/results/<date>/` with the method and raw outputs. |

## Coding guidelines

- **Match the code around you:** naming, comment density and idiom.
- **Keep the SDKs dependency-free and in step.** A new client method goes into both the Python and the JavaScript SDK, with the same name and arguments in each language's style.
- **Keep the OpenAI wire format intact.** N-ATLaS-specific behaviour goes in optional fields (such as `atlas_language`) or under `/v1/atlas/*`, never in breaking changes to the standard routes.
- **Test every behaviour change**, and update the docs in the same pull request.
- **Write Nigerian languages correctly.** Use proper tone marks and letters (for example ẹ, ọ, ṣ in Yoruba; ɓ, ɗ, ƙ in Hausa; ị, ọ, ụ in Igbo). Where possible, ask a native speaker to review prompts, examples and evaluation data.
- **Cite data sources.** For datasets, give the source and licence, and never mix test sentences into training data.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Make a focused change. Several small pull requests are easier to review than one large one.
3. Run `pytest tests -q` and make sure it passes. CI runs the same suite on every pull request.
4. Open the pull request and fill in the template: what changed, why, and how you tested it. Link the issue it addresses.
5. A maintainer will review it. We aim to respond within a week.

Write commit messages in the imperative mood ("Add Igbo summarisation example") and explain *why* in the body when it isn't obvious.

## Reporting security issues

Please don't open a public issue for a vulnerability. Report it privately through the repository's **Security** tab (**Report a vulnerability**), and we'll respond there.

## Licence

AtlasKit is licensed under [Apache-2.0](LICENSE). Under section 5 of that licence, any contribution you submit is provided under the same licence, unless you state otherwise.
