"""Command-line interface: `atlaskit chat|translate|transcribe|detect|health`."""
from __future__ import annotations

import argparse
import json
import sys

from .client import AtlasKit, AtlasKitError


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="atlaskit", description="Talk to N-ATLaS through an AtlasKit gateway")
    p.add_argument("--url", help="Gateway URL (or ATLASKIT_BASE_URL)")
    p.add_argument("--key", help="API key (or ATLASKIT_API_KEY)")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chat", help="Chat with N-ATLaS (omit the prompt for an interactive session)")
    c.add_argument("prompt", nargs="?")
    c.add_argument("-l", "--language", help="Reply language: en, yo, ha, ig")
    c.add_argument("-s", "--system")

    t = sub.add_parser("translate")
    t.add_argument("text")
    t.add_argument("-t", "--to", required=True)
    t.add_argument("-f", "--from", dest="source")

    a = sub.add_parser("transcribe")
    a.add_argument("audio")
    a.add_argument("-l", "--language", required=True)

    d = sub.add_parser("detect")
    d.add_argument("text")
    sub.add_parser("health")
    sub.add_parser("stats")

    args = p.parse_args(argv)
    atlas = AtlasKit(args.url, args.key)
    try:
        if args.cmd == "chat":
            if args.prompt:
                for delta in atlas.stream(args.prompt, language=args.language, system=args.system):
                    print(delta, end="", flush=True)
                print()
            else:
                convo = atlas.conversation(language=args.language, system=args.system)
                print("N-ATLaS chat — Ctrl+D to exit")
                while True:
                    try:
                        text = input("you> ").strip()
                    except EOFError:
                        print()
                        break
                    if text:
                        print("n-atlas> ", end="")
                        for delta in convo.stream(text):
                            print(delta, end="", flush=True)
                        print()
        elif args.cmd == "translate":
            print(atlas.translate(args.text, args.to, args.source))
        elif args.cmd == "transcribe":
            print(atlas.transcribe(args.audio, args.language)["text"])
        elif args.cmd == "detect":
            print(json.dumps(atlas.detect_language(args.text), ensure_ascii=False))
        elif args.cmd == "health":
            print(json.dumps(atlas.health(), indent=2))
        elif args.cmd == "stats":
            print(json.dumps(atlas.stats(), indent=2))
    except AtlasKitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
