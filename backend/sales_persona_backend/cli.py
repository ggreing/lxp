"""Tiny command‑line demo for backend testing."""

import argparse, sys
from .ai import SalesPersonaAI, generate_first_greeting
from .personas import SCENARIOS


def main():
    ap = argparse.ArgumentParser(description="CLI simulation for Sales Persona AI backend")
    ap.add_argument("--scenario", default="intro_meeting", choices=SCENARIOS.keys())
    args = ap.parse_args()

    engine = SalesPersonaAI(scenario=args.scenario)
    greeting = generate_first_greeting(engine.persona, args.scenario)
    print("AI:", greeting)

    try:
        while True:
            seller_msg = input("판매자> ").strip()
            if not seller_msg:
                continue
            for chunk in engine.stream_response(seller_msg):
                pass  # we just need the final response in CLI
            print("AI:", engine.history[-1].split(": ",1)[1])
            autoclose, reason = engine.maybe_autoclose()
            if autoclose:
                print(f"[세션 자동 종료] – reason: {reason}")
                break
    except KeyboardInterrupt:
        print("\n[종료]")

if __name__ == "__main__":
    main()