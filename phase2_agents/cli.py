# agent/cli.py

import sys

from .agent import run_agent


def main() -> None:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Question: ").strip()

    if not question:
        print("No question provided.")
        return

    try:
        answer = run_agent(question)
    except KeyboardInterrupt:
        print("\nCancelled.")
        return
    except Exception as exc:
        print(f"\nError: {exc}")
        return

    print("\nAnswer:\n")
    print(answer)


if __name__ == "__main__":
    main()