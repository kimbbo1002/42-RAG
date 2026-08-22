import fire
from .cli import CLI


def main() -> None:
    """Main entry point for the command-line interface."""
    try:
        fire.Fire(CLI)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
