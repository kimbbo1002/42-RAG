import fire
from .cli import CLI


def main() -> None:
    try:
        fire.Fire(CLI)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
