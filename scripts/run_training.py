import logging

import fire as fire


def main():
    ...


if __name__ == "__main__":
    try:
        fire.Fire(main)
    except Exception as e:
        logging.error(f"An unexpected error occured {e}", exc_info=True)
