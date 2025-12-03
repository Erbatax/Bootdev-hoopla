import argparse

from lib.describe_image import describe_image_command


def main():
    parser = argparse.ArgumentParser(description="Image Description CLI")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image file to be described",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query to describe the image",
    )

    args = parser.parse_args()
    result = describe_image_command(args.image, args.query)


if __name__ == "__main__":
    main()
