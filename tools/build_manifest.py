import argparse
from lahn_driveclip.data.builders import (
    build_bddx_manifest, build_talk2car_manifest,
    build_localization_csv_manifest,
)

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="kind", required=True)

    p = sub.add_parser("bddx")
    p.add_argument("--csv", required=True)
    p.add_argument("--split-file", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--frame-template", default="{video_id}.jpg")

    p = sub.add_parser("talk2car")
    p.add_argument("--json", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", required=True)

    p = sub.add_parser("localization_csv")
    p.add_argument("--csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--image-root")

    args = parser.parse_args()
    if args.kind == "bddx":
        n = build_bddx_manifest(args.csv, args.split_file, args.image_root,
                                args.output, args.split, args.frame_template)
    elif args.kind == "talk2car":
        n = build_talk2car_manifest(args.json, args.image_root,
                                    args.output, args.split)
    else:
        n = build_localization_csv_manifest(
            args.csv, args.output, args.dataset, args.split, args.image_root
        )
    print(f"Wrote {n} samples to {args.output}")

if __name__ == "__main__":
    main()
