''' usage: generate command script to download youtube video
'''
from argparse import ArgumentParser
import glob
import json
import os

def crosscheck_videos(video_path, ann_file):
    # Get existing videos
    existing_vids = glob.glob("%s/*.mp4" % video_path)
    for idx, vid in enumerate(existing_vids):
        basename = os.path.basename(vid).split(".mp4")[0]
        if len(basename) == 13:
            existing_vids[idx] = basename[2:]
        elif len(basename) == 11:
            existing_vids[idx] = basename
        else:
            raise RuntimeError("Unknown filename format: %s", vid)
    # Read an get video IDs from annotation file
    with open(ann_file, "r") as fobj:
        anet_v_1_0 = json.load(fobj)
    if 'VATEX' in video_path:
        all_vids = list(set(['_'.join(item['videoID'].split('_')[:-2]) for item in anet_v_1_0]))
    else:
        raise ValueError('Not VATEX form data, you need to cumtomize the code.')
    non_existing_videos = []
    for vid in all_vids:
        if vid in existing_vids:
            continue
        else:
            non_existing_videos.append(vid)
    return non_existing_videos

def main(args):
    non_existing_videos = crosscheck_videos(args.video_path, args.ann_file)
    filename = os.path.join(args.video_path, "v_%s.mp4")
    cmd_base = "yt-dlp --merge-output-format mp4 "
    cmd_base += '"https://www.youtube.com/watch?v=%s" '
    cmd_base += '-o "%s" ' % filename 
    cmd_base += '|| mv "%s.part" "%s"' % (filename,filename)
    with open(args.output_filename, "w") as fobj:
        for vid in non_existing_videos:
            cmd = cmd_base % (vid, vid, vid, vid)
            fobj.write("%s\n" % cmd)

if __name__ == "__main__":
    parser = ArgumentParser(description="Script to double check video content.")
    parser.add_argument("--video_path", required=True, help="Where are located the videos? (Full path)")
    parser.add_argument("--ann_file", required=True, help="Where is the annotation file?")
    parser.add_argument("--output_filename", default='./VATEX/cmd_list.txt',required=True, help="Output script location.")
    args = parser.parse_args()
    main(args)
