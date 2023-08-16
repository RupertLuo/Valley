# Download Video from VATEX and Jukinmedia
first, cd into the Crawler folder
```shell
cd Crawler
```
## VATEX
run the following script to generate the download cmd string to a file
```shell
python download_cmd_generation.py --video_path [VIDEO SAVE FOLDER] --ann_file [VATEX ANNOTATION FILE PATH] --output_filename [CMD FILE OUTPUT PATH]
```

Then run following script to download the video from Youtube.(**PS: This script calls the youtube video download tool [yt-dlp](https://github.com/yt-dlp/yt-dlp)**. You need to install it first)

```shell
python parallel_download_VATEX.py --num_process 32 --cmd_file [THE FILE YOU GENERATE FROM LAST STEP]
```

## Jukin Media
Run the following script to generate the video_id and video information from jukinmedia to a file.
```shell
python download_jukin_video_id.py --savefolder [THE VIDEO INFO SAVEFOLDER]
```

Run the folowing script to download the video
```shell
python parallel_download_jukin.py --save_dir [VIDEO SAVE FOLDER] --input_file [THE FILE YOU GENERATE FROM LAST STEP] --num_process 5
```
