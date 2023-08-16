from argparse import ArgumentParser
from multiprocessing import Pool
import subprocess
from tqdm import tqdm
def download(cmd):
    try:
       subprocess.run(cmd, shell=True, capture_output=True)
    except:
        pass
def main(args):
    cmd_list = open(args.cmd_file,'r').readlines()
    pbar = tqdm(total=len(cmd_list))
    pbar.set_description('download')
    update = lambda *args: pbar.update()
    p = Pool(int(args.num_process))  # 指定进程池中的进程数
    for i, cmd in enumerate(cmd_list):
        p.apply_async(download, args = (cmd.strip(),), callback=update)

    print('Waiting for all subprocesses done...')
    p.close()
    p.join() 
    print('All subprocesses done.')
    
if __name__ == "__main__":
    parser = ArgumentParser(description="Script to parallel downloads videos")
    parser.add_argument("--num_process", default=32,)
    parser.add_argument("--cmd_file", default='./VATEX/cmd_list.txt',)
    args = parser.parse_args()
    main(args)