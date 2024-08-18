import os
import subprocess
import time
import json
import cv2
import numpy as np
import argparse
from multiprocessing import Pool, current_process


def command(cmd, cwd=None, timeout=3):
    p = subprocess.Popen(cmd, stderr=subprocess.PIPE, cwd=cwd, stdout=subprocess.PIPE, universal_newlines=True)
    t_beginning = time.time()
    while True:
        if p.poll() is not None:
            break
        seconds_passed = time.time() - t_beginning
        if timeout and seconds_passed > timeout:
            p.terminate()
            return None, 'timeout'
        time.sleep(0.01)
    try:
        out, err = p.communicate()
    except:
        return None, 'skip'
    return out, err


def render_ground_truth(item, base_tmp_dir, output_dir, image_width, image_height, dpi):
    # Create a unique temporary directory for the current process
    process_id = current_process().pid  # Gets the process ID
    tmp_dir = os.path.join(base_tmp_dir, f"tmp_{process_id}")
    os.makedirs(tmp_dir, exist_ok=True)  # Ensure the directory exists
    
    filename = 'infer'
    tex_path = f'{tmp_dir}/{filename}.tex'
    if os.path.exists(tex_path):
        os.remove(tex_path)
    
    with open(tex_path, 'w') as wp:
        latex = item['latex'].strip()
        wp.write('\\documentclass[border=1pt]{standalone}\n\\usepackage{amsmath}\n\\begin{document}\n$' + latex + '$\n\\end{document}')
    
    cmd = ['pdflatex', f'{filename}.tex']
    out, err = command(cmd, cwd=tmp_dir)
    
    if out is None:
        im = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
    else:
        pdf_path = f'{tmp_dir}/{filename}.pdf'
        png_path = f'{tmp_dir}/{filename}.png'
        
        command(['convert', '-density', f'{dpi}', pdf_path, '-background', 'white', '-alpha', 'remove', png_path], cwd=tmp_dir)
        
        im = cv2.imread(png_path)
        width, height = im.shape[1], im.shape[0]
        
        if width > image_width or height > image_height:
            command(['convert', '-density', f'{dpi}', f'{filename}.pdf', '-resize', f'{image_width}x{image_height}', '-background', 'white', 
                    '-gravity', 'NorthWest', '-extent', f'{image_width}x{image_height}', '-alpha', 'remove', f'{filename}.png'], cwd=tmp_dir)
            im = cv2.imread(f'{tmp_dir}/{filename}.png')
    
    cv2.imwrite(f'{output_dir}/{item["image"]}', im)

    # Clean up the temporary directory after processing to avoid filling up the disk
    for file_name in os.listdir(tmp_dir):
        file_path = os.path.join(tmp_dir, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
    os.rmdir(tmp_dir)


def worker(data_chunk, base_tmp_dir, output_dir, image_width, image_height, dpi):
    for item in data_chunk:
        render_ground_truth(
            item, base_tmp_dir=base_tmp_dir, output_dir=output_dir,
            image_width=image_width, image_height=image_height, dpi=dpi
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, help='The path to train.jsonl, validate.jsonl or test.jsonl')
    parser.add_argument('--output_dir', type=str, help='The directory to store the rendered images')
    parser.add_argument('--image_width', type=int, default=1344, help='The width resolution of rendered images')
    parser.add_argument('--image_height', type=int, default=224, help='The height resolution of the rendered images.')
    parser.add_argument('--dpi', type=int, default=240, help='The dpi when converting PDF files to PNG images.')
    parser.add_argument('--base_tmp_dir', type='str', default='/tmp/', help='The directory to store temp files.')
    parser.add_argument('--n_workrs', type=int, default=16, help='Number of processors used.')
    
    args = parser.parse_args()
    
    with open(args.filename, 'r') as fp:
        data = [json.loads(line) for line in fp.readlines()]
    
    num_processes = args.n_workers
    data_chunks = np.array_split(data, num_processes)
    
    with Pool(processes=num_processes) as pool:
        pool.map(worker, data_chunks, args.base_tmp_dir, args.output_dir, args.image_width, args.image_height, args.dpi)


if __name__ == "__main__":
    main()
