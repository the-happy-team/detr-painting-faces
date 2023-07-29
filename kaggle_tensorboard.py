from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
from subprocess import Popen
from os import chmod
from os.path import isfile
import json
import time
import psutil

def launch_tensorboard(logdir="./logs"):
    tb_process, ngrok_process = None, None

    # Launch TensorBoard
    if not is_process_running('tensorboard'):
        tb_command = f'tensorboard --logdir {logdir} --host 0.0.0.0 --port 6006'
        tb_process = run_cmd_async_unsafe(tb_command)

    # Install ngrok
    if not isfile('./ngrok'):
        ngrok_url = 'https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip'
        download_and_unzip(ngrok_url)
        chmod('./ngrok', 0o755)

    # Create ngrok tunnel and print its public URL
    if not is_process_running('ngrok'):
        ngrok_process = run_cmd_async_unsafe('./ngrok http 6006')
        time.sleep(1) # Waiting for ngrok to start the tunnel
    ngrok_api_res = urlopen('http://127.0.0.1:4040/api/tunnels', timeout=10)
    ngrok_api_res = json.load(ngrok_api_res)
    assert len(ngrok_api_res['tunnels']) > 0, 'ngrok tunnel not found'
    tb_public_url = ngrok_api_res['tunnels'][0]['public_url']
    print(f'TensorBoard URL: {tb_public_url}')

    return tb_process, ngrok_process


def download_and_unzip(url, extract_to='.'):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)


def run_cmd_async_unsafe(cmd):
    return Popen(cmd, shell=True)


def is_process_running(process_name):
    running_process_names = (proc.name() for proc in psutil.process_iter())
    return process_name in running_process_names