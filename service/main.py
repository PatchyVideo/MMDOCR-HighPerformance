
import asyncio
import time
import aiohttp
import string
import random
import os
import subprocess
import atexit
from subprocess import Popen, PIPE, STDOUT
from datetime import datetime
from contextlib import closing

SESSION_ID = 'eyJfcGVybWFuZW50Ijp0cnVlLCJzaWQiOnsiIGIiOiJNV0UxTjJabVlqQXlPV1ZsWlRsak5qY3dPVGs1TVRjMU9XTmpaakkyWTJZPSJ9fQ.Xy4TGQ.P2WH6csZ789j-uKatimxlbqG2C8'
WORKER_ID = 'ae4re4tgr25UERqEt1Ac80Uq'
BACKEND_URL = 'https://thvideo.tv'
VERSION = 2

RESERVED_VIDEOS = []

def get_random_string(length) :
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

async def perform_ocr(video_filename, srt_filename) :
    # this is a blocking call, only one process runs at a time
    # placeholder
    print('Processing file %s'%video_filename)

    # time.sleep(10)
    # with open(srt_filename, 'w') as fp :
    #     fp.write('Subtitle file=%s,video file=%s'%(srt_filename,video_filename))
    # return True
    await asyncio.sleep(0) # yield
    process = Popen('MMDOCR-HighPerformance.exe %s %s' % (video_filename, srt_filename), shell=True, stderr=STDOUT, stdout=subprocess.PIPE, close_fds=True)
    process.wait()
    process_output, _ =  process.communicate()
    with open('ocr.log', 'a+') as fp :
        fp.write('%s\n' % datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        fp.write(process_output.decode('utf-8'))
        fp.write('\n')
    return process.returncode == 0

async def retrive_queue(max_videos = 20) :
    async with aiohttp.ClientSession() as session :
        async with session.post(
            BACKEND_URL + '/be/subtitles/worker/query_queue.do',
            json = {'max_videos': max_videos, 'worker_id': WORKER_ID},
            cookies = {'session': SESSION_ID}
            ) as resp :
            if resp.status // 100 == 2 :
                resp_json = await resp.json()
                if resp_json['status'] == 'SUCCEED' :
                    return resp_json['data']['urls']
                else :
                    print('[*] retrive_queue failed, reason = %s' % repr(resp_json))
            else :
                print('[*] retrive_queue failed, reason = status_code=%d' % resp.status)
    return []

async def download_file(url, filename, session) :
    print('[+] Downloading into %s' % filename)
    with open(filename, 'wb') as fp :
        async with session.get(url) as response :
            if response.status // 100 == 2 :
                while True:
                    chunk = await response.content.read(1024 * 1024 * 4) # 4MB
                    if not chunk:
                        break
                    fp.write(chunk)
            else :
                raise Exception('response.status_code=%d' % response.status)

class TempFile(object) :
    def __init__(self, l = 12) :
        self.l = l

    def __enter__(self) :
        self.vfilename = get_random_string(12)
        self.sfilename = self.vfilename + '.srt'
        return self

    def __exit__(self, type, value, traceback) :
        try :
            os.remove(self.vfilename)
            os.remove(self.sfilename)
        except :
            pass

def get_suitable_resolution(streams) :
    best = streams[0]['src'][0]
    for s in streams :
        if s['quality'].split('p')[0] == '360' :
            best = s['src'][0]
    return best

async def process_single_video(unique_id, url) :
    try_count = 0
    succeed = False
    while try_count < 3 :
        try :
            async with aiohttp.ClientSession() as session :
                video_url = ''
                print('[+] retriving video URL for %s' % unique_id)
                #raise Exception('test')
                async with session.post(
                    BACKEND_URL + '/be/helper/get_video_stream',
                    json = {'url': url},
                    cookies = {'session': SESSION_ID}
                    ) as resp :
                    resp_json = await resp.json()
                    if resp_json['status'] == 'SUCCEED' :
                        video_url = resp_json['data']['streams'][0]['src'][0]
                        try :
                            best_url = get_suitable_resolution(resp_json['data'])
                            video_url = best_url
                        except :
                            pass
                        print('[+] video url = %s'%video_url)
                    else :
                        raise Exception(repr(resp_json))
                if not video_url :
                    raise Exception('no video url retrived')
                # notify video downloading
                print('[+] Notify video being downloaded for %s' % url)
                async with session.post(
                    BACKEND_URL + '/be/subtitles/worker/update_status.do',
                    json = {'unique_id_status_map': {unique_id: 'Downloading'}, 'worker_id': WORKER_ID},
                    cookies = {'session': SESSION_ID}
                    ) as resp :
                    pass
                # download
                print('[+] Downloading %s' % url)
                with TempFile(12) as tmp_file :
                    await download_file(video_url, tmp_file.vfilename, session)
                    # notify processing
                    print('[+] Notify video being processed for %s' % url)
                    async with session.post(
                        BACKEND_URL + '/be/subtitles/worker/update_status.do',
                        json = {'unique_id_status_map': {unique_id: 'Processing'}, 'worker_id': WORKER_ID},
                        cookies = {'session': SESSION_ID}
                        ) as resp :
                        pass
                    if not await perform_ocr(tmp_file.vfilename, tmp_file.sfilename) :
                        raise Exception('OCR failed')
                    print('[+] Uploading subtitle for %s' % url)
                    with open(tmp_file.sfilename, 'r') as fp :
                        # send result
                        content = fp.read()
                        print('[+] Subtitle size = %d' % len(content.encode('utf-8')))
                        async with session.post(
                            BACKEND_URL + '/be/subtitles/worker/post_ocr_result.do',
                            json = {'unique_id': unique_id, 'content': content, 'format': 'vtt', 'version': VERSION, 'worker_id': WORKER_ID},
                            cookies = {'session': SESSION_ID}
                            ) as resp :
                            resp_json = await resp.json()
                            if resp_json['status'] == 'SUCCEED' :
                                print('[+] Subtitle upload succeed, subid = %s' % resp_json['data']['subid']['$oid'])
                        succeed = True
                if succeed :
                    break
        except KeyboardInterrupt :
            print('[*] KeyboardInterrupt!!!')
            succeed = False
            return
        except Exception as e :
            print('[*] Process failed!!!')
            print(e)
            try_count += 1
            print('[*] Retrying %d'%try_count)
    if not succeed :
        # notify failed
        print('[+] Notify video processe failed for %s' % url)
        async with aiohttp.ClientSession() as session :
            async with session.post(
                BACKEND_URL + '/be/subtitles/worker/update_status.do',
                json = {'unique_id_status_map': {unique_id: 'Error'}, 'worker_id': WORKER_ID},
                cookies = {'session': SESSION_ID}
                ) as resp :
                pass
    RESERVED_VIDEOS.remove(unique_id)

async def main() :
    got_jobs = True
    while True :
        if got_jobs :
            print('[+] Waiting for jobs...')
        urls = await retrive_queue()
        if urls :
            print('[+] Retrived %d jobs' % len(urls))
            got_jobs = True
        if not urls :
            await asyncio.sleep(10)
            got_jobs = False
            continue
        tasks = []
        for item in urls :
            RESERVED_VIDEOS.append(item['unique_id'])
            tasks.append(asyncio.create_task(process_single_video(item['unique_id'], item['url'])))
        for task in tasks :
            await task


# async def sleep_print(idx: int) :
#     print('[%d] sleeping for 5 sec' % idx)
#     await asyncio.sleep(5-idx)
#     print('[%d] has waken up' % idx)

# async def run_sleep_print() :
#     sleep_print_futures = [asyncio.create_task(sleep_print(i)) for i in range(5)]
#     print('all tasks created')
#     for sleep_print_future in sleep_print_futures:
#         asyncio.gather(sleep_print_future)
#     print('all tasks gather')
#     for sleep_print_future in sleep_print_futures:
#         await sleep_print_future

# with closing(asyncio.get_event_loop()) as loop :
#     loop.run_until_complete(run_sleep_print())

def cleanup() :
    print('[+] please wait for program to cleanup and exit')
    import requests
    status_map = {unique_id: 'Queuing' for unique_id in RESERVED_VIDEOS}
    # queue unfinished jobs
    ret = requests.post(
        BACKEND_URL + '/be/subtitles/worker/update_status.do',
        json = {'unique_id_status_map': status_map, 'worker_id': WORKER_ID},
        cookies = {'session': SESSION_ID}
        )

atexit.register(cleanup)

loop = asyncio.get_event_loop()
runner, site = loop.run_until_complete(main())

try:
	loop.run_forever()
except KeyboardInterrupt as err:
	loop.run_until_complete(runner.cleanup())

    
