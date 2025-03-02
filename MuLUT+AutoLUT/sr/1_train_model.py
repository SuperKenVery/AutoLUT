import math
import os
import sys
import time
from tqdm import tqdm, trange
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.multiprocessing as mp
from PIL import Image
from rich.traceback import install
install(show_locals=False ,suppress=[torch,np])

import model
from data import Provider, SRBenchmark, TestDataset, DebugDataProvider, DebugDataset

sys.path.insert(0, "../")  # run under the project directory
from torch.utils.tensorboard import SummaryWriter
from common.option import TrainOptions
from common.utils import PSNR, logger_info, _rgb2ycbcr
from common.network import LrLambda

torch.backends.cudnn.benchmark = True

val_server_key=None


def makeValServer(opt, logger):
    from cryptography.fernet import Fernet
    import json
    from io import BytesIO
    import base64
    from http.server import BaseHTTPRequestHandler
    from queue import Queue
    import threading

    val_data=SRBenchmark(opt.valDir, scale=opt.scale)
    key=Fernet.generate_key()
    logger.info('<'*20+' Your Key ' +'>'*20)
    logger.info(key.decode())
    f=Fernet(key)

    writer = SummaryWriter(log_dir=opt.expDir)

    modes = list(opt.modes)
    stages = opt.stages
    model_class = getattr(model, opt.model)
    model_G = model_class(nf=opt.nf, scale=opt.scale, modes=modes, stages=stages).cuda()

    queue=Queue()

    def val_model():
        while True:
            model_file, iter_num=queue.get()
            lm = torch.load(model_file)
            model_G.load_state_dict(lm.state_dict(), strict=True)
            valid_steps(model_G, val_data, opt, iter_num, writer, 0, logger)

    thread=threading.Thread(target=val_model,daemon=True)
    thread.start()


    class ValServer(BaseHTTPRequestHandler):
        """Validate server

        Request format:
        {
            "model": "Base64 encoded model",
            "iter": 114514
        }

        Encrypt the json str with
        ```python3
        from cryptography.fernet import fernet
        Fernet(key).encrypt(req)
        ```
        """

        def do_POST(self):
            from cryptography.fernet import InvalidToken
            try:
                length=int(self.headers['Content-Length'])
                data=self.rfile.read(length)
                content=f.decrypt(data)
                request=json.loads(content.decode())
                logger.info(f"Received validate request, iter={request['iter']}")
                model_base64=request['model']
                model_data=base64.b64decode(model_base64)
                model_file=BytesIO(model_data)

                queue.put((model_file, request['iter']))
            except InvalidToken:
                logger.error("fernet: Invalid token")
                logger.error(f"length={length} some data={data[:50]}")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'Started working')


        def do_GET(self):
            logger.info(f"GET request at {self.path}")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"It's working!")


    return ValServer




def SaveCheckpoint(model_G, opt_G, Lr_sched_G, opt, i, logger, best=False):
    str_best = ''
    if best:
        str_best = '_best'

    torch.save(model_G, os.path.join(
        opt.expDir, 'Model_{:06d}{}.pth'.format(i, str_best)))
    torch.save(opt_G, os.path.join(
        opt.expDir, 'Opt_{:06d}{}.pth'.format(i, str_best)))
    if Lr_sched_G != None:
        torch.save(Lr_sched_G, os.path.join(
            opt.expDir, 'LRSched_{:06d}.pth'.format(i)))


    logger.info("Checkpoint saved {}".format(str(i)))


def valid_steps(model_G, valid, opt, iter, writer, rank, logger):
    if opt.debug or True:
        datasets = ['Set5', 'Set14']
    else:
        datasets = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109', 'DIV2K']

    with torch.no_grad():
        model_G.eval()

        for dataset in datasets:
            provider=Provider(
                1,
                opt.workerNum*2,
                opt.scale,
                opt.trainDir,
                opt.cropSize,
                debug=opt.debug,
                gpuNum=opt.gpuNum,
                data_class=TestDataset.get_init(dataset, valid),
                length=len(valid.files[dataset])
            )
            psnrs = []
            # files = valid.files[dataset]

            result_path = os.path.join(opt.valoutDir, dataset)
            if not os.path.isdir(result_path):
                os.makedirs(result_path)

            for i in range(len(provider)):
                lr, hr=provider.next()
                lb = hr.to(rank)
                input_im = lr.to(rank)

                im = input_im / 255.0
                im = torch.transpose(im,2,3)
                im = torch.transpose(im,1,2)

                pred=model_G.forward(im, phase='valid')

                pred=torch.squeeze(pred,0)
                pred=torch.transpose(pred,0,1)
                pred=torch.transpose(pred,1,2)
                pred=torch.round(torch.clamp(pred,0,255)).to(torch.uint8)

                lb = torch.squeeze(lb,0)
                left, right = _rgb2ycbcr(pred)[:, :, 0], _rgb2ycbcr(lb)[:, :, 0]
                psnrs.append(PSNR(left, right, opt.scale))  # single channel, no scale change

                # if iter < 10000:  # save input and gt at start
                #     input_img = torch.round(torch.clamp(input_im * 255.0, 0, 255)).to(torch.uint8).cpu()
                #     input_img = np.array(input_img)
                #     Image.fromarray(input_img).save(
                #         os.path.join(result_path, '{}_input.png'.format(key.split('_')[-1])))
                #     Image.fromarray(np.array(lb.cpu()).astype(np.uint8)).save(
                #         os.path.join(result_path, '{}_gt.png'.format(key.split('_')[-1])))

                Image.fromarray(np.array(pred.cpu())).save(
                    os.path.join(result_path, '{}_net.png'.format(f"{dataset}_{i}")))

            if rank==0:
                logger.info(
                    'Iter {} | Dataset {} | AVG Val PSNR: {:02f}'.format(iter, dataset, np.mean(np.asarray(psnrs))))
                writer.add_scalar('PSNR_valid/{}'.format(dataset), np.mean(np.asarray(psnrs)), iter)
                writer.flush()

                if dataset=='Set5':
                    set5_psnr=np.mean(np.asarray(psnrs))

    return set5_psnr

def send_val_request(url, model_G, iter, logger):
    import requests
    from io import BytesIO
    import base64
    import json
    from cryptography.fernet import Fernet

    save_file=BytesIO()
    torch.save(model_G, save_file)
    model_bytes=bytes(save_file.getbuffer())
    req={
        'model': base64.b64encode(model_bytes).decode(),
        'iter': iter
    }

    req_json=json.dumps(req).encode()
    f=Fernet(val_server_key)
    req_enc=f.encrypt(req_json)

    requests.post(url, data=req_enc)
    logger.info(f"Iter {iter}: Sent validation request to {url}")


def train(opt, logger, rank=0):
    # Tensorboard for monitoring
    writer = SummaryWriter(log_dir=opt.expDir)

    stages = opt.stages

    model_class = getattr(model, opt.model)

    model_G = model_class(opt.numSamplers, opt.sampleSize, nf=opt.nf, scale=opt.scale, stages=stages, act=opt.activateFunction, ).cuda()

    if opt.gpuNum > 1:
        model_G = model_G.to(rank)
        model_G = DDP(model_G, device_ids=[rank])

    # Optimizers
    params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))
    # lr would be multiplied by lr_lambda(i)
    initial_lr = 1 if opt.lambda_lr else opt.lr0
    opt_G = optim.Adam(params_G, lr=initial_lr, weight_decay=opt.weightDecay, amsgrad=False, fused=True)

    if opt.lambda_lr:
        lr_sched_obj=LrLambda(opt)
        lr_sched = optim.lr_scheduler.LambdaLR(opt_G, lr_sched_obj)

    # Load saved params
    if opt.startIter > 0:
        map_location = None
        if opt.gpuNum > 1:
            map_location = { 'cuda:0': f'cuda:{rank}'}
        lm = torch.load(
            os.path.join(opt.expDir, 'Model_{:06d}.pth'.format(opt.startIter)),
            map_location = map_location
        )
        model_G.load_state_dict(lm.state_dict(), strict=True)

        lm = torch.load(
            os.path.join(opt.expDir, 'Opt_{:06d}.pth'.format(opt.startIter))
        )
        opt_G.load_state_dict(lm.state_dict())

        if opt.lambda_lr:
            lm = torch.load(
                os.path.join(opt.expDir, 'LRSched_{:06d}.pth'.format(opt.startIter))
            )
            lr_sched.load_state_dict(lm.state_dict())

    # Training dataset
    train_iter = Provider(opt.batchSize, opt.workerNum, opt.scale, opt.trainDir, opt.cropSize, debug=opt.debug, gpuNum=opt.gpuNum)
    if opt.debug:
        train_iter = DebugDataProvider(DebugDataset())

    # Valid dataset
    valid = SRBenchmark(opt.valDir, scale=opt.scale)

    l_accum = [0., 0., 0.]
    dT = 0.
    rT = 0.
    accum_samples = 0

    # TRAINING
    i = opt.startIter

    with trange(opt.startIter + 1, opt.totalIter + 1, dynamic_ncols=True) as pbar:
        for i in pbar:
            model_G.train()

            # Data preparing
            st = time.time()
            im, lb = train_iter.next()
            im = im.to(rank)
            lb = lb.to(rank)
            dT += time.time() - st

            # TRAIN G
            st = time.time()
            opt_G.zero_grad()

            pred=model_G.forward(im, 'train')

            loss_G = F.mse_loss(pred, lb)
            loss_G.backward()
            opt_G.step()
            if opt.lambda_lr:
                lr_sched.step()
                pbar.set_postfix(lr=lr_sched.get_last_lr(), val_step=lr_sched_obj.opt.valStep)

            rT += time.time() - st

            # For monitoring
            accum_samples += opt.batchSize
            l_accum[0] += loss_G.item()

            # Show information
            if i % opt.displayStep == 0 and rank==0:
                writer.add_scalar('loss_Pixel', l_accum[0] / opt.displayStep, i)
                if opt.lambda_lr:
                    writer.add_scalar('learning_rate', torch.tensor(lr_sched.get_last_lr()), i)

                l_accum = [0., 0., 0.]
                dT = 0.
                rT = 0.

            # Save models
            if i % opt.saveStep == 0:
                if opt.gpuNum > 1 and rank==0:
                    SaveCheckpoint(model_G.module, opt_G, lr_sched if opt.lambda_lr else None, opt, i, logger)
                elif opt.gpuNum == 1:
                    SaveCheckpoint(model_G, opt_G, lr_sched if opt.lambda_lr else None, opt, i, logger)

            # Validation
            target_val_step = opt.valStep if opt.lambda_lr==False else lr_sched_obj.opt.valStep
            if i % target_val_step == 0:
                # validation during multi GPU training
                if opt.gpuNum > 1:
                    set5_psnr=valid_steps(model_G.module, valid, opt, i, writer, rank, logger)
                else:
                    if opt.valServer.lower() == "none":
                        set5_psnr=valid_steps(model_G, valid, opt, i, writer, rank, logger)
                    else:
                        send_val_request('http://'+opt.valServer, model_G, i, logger)
                if opt.lambda_lr:
                    lr_sched_obj.set5_psnr=set5_psnr

            # Display gradients
            if i % opt.gradientStep == 0:
                for name, param in model_G.named_parameters():
                    writer.add_histogram(f'gradients/{name}', param.grad, i)


    logger.info("Complete")

def val_server(opt,logger):
    from http.server import HTTPServer

    server=makeValServer(opt, logger)
    addr=opt.valServer
    bind,port=addr.split(':')
    listen=(bind,int(port))
    httpd=HTTPServer(listen, server)
    logger.info(f"Starting validate server at {listen}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

    httpd.server_close()
    logger.info("Stopped server.")

def main():
    global val_server_key
    opt_inst = TrainOptions()
    opt = opt_inst.parse()

    logger=logger_info(opt.debug, os.path.join(opt.expDir, 'train-{time}.log'))

    desc=opt_inst.describe_options(opt)
    for line in desc.split('\n'):
        logger.info(line)

    if opt.valServerMode:
        assert opt.gpuNum==1, "Multi-GPU validate server isn't supported yet"
        val_server(opt, logger)
    else:
        logger.complete()
        if opt.valServer.lower()!='none':
            val_server_key=input("Validate server key: ")
        if opt.gpuNum==1:
            train(opt, logger)
        else:
            logger.debug(f"Using {opt.gpuNum} GPUs")
            train_ddp(opt, opt.gpuNum, logger)


# For DDP
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group('gloo', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_in_one_process(rank, opt, world_size, logger):
    logger.debug(f"train_in_one_process: rank={rank}")
    setup(rank, world_size)
    try:
        train(opt, logger, rank=rank)
    except BaseException as e:
        logger.exception(e)
        logger.error(f"[rank {rank}] Received above exception, cleaning up...")
        cleanup()
        logger.info(f"[rank {rank}] Done cleaning up")

def train_ddp(opt, world_size, logger):
    mp.spawn(
        train_in_one_process,
        args=(opt, world_size, logger),
        nprocs=world_size,
        join=True
    )

if __name__=='__main__':
    main()
