import argparse
import logging
from collections import defaultdict
from datetime import datetime
from time import time

import torch
import torchaudio
from torch import nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchaudio.models._tacotron2 import _Tacotron2
from utils import MetricLogger, count_parameters, save_checkpoint
from losses import Tacotron2Loss
from dataset import TextMelLoader, TextMelCollate, batch_to_gpu
from processing import symbols


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        default=1,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint",
    )
    parser.add_argument(
        "-d",
        "--dataset-path",
        type=str,
        default="/private/home/jimchen90/datasets",
        help="Path to dataset",
    )
    parser.add_argument(
        "--epochs",
        default=1000,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="manual epoch number"
    )
    parser.add_argument(
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency in epochs",
    )
    parser.add_argument(
        "--dataset",
        default="ljspeech",
        choices=["ljspeech", "libritts"],
        type=str,
        help="select dataset to train with",
    )
    parser.add_argument(
        "--batch-size", default=48, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--learning-rate", default=1e-3, type=float, metavar="LR", help="learning rate",
    )
    parser.add_argument(
        "--weight-decay", default=1e-6, type=float, help="Weight decay",
    )
    parser.add_argument("--clip-grad", metavar="NORM", type=float, default=1.0)
    parser.add_argument(
        "--anneal-steps",
        nargs="*",
        default=[500, 1000, 1500],
        help="Epochs after which decrease learning rate",
    )
    parser.add_argument(
        "--anneal-factor",
        type=float,
        choices=[0.1, 0.3],
        default=0.1,
        help="Factor for annealing learning rate",
    )
    parser.add_argument(
        "--jit", default=False, action="store_true", help="if used, model is jitted"
    )

    # dataset parameters
    parser.add_argument(
        "--load-mel-from-disk",
        action="store_true",
        help="Loads mel spectrograms from disk instead of computing them on the fly",
    )
    parser.add_argument(
        "--training-files",
        default="/private/home/jimchen90/datasets/filelist/ljs_audio_text_train_filelist.txt",
        type=str,
        help="Path to training filelist",
    )
    parser.add_argument(
        "--validation-files",
        default="/private/home/jimchen90/datasets/filelist/ljs_audio_text_val_filelist.txt",
        type=str,
        help="Path to validation filelist",
    )
    parser.add_argument(
        "--text-cleaners",
        nargs="*",
        default=["english_cleaners"],
        type=str,
        help="Type of text cleaners for input text",
    )

    # audio parameters
    parser.add_argument(
        "--sampling-rate",
        default=22050,
        type=int,
        help="the rate of audio dimensions (samples per second)",
    )
    parser.add_argument(
        "--max-wav-value", default=32768.0, type=float, help="Maximum audiowave value"
    )
    parser.add_argument("--filter-length", default=1024, type=int, help="Filter length")
    parser.add_argument(
        "--hop-length",
        default=256,
        type=int,
        help="the number of samples between the starts of consecutive frames",
    )
    parser.add_argument(
        "--win-length", default=1024, type=int, help="the length of the STFT window",
    )
    parser.add_argument(
        "--mel-fmin", default=0.0, type=float, help="the minimum frequency",
    )
    parser.add_argument(
        "--mel-fmax", default=8000.0, type=float, help="Maximum mel frequency"
    )
    parser.add_argument(
        "--n-mel-channels",
        default=80,
        type=int,
        help="Number of bins in mel-spectrograms",
    )

    # symbols parameters
    global symbols
    len_symbols = len(symbols)
    parser.add_argument(
        "--mask-padding", default=False, type=bool, help="Use mask padding"
    )

    parser.add_argument(
        "--n-symbols",
        default=len_symbols,
        type=int,
        help="Number of symbols in dictionary",
    )
    parser.add_argument(
        "--n-symbols-embedding", default=512, type=int, help="Input embedding dimension"
    )

    # encoder parameters
    parser.add_argument(
        "--encoder-kernel-size", default=5, type=int, help="Encoder kernel size"
    )
    parser.add_argument(
        "--encoder-n-convolutions",
        default=3,
        type=int,
        help="Number of encoder convolutions",
    )
    parser.add_argument(
        "--n-encoder-embedding",
        default=512,
        type=int,
        help="Encoder embedding dimension",
    )

    # decoder parameters
    parser.add_argument(
        "--n-frames-per-step",
        default=1,
        type=int,
        help="Number of frames processed per step",
    )  # currently only 1 is supported
    parser.add_argument(
        "--n-decoder-rnn",
        default=1024,
        type=int,
        help="Number of units in decoder LSTM",
    )
    parser.add_argument(
        "--n-prenet",
        default=256,
        type=int,
        help="Number of ReLU units in prenet layers",
    )
    parser.add_argument(
        "--max-decoder-steps",
        default=2000,
        type=int,
        help="Maximum number of output mel spectrograms",
    )
    parser.add_argument(
        "--gate-threshold",
        default=0.5,
        type=float,
        help="Probability threshold for stop token",
    )
    parser.add_argument(
        "--p-attention-dropout",
        default=0.1,
        type=float,
        help="Dropout probability for attention LSTM",
    )
    parser.add_argument(
        "--p-decoder-dropout",
        default=0.1,
        type=float,
        help="Dropout probability for decoder LSTM",
    )
    parser.add_argument(
        "--decoder-no-early-stopping",
        action="store_true",
        help="Stop decoding once all samples are finished",
    )

    # attention parameters
    parser.add_argument(
        "--n-attention-rnn",
        default=1024,
        type=int,
        help="Number of units in attention LSTM",
    )
    parser.add_argument(
        "--n-attention",
        default=128,
        type=int,
        help="Dimension of attention hidden representation",
    )

    # location layer parameters
    parser.add_argument(
        "--attention-location-n-filters",
        default=32,
        type=int,
        help="Number of filters for location-sensitive attention",
    )
    parser.add_argument(
        "--attention-location-kernel-size",
        default=31,
        type=int,
        help="Kernel size for location-sensitive attention",
    )

    # Mel-post processing network parameters
    parser.add_argument(
        "--n-postnet-embedding",
        default=512,
        type=int,
        help="Postnet embedding dimension",
    )
    parser.add_argument(
        "--postnet-kernel-size", default=5, type=int, help="Postnet kernel size"
    )
    parser.add_argument(
        "--postnet-n-convolutions",
        default=5,
        type=int,
        help="Number of postnet convolutions",
    )

    args = parser.parse_args()
    return args


def adjust_learning_rate(epoch, optimizer, learning_rate, anneal_steps, anneal_factor):

    p = 0
    if anneal_steps is not None:
        for i, a_step in enumerate(anneal_steps):
            if epoch >= int(a_step):
                p = p + 1

    if anneal_factor == 0.3:
        lr = learning_rate * ((0.1 ** (p // 2)) * (1.0 if p % 2 == 0 else 0.3))
    else:
        lr = learning_rate * (anneal_factor ** p)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch):

    model.train()

    sums = defaultdict(lambda: 0.0)
    start1 = time()

    metric = MetricLogger("train_iteration")
    metric["epoch"] = epoch

    for i, batch in enumerate(data_loader):

        start2 = time()

        adjust_learning_rate(
            epoch, optimizer, args.learning_rate, args.anneal_steps, args.anneal_factor
        )

        model.zero_grad()
        x, y, _ = batch_to_gpu(batch)

        y_pred = model(x)
        loss = criterion(y_pred, y)

        loss_item = loss.item()
        sums["loss"] += loss_item
        metric["loss"] = loss_item

        optimizer.zero_grad()
        loss.backward()

        if args.clip_grad > 0:
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.clip_grad
            )
            sums["gradient"] += gradient.item()
            metric["gradient"] = gradient.item()

        optimizer.step()

        metric["iteration"] = sums["iteration"]
        metric["time"] = time() - start2
        metric()
        sums["iteration"] += 1

    avg_loss = sums["loss"] / len(data_loader)

    metric = MetricLogger("train_epoch")
    metric["epoch"] = epoch
    metric["loss"] = sums["loss"] / len(data_loader)
    metric["gradient"] = avg_loss
    metric["time"] = time() - start1
    metric()


def validate(model, criterion, data_loader, device, epoch):

    with torch.no_grad():

        model.eval()
        sums = defaultdict(lambda: 0.0)
        start = time()

        for i, batch in enumerate(data_loader):

            start2 = time()

            x, y, _ = batch_to_gpu(batch)

            y_pred = model(x)
            loss = criterion(y_pred, y)

            loss_item = loss.item()
            sums["loss"] += loss_item

        avg_loss = sums["loss"] / len(data_loader)

        metric = MetricLogger("validation")
        metric["epoch"] = epoch
        metric["loss"] = avg_loss
        metric["time"] = time() - start
        metric()

        return avg_loss


def main(args):

    devices = ["cuda" if torch.cuda.is_available() else "cpu"]

    logging.info("Start time: {}".format(str(datetime.now())))

    train_dataset = TextMelLoader(args.dataset_path, args.training_files, args)

    val_dataset = TextMelLoader(args.dataset_path, args.validation_files, args)

    loader_training_params = {
        "num_workers": args.workers,
        "pin_memory": False,
        "shuffle": True,
        "drop_last": True,
    }
    loader_validation_params = loader_training_params.copy()
    loader_validation_params["shuffle"] = False

    collate_fn = TextMelCollate(args.n_frames_per_step)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        **loader_training_params,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        **loader_validation_params,
    )

    model = _Tacotron2(
        # optimization
        mask_padding=args.mask_padding,
        # audio
        n_mel_channels=args.n_mel_channels,
        # symbols
        n_symbols=args.n_symbols,
        n_symbols_embedding=args.n_symbols_embedding,
        # encoder
        encoder_kernel_size=args.encoder_kernel_size,
        encoder_n_convolutions=args.encoder_n_convolutions,
        n_encoder_embedding=args.n_encoder_embedding,
        # attention
        n_attention_rnn=args.n_attention_rnn,
        n_attention=args.n_attention,
        # attention location
        attention_location_n_filters=args.attention_location_n_filters,
        attention_location_kernel_size=args.attention_location_kernel_size,
        # decoder
        n_frames_per_step=args.n_frames_per_step,
        n_decoder_rnn=args.n_decoder_rnn,
        n_prenet=args.n_prenet,
        max_decoder_steps=args.max_decoder_steps,
        gate_threshold=args.gate_threshold,
        p_attention_dropout=args.p_attention_dropout,
        p_decoder_dropout=args.p_decoder_dropout,
        # postnet
        n_postnet_embedding=args.n_postnet_embedding,
        postnet_kernel_size=args.postnet_kernel_size,
        postnet_n_convolutions=args.postnet_n_convolutions,
        decoder_no_early_stopping=args.decoder_no_early_stopping,
    )

    if args.jit:
        model = torch.jit.script(model)

    # model = torch.nn.DataParallel(model)
    model = model.to(devices[0], non_blocking=True)

    n = count_parameters(model)
    logging.info(f"Number of parameters: {n}")

    # Optimizer
    optimizer_params = {"lr": args.learning_rate, "weight_decay": args.weight_decay}

    optimizer = Adam(model.parameters(), **optimizer_params)

    criterion = Tacotron2Loss()

    best_loss = 1.0

    if args.checkpoint and os.path.isfile(args.checkpoint):
        logging.info(f"Checkpoint: loading '{args.checkpoint}'")
        checkpoint = torch.load(args.checkpoint)

        args.start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]

        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        logging.info(
            f"Checkpoint: loaded '{args.checkpoint}' at epoch {checkpoint['epoch']}"
        )
    else:
        logging.info("Checkpoint: not found")

        save_checkpoint(
            {
                "epoch": args.start_epoch,
                "state_dict": model.state_dict(),
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
            },
            False,
            args.checkpoint,
        )

    for epoch in range(args.start_epoch, args.epochs):

        train_one_epoch(
            model, criterion, optimizer, train_loader, devices[0], epoch,
        )

        if not (epoch + 1) % args.print_freq or epoch == args.epochs - 1:

            sum_loss = validate(model, criterion, val_loader, devices[0], epoch)

            is_best = sum_loss < best_loss
            best_loss = min(sum_loss, best_loss)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                args.checkpoint,
            )

    logging.info(f"End time: {datetime.now()}")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args)
