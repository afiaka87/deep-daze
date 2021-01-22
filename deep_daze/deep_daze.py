from pathlib import Path
from datetime import datetime
import signal
import torch
import torch.nn.functional as F
import torchvision
from siren_pytorch import SirenNet, SirenWrapper
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torchvision.utils import save_image
from tqdm import trange

from deep_daze.clip import load, tokenize
from deep_daze.util import *


perceptor, normalize_image = load()

norm_siren_output = torchvision.transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


class DeepDaze(nn.Module):
    def __init__(
        self,
        total_batches,
        batch_size,
        num_layers=12,
        image_width=512,
        loss_coef=-100,
    ):
        super().__init__()
        self.loss_coef = loss_coef
        self.image_width = image_width

        self.batch_size = batch_size
        self.total_batches = total_batches
        self.num_batches_processed = 0

        siren = SirenNet(
            dim_in=2,
            dim_hidden=256,
            num_layers=num_layers,
            dim_out=3,
            use_bias=True
        )

        self.model = SirenWrapper(
            siren,
            image_width=image_width,
            image_height=image_width
        )

        self.generate_size_schedule()

    '''
    Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int
    '''

    def forward(self, text, return_loss=False):
        siren_out = self.model()

        if not return_loss:
            return siren_out

        pieces = []
        size_slice = slice(self.num_batches_processed,
                           self.num_batches_processed + self.batch_size)

        for size in self.scheduled_sizes[size_slice]:
            cropped_segment = rand_cutout(siren_out, size)
            # wild guess, but 224px is 512 * 0.875 / 2. Maybe that scales to other image sizes.
            bilinear_scaled_segment = norm_siren_output(
                interpolate(cropped_segment, self.image_width * 0.875 / 2))
            pieces.append(bilinear_scaled_segment)

        stitched_image = torch.cat(pieces)

        with autocast(enabled=False):
            image_embed = perceptor.encode_image(stitched_image)
            text_embed = perceptor.encode_text(text)

        self.num_batches_processed += self.batch_size

        loss = -self.loss_coef * \
            torch.cosine_similarity(text_embed, image_embed, dim=-1).mean()
        return loss

    def generate_size_schedule(self):
        batches = 0
        counter = 0
        self.scheduled_sizes = []

        while batches < self.total_batches:
            counter += 1
            sizes = self.sample_sizes(counter)
            batches += len(sizes)
            self.scheduled_sizes.extend(sizes)

    def sample_sizes(self, counter):
        pieces_per_group = 4

        # 6 piece schedule increasing in context as model saturates
        if counter < 500:
            partition = [4, 5, 3, 2, 1, 1]
        elif counter < 1000:
            partition = [2, 5, 4, 2, 2, 1]
        elif counter < 1500:
            partition = [1, 4, 5, 3, 2, 1]
        elif counter < 2000:
            partition = [1, 3, 4, 4, 2, 2]
        elif counter < 2500:
            partition = [1, 2, 2, 4, 4, 3]
        elif counter < 3000:
            partition = [1, 1, 2, 3, 4, 5]
        else:
            partition = [1, 1, 1, 2, 4, 7]

        dbase = .38
        step = .1
        width = self.image_width

        sizes = []
        for part_index in range(len(partition)):
            groups = partition[part_index]
            for _ in range(groups * pieces_per_group):
                sizes.append(torch.randint(
                    int((dbase + step * part_index + .01) * width),
                    int((dbase + step * (1 + part_index)) * width), ()))

        sizes.sort()
        return sizes


class Imagine(nn.Module):
    def __init__(
        self,
        text,
        *,
        lr=1e-5,
        batch_size=4,
        gradient_accumulate_every=4,
        save_every=100,
        image_width=512,
        num_layers=16,
        epochs=20,
        iterations=1050,
        save_progress=False,
        seed=None,
        open_folder=True
    ):
        super().__init__()

        if exists(seed):
            print(f'setting seed: {seed}')
            torch.manual_seed(seed)

        self.epochs = epochs
        self.iterations = iterations
        total_batches = epochs * iterations * batch_size * gradient_accumulate_every
        # For each epoch
        # for each iteration
        # generate 8 and accumulate them before continuing?

        deep_daze_model = DeepDaze(
            total_batches=total_batches,
            batch_size=batch_size,
            image_width=image_width,
            num_layers=num_layers
        ).cuda()

        self.model = deep_daze_model

        self.scaler = GradScaler()
        self.optimizer = Adam(self.model.parameters(), lr)
        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_every = save_every

        self.text = text
        textpath = self.text.replace(' ', '_')

        self.textpath = textpath
        self.filename = Path(f'./{textpath}.png')
        self.save_progress = save_progress
        self.encoded_text = tokenize(text).cuda()
        self.open_folder = open_folder

    def generate_image_cpu(self, current_epoch, current_iter):
        with torch.no_grad():
            img = norm_siren_output(self.model(
                self.encoded_text, return_loss=False).cpu())
            save_image(img, str(self.filename))
            print(f'image updated at "./{str(self.filename)}"')

            if self.save_progress:
                current_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
                save_image(img, Path(
                    f'./{self.textpath}_{current_epoch}_{current_iter}_{current_time}.png'))

    def train_step(self, epoch, iteration):
        total_loss = 0

        # Generate {gradient_accumulate_every}
        print(f'accumulating loss from {self.gradient_accumulate_every} generations of the model')
        for _ in range(self.gradient_accumulate_every):
            with autocast():
                loss = self.model(self.encoded_text)
            # Each generation is given 1/{num} importance in loss bc they can't change here.
            # No use in penalizing a net that has no ability to change.
            weighted_loss = loss / self.gradient_accumulate_every
            total_loss += weighted_loss
            print(f'  accumulation generation {_}: loss: {loss} | weighted_loss: {weighted_loss} | total_loss (so far): {total_loss}')

        self.scaler.scale(total_loss).backward() 
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.generate_image_cpu(epoch, iteration)

        return total_loss

    # graceful keyboard interrupt
    terminate = False

    def signal_handling(signum, frame):
        global terminate
        terminate = True

    signal.signal(signal.SIGINT, signal_handling)

    def forward(self):
        print(f'Imagining "{self.text}" from the depths of my weights...')

        if self.open_folder:
            open_folder('./')

        for epoch_pbar in trange(self.epochs, desc='epochs'):
            for iter_pbar in trange(self.iterations, desc='iterations'):
                loss = self.train_step(epoch_pbar, iter_pbar)
                iter_pbar.set_description(f'loss: {loss.item():.2f}')

                if terminate:
                    print('interrupted by keyboard, gracefully exiting')
                    return
