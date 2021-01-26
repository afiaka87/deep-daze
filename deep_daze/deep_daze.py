import signal

import torch
from siren_pytorch import SirenNet, SirenWrapper
from torch import nn
from torch.cuda.amp import autocast

from deep_daze.clip import siren_model_and_norm
from deep_daze.util import signal_handling, interpolate, rand_cutout, _log

assert torch.cuda.is_available(), 'CUDA must be available in order to use Deep Daze'

terminate = False  # graceful keyboard interrupt

signal.signal(signal.SIGINT, signal_handling)
perceptor, normalize_image = siren_model_and_norm()


def norm_siren_output(img):
    return ((img + 1) * 0.5).clamp(0, 1)


class DeepDaze(nn.Module):
    def __init__(
            self,
            total_batches,
            batch_size,
            num_layers=8,
            image_width=512,
            loss_coefficient=100,
            debug=True
    ):
        super().__init__()
        self.debug = debug
        self.scheduled_sizes = []
        self.loss_coefficient = loss_coefficient
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

        self.siren_model = SirenWrapper(
            siren,
            image_width=image_width,
            image_height=image_width
        )

        self.generate_size_schedule()

    def forward(self, text, return_loss=True):
        siren_model_output = self.siren_model()
        # TODO is this the line that wasn't needed?
        siren_normalized_output = norm_siren_output(siren_model_output)

        if return_loss:
            pieces = []
            width = siren_normalized_output.shape[-1]
            size_slice = slice(self.num_batches_processed, self.num_batches_processed + self.batch_size)

            for size in self.scheduled_sizes[size_slice]:
                apper = rand_cutout(siren_normalized_output, size)
                _log(f"Random cutout: {apper}")
                apper = interpolate(apper, 224)
                _log(f"Random cutout bilinearly interpolated to 224 px: {apper}")
                # TODO Is this the line that isn't needed?
                pieces.append(normalize_image(apper))

            image = torch.cat(pieces)
            _log(f"Concatenated cutouts: {image}")

            with autocast(enabled=False):
                image_embed = perceptor.encode_image(image)
                text_embed = perceptor.encode_text(text)

            self.num_batches_processed += self.batch_size
            _log(f"Total batches processed: {self.num_batches_processed}")

            loss = -self.loss_coefficient * torch.cosine_similarity(text_embed, image_embed, dim=-1).mean()
            return loss

    def generate_size_schedule(self):
        batches = 0
        counter = 0

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
        _log(f"counter: {counter}, sizes: {sizes}")
        return sizes
