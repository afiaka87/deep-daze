from pathlib import Path

import torch
from torch import nn, random
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torchvision.utils import save_image

from deep_daze import DeepDaze
from deep_daze.clip import tokenize
from deep_daze.deep_daze import normalize_image, terminate
from deep_daze.util import exists, open_folder, _log


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
            open_folder=False,
            debug=False
    ):
        super().__init__()

        self.debug = debug
        if exists(seed):
            _log(f'setting seed: {seed}', debug=self.debug)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True

        self.epochs = epochs
        self.iterations = iterations
        total_batches = epochs * iterations * batch_size * gradient_accumulate_every

        model = DeepDaze(
            total_batches=total_batches,
            batch_size=batch_size,
            image_width=image_width,
            num_layers=num_layers
        ).cuda()

        self.model = model

        self.scaler = GradScaler()
        self.optimizer = Adam(model.parameters(), lr)
        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_every = save_every

        self.text = text
        text_as_path = self.text.replace(' ', '_')

        self.text_as_path = text_as_path
        self.filename = Path(f'./{text_as_path}.png')
        self.save_progress = save_progress

        self.encoded_text = tokenize(text).cuda()

        self.open_folder = open_folder

    def train_step(self, epoch, i):
        total_loss = 0

        for _ in range(self.gradient_accumulate_every):
            with autocast():
                loss = self.model(self.encoded_text)
            loss = loss / self.gradient_accumulate_every
            total_loss += loss
            self.scaler.scale(loss).backward()

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if i % self.save_every == 0:
            with torch.no_grad():
                img = normalize_image(self.model(self.encoded_text, return_loss=False).cpu())
                img.clamp_(0., 1.)
                save_image(img, str(self.filename))
                _log(f'image updated at "./{str(self.filename)}"', debug=self.debug)

                if self.save_progress:
                    current_total_iterations = epoch * self.iterations + i
                    num = current_total_iterations // self.save_every
                    save_image(img, Path(f'./{self.text_as_path}.{num}.png'))

        return total_loss

    def forward(self):
        _log(f'Imagining "{self.text}" from the depths of my weights...', debug=self.debug)

        if self.open_folder:
            open_folder('./')
            self.open_folder = False

        for epoch in range(self.epochs):
            for iteration in range(self.iterations):
                loss = self.train_step(epoch, iteration)
                _log(f'iteration: {iteration}, epoch: {epoch}, loss: {loss}')

                if terminate:
                    _log('interrupted by keyboard, gracefully exiting', debug=False)
                    return
