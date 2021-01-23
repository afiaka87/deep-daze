from deep_daze import Imagine

TEXT = "a neural net learning how to see and communicate with itself"
imagine = Imagine(
    text=TEXT,
    num_layers=16,
    image_width=128,
    iterations=1050,
    epochs=20,
    gradient_accumulate_every=8,
    batch_size=4,
    lr=1e-7,
    save_progress=True,
    save_every=8
)

imagine()
