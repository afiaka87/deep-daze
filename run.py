from deep_daze import Imagine

TEXT = "a neural net learning how to see and communicate with itself"
imagine = Imagine(
    text=TEXT,
    num_layers=16,
    image_width=512,
    iterations=1050,
    epochs=20,
    gradient_accumulate_every=8,
    batch_size=2,
    lr=2e-5,
    save_progress=True
)

imagine()
