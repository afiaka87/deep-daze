import subprocess


def colab_init():
    try:
        from google.colab import output
        IN_COLAB = True
    except:
        IN_COLAB = False

    if IN_COLAB:
        # !pip install torch == 1.7.1{torch_version_suffix} torchvision == 0.8.2{torch_version_suffix} - f https: // download.pytorch.org/whl/torch_stable.html ftfy regex
        # %cd / content/
        # !git clone https: // github.com/openai/CLIP.git
        # %cd / content/CLIP/
        # !pip install ftfy

        CUDA_version = [s for s in subprocess.check_output(
            ["nvcc", "--version"]).decode("UTF-8").split(", ") if s.startswith("release")][0].split(" ")[-1]
        print("CUDA version:", CUDA_version)

        if CUDA_version == "10.0":
            torch_version_suffix = "+cu100"
        elif CUDA_version == "10.1":
            torch_version_suffix = "+cu101"
        elif CUDA_version == "10.2":
            torch_version_suffix = ""
        else:
            torch_version_suffix = "+cu110"
