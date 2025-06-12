def model():

    import gdown

    # Google Drive file ID
    file_id = "1RRCUCVassEPfsJTorS5C9V0_BqMh1wdy"

    # Construct the download URL
    url = f"https://drive.google.com/uc?id={file_id}"

    # Output file name (you can set this to anything you like)
    output = "./best.onnx"

    # Download the file
    gdown.download(url, output, quiet=False)

    return "Model Checkpoint Downloaded !"