### Get Age & Gender Model
def model():

    import gdown

    # Google Drive file ID
    # file_id = "1jvwSSUTyQSaunzRjEzI4hINM285AKNt8"
    file_id = "1gOU7n8v1cQ2T77Uwn0IM1ioegkug2skX"

    # Construct the download URL
    url = f"https://drive.google.com/uc?id={file_id}"

    # Output file name (you can set this to anything you like)
    output = "./best.onnx"

    # Download the file
    gdown.download(url, output, quiet=False)

    return "Model Checkpoint Downloaded !"