

def model():

    import gdown

    # Google Drive file ID
    file_id = "1gHR8K_hVhs2lkT860KODhP_mbfg7f7As"

    # Construct the download URL
    url = f"https://drive.google.com/uc?id={file_id}"

    # Output file name (you can set this to anything you like)
    output = "./best_weights.pth"

    # Download the file
    gdown.download(url, output, quiet=False)

    return "Model Checkpoint Downloaded !"