import os
import glob

def read_dataset(data_dir, ext):
    """
    Read dataset from the root folder of the cmu indoor dataset.
    The folder structure must be remained the same, or modify the
    code as you like.
    """
    imagelist = glob.glob(os.path.join(data_dir, "raw_image/*." + ext))
    labellist = glob.glob(os.path.join(data_dir, "ground_truth/*_gt_bw." + ext))

    pairlist = []
    for image, label in zip(imagelist, labellist):
        pair = {
            "image": image,
            "label": label
        }
        pairlist.append(pair)

    return pairlist
