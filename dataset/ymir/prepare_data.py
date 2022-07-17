from random import sample
from typing import List
import torch
import numpy as np
import glob
import h5py
import os


def convertToPointCloud(
    files: List[str],
    outPutFolder: str,
    split: str = 'train',
    samplePoints: int = 0,  # no sampling
):

    train_instance_numpoints = 0
    train_instances = 0
    for file in files:
        name = os.path.basename(file).strip('.h5')
        outFilePath = os.path.join(outPutFolder, name + '.pth')
        # read in file
        with h5py.File(file, "r") as data:

            raw = np.array(data['raw'])
            colors = raw.flatten()  # column first
            colors = np.repeat(colors[:, np.newaxis], 3, axis=1)
            colors = colors.astype(np.float32)
            # normalize
            colors = colors / 32767.5 - 1

            coords = np.mgrid[
                0:1:raw.shape[0] * 1j,
                0:1:raw.shape[1] * 1j,
                0:1:raw.shape[2] * 1j,
            ].reshape(3, -1).T
            coords = coords.astype(np.float32)

            # sampling of points
            samples = np.arange(0, coords.shape[0])
            if samplePoints > 0:
                samples = np.random.choice(coords.shape[0], samplePoints)

            colors = colors[samples]
            coords = coords[samples]

            if split != 'test':
                # seems a bit weird, but they used float64 fort the labels so
                # let's use it as well
                sem_labels = np.array(data['foreground']).flatten().astype(np.float64)
                # map the background value (= 0 i.e. sugar walls) to -100
                sem_labels[sem_labels == 0] = -100

                # seems a bit weird, but they used float64 fort the labels so
                # let's use it as well
                instance_labels = np.array(data['label']).flatten().astype(np.float64)

                # sampling
                sem_labels = sem_labels[samples]
                instance_labels = instance_labels[samples]

                # keep track of the mean number of points per instance for the training dataset
                # NOTE: This does only work as long as we have one type of class
                if split == 'train':
                    values, counts = np.unique(
                        instance_labels, return_counts=True)
                    assert values[0] == 0
                    print(values, counts)
                    train_instance_numpoints += np.sum(counts[1:])
                    train_instances += len(counts[1:])

                torch.save((coords, colors, sem_labels,
                            instance_labels), outFilePath)
            else:
                torch.save((coords, colors), outFilePath)

    if split == 'train':
        assert train_instances > 0
        print('class_numpoints_mean: ', train_instance_numpoints / train_instances)


def getFiles(files, fileSplit):
    res = []
    for filePath in files:
        name = os.path.basename(filePath)
        num = name[:2] if name[:2].isdigit() else name[:1]
        if int(num) in fileSplit:
            res.append(filePath)
    return res


if __name__ == '__main__':
    data_folder = 'overfit_no_filter'
    split = 'train'
    trainFiles = sorted(glob.glob(data_folder + "/" + split + '/*.h5'))
    print(trainFiles)
    assert len(trainFiles) > 0
    trainOutDir = split
    os.makedirs(trainOutDir, exist_ok=True)
    convertToPointCloud(trainFiles, trainOutDir, split, samplePoints=35145)

    split = 'val'
    valFiles = sorted(glob.glob(data_folder + "/" + split + '/*.h5'))
    print(valFiles)
    assert len(valFiles) > 0
    valOutDir = split
    os.makedirs(valOutDir, exist_ok=True)
    convertToPointCloud(valFiles, valOutDir, split, samplePoints=35145)
