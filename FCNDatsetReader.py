"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc

class FCNDatsetReader(object):
    """
        A general dataset reader for fully convolutional networks
    """
    filelist = []
    images = []
    labels = []
    options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, filelist, options=None):
        """
            Intialize a generic dataset reader with list of image-label file pairs.
            :param filelist: list of file records to read -
            sample record: {'image': image_file, 'label': labeled_file}
            :param options: A dictionary of options for modifying the output image
            Available options:
            resize = True/ False,
            resize_size = #size of output image - does bilinear resize,
            color=True/False
        """
        print('Initialize dataset reader')
        if options is not None:
            print(options)
        else:
            options = {}
            options["resize"] = False
            options["color"] = True

        self.filelist = filelist
        self.options = options
        self.__read_images()

    def __read_images(self):
        self.images = np.array(
            [self.__transform(filename['image'], True) for filename in self.filelist])
        self.labels = np.array(
            [np.expand_dims(self.__transform(filename['label'], False), axis=3)
             for filename in self.filelist])
        for label in self.labels: # temporal code, must be changed to be universal
            label[label == 255] = 1
        print('Constructed image shape:' + self.images.shape)
        print('Constructed label shape:' + self.labels.shape)

    def __transform(self, filename, multi_channels):
        image = misc.imread(filename)
        if multi_channels and len(image.shape) < 3:  # make sure images have (w, h, c) shape
            image = np.array([image for i in range(3)])

        if self.options.get("resize", False) and self.options["resize"]:
            resize_size = int(self.options["resize_size"])
            resize_image = misc.imresize(image, [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)

    def get_records(self):
        """ Return the filenames of images and labels """
        return self.images, self.labels

    def set_batch_offset(self, offset=0):
        """ Set file offset to given offset """
        self.batch_offset = offset

    def next_batch(self, batch_size):
        """
        Return batch of images and corresponding label images.
        All images are guaranteed to be consumed.
        """
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("*** Epochs completed: %d ***" % self.epochs_completed)
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.labels[start:end]

    def get_random_batch(self, batch_size):
        """
        Return random batch of images and corresponding label images.
        """
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.labels[indexes]
