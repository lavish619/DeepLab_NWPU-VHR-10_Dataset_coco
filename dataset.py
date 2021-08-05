import os
import numpy as np

import skimage.color
import skimage.io
import skimage.transform

import torchvision
from torch.utils.data import DataLoader

class CocoDataset(torchvision.datasets.coco.CocoDetection):
    def __init__(self,image_dir, annfile, transform=None, target_transform=None):
        super(CocoDataset,self).__init__(image_dir, annfile, transform, target_transform)
        self.image_info = []
        # Background is always the first class
        # self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.class_info = []
        self.source_class_ids = {}

        #  # Load all classes or a subset?
        # if not class_ids:
        #     # All classes
        class_ids = sorted(self.coco.getCatIds())

        # # All images or a subset?
        # if class_ids:
        #     image_Ids = []
        #     for id in class_ids:
        #         image_Ids.extend(list(self.coco.getImgIds(catIds=[id])))
        #     # Remove duplicates
        #     self.image_ids = list(set(image_Ids))
        # else:
        #     # All images
        self.image_ids = list(self.coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, self.coco.loadCats(i)[0]["name"])

        # Add images
        for i in self.image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, self.coco.imgs[i]['file_name']),
                width=self.coco.imgs[i]["width"],
                height=self.coco.imgs[i]["height"],
                annotations=self.coco.loadAnns(self.coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        
        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                        for info, id in zip(self.class_info, self.class_ids)}
        

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)


    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):        
        image_id = self.image_ids.index(self.image_info[idx]["id"])
        image = self.load_image(image_id)
        mask = self.load_normalmask(image_id)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)
            # orig_mask = mask.clone().detach()
            mask[mask!=0] = 1 #convert non zero values to 1
        return [image, mask, image_id]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_normalmask(self, image_id, class_channels = True):
        """Load semantic segmenation masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, classes].

        Returns:
        masks: A bool array of shape [height, width, classes] with
            one mask per class.
        """
        image_info = self.image_info[image_id]

        if class_channels:
            mask = np.zeros((image_info['height'],image_info['width'], self.num_classes))
        else:
            mask = np.zeros((image_info['height'],image_info['width']))

        annotations = self.image_info[image_id]["annotations"]
        for annotation in annotations:
            # className = self.class_info[annotation['category_id']]['name']
            # pixel_value = self.class_names.index(className)
            ann_mask = self.coco.annToMask(annotation)
            pixel_value = annotation['category_id']-1
            # pixel_value = self.map_source_class_id(f'{self.class_info[]['source']}.{}')
            if class_channels:
                mask[:,:,pixel_value] = np.maximum(ann_mask ,mask[:,:,pixel_value]) 
            # else:
            #     mask = np.maximum(ann_mask*pixel_value, mask)
        # mask[:,:,0] = np.logical_not(np.any(mask, axis=-1))
        return mask
    
    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            # class_id = self.map_source_class_id(
            #     "coco.{}".format(annotation['category_id']))
            class_id = annotation['category_id']
           
            m = self.coco.annToMask(annotation)
            # Some objects are so small that they're less than 1 pixel area
            # and end up rounded out. Skip those objects.
            if m.max() < 1:
                continue
            # Is it a crowd? If so, use a negative class ID.
            if annotation['iscrowd']:
                # Use negative class ID for crowds
                class_id *= -1
                # For crowd masks, annToMask() sometimes returns a mask
                # smaller than the given dimensions. If so, resize it.
                if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                    m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
            instance_masks.append(m)
            class_ids.append(class_id)

        # Pack instance masks into an array
        mask = np.stack(instance_masks, axis=2).astype(np.bool)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids


def prepare_dataset(images_path, train_annotations, val_annotations, 
                    batch_size=32, transform=None,target_transform=None ):
    
    train_set = CocoDataset(images_path, train_annotations, 
                        transform = transform, target_transform = target_transform)
    val_set = CocoDataset(images_path, val_annotations, 
                        transform = transform, target_transform = target_transform)
    
    image_sets = {
        'train': train_set,'val':val_set
     }

    dataloaders = {
        'train' : DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
        }

    return dataloaders, image_sets
