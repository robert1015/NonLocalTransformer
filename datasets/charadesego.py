""" Dataset loader for the Charades dataset """
from .charadesrgb import Charades, default_loader
from glob import glob
import random
from random import choice
import matplotlib.pyplot as plt
from PIL import Image

plt.switch_backend('agg')

def to_ego_time(thirdtime, egoscale):
    return int(round(thirdtime * egoscale))


def get_neg_time(egoii, n_ego, d):
    allframes = range(n_ego)
    candidates = [x for x in allframes if not egoii - d <= x <= egoii + d]
    if len(candidates) == 0:
        return None
    else:
        return choice(candidates)

def actions_match(first_action, third_action, n_ego, n, ii , egoii ):
    if (len(first_action[1]) >6 or len(third_action[1]) >6):
        return False
    first_scale = n_ego / float(first_action[1])
    third_scale = n / float(third_action[1])
    first_time = egoii / float(first_scale)
    third_time = ii / float(third_scale)
    first_class = []
    third_class = []
    for label_i in first_action[0]:
        if first_time <= label_i['end']  and first_time >= label_i['start'] :
            first_class.append(label_i['class'])

    for label_i in third_action[0]:
        if third_time <= label_i['end']  and third_time >= label_i['start'] :
            third_class.append(label_i['class'])

    l1 = len(first_class)
    l3 = len(third_class)
    if( l1 ==0 or l3==0):
        return False

    if (set(first_class) < set(third_class)  or set(third_class) < set(first_class)):
        return True
    else:
        return False



class CharadesEgo(Charades):
    def __init__(self, *args, **kwargs):
        self.fps = 24
        self.deltaneg = 10 * self.fps
        super(CharadesEgo, self).__init__(*args, **kwargs)

    def prepare(self, path, labels, split):
        datadir = path
        image_paths, targets, ids, meta = [], [], [], []
        pic = plt.figure()

        for i, (vid, label) in enumerate(labels.iteritems()):
            if vid[-3:] == 'EGO':
                firstvid = vid
                thirdvid = vid[:-3]
            else:
                firstvid = vid + 'EGO'
                thirdvid = vid
            gap = 4
            iddir = datadir + '/' + vid
            n = len(glob(iddir + '/*.jpg'))
            n_ego = len(glob(iddir + 'EGO/*.jpg'))
            scale = (n_ego - 1) / float(n - 1)
            if i % 100 == 0:
                print("{} {}".format(i, iddir))
            if n == 0 or n_ego == 0:
                continue

            if split == 'val_video':
                pass
            else:
                for ii in range(0, n, gap):
                    impath = '{}/{}-{:06d}.jpg'.format(iddir, vid, ii + 1)
                    egoii = to_ego_time(ii, scale)
                    impathego = '{}EGO/{}EGO-{:06d}.jpg'.format(iddir, vid, egoii + 1)
                    negii = get_neg_time(egoii, n_ego, self.deltaneg)
                    if negii is None:
                        continue
                    if not actions_match(labels[firstvid],labels[thirdvid],n_ego,n,ii,egoii):
                        """
                        ax1 = pic.add_subplot(2, 1, 1)
                        ax1.imshow(Image.open(impath))
                        ax2 = pic.add_subplot(2, 1, 2)
                        ax2.imshow(Image.open(impathego))
                        pic.savefig('/home/yhy/badpair/'+str(i)+'.jpg')
                        plt.clf()"""

                        continue
                    impathegoneg = '{}EGO/{}EGO-{:06d}.jpg'.format(iddir, vid, negii + 1)
                    image_paths.append((impathego, impath, impathegoneg))
                    targets.append(1)
                    ids.append(vid)
                    meta.append({'thirdtime': ii,
                                 'firsttime_pos': egoii,
                                 'firsttime_neg': negii,
                                 'n': n,
                                 'n_ego': n_ego})
        return {'image_paths': image_paths, 'targets': targets, 'ids': ids, 'meta': meta}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        impaths = self.data['image_paths'][index]
        target = self.data['targets'][index]
        meta = self.data['meta'][index]
        meta['id'] = self.data['ids'][index]
        try:
            ims = [default_loader(im) for im in impaths]
            if self.transform is not None:
                ims = [self.transform(im) for im in ims]
            # if random.random() > 0.5:
            #     ims[2], ims[0] = ims[0], ims[2]
            #     target = -1
            if self.target_transform is not None:
                target = self.target_transform(target)
            return ims, target, meta
        except Exception:
            for path in impaths:
                print('failed loading item: {}'.format(path))
            print('fetching another random item instead')
            return self[random.randrange(len(self))]

    @classmethod
    def get(cls, args):
        return super(CharadesEgo, cls).get(args, scale=(0.8, 1.0))
