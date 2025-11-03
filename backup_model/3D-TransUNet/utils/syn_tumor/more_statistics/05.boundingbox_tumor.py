from scipy import ndimage
import os, numpy, nibabel, copy, pandas, glob, argparse, numpy as np, math
from tqdm import tqdm
from multiprocessing import Pool, Manager



label_of_interest = {
    'pdac': 3,
    'cyst': 4,
    'pnet': 5,
}

class Counter(object):
    def __init__(self):
        self.val = Manager().Value('i', 0)
        self.lock = Manager().Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value

def bounding_box_calculation(mask):
    positions = numpy.where(mask>0)
    bbox = [
        [numpy.min(positions[2]), numpy.max(positions[2])], 
        [numpy.min(positions[1]), numpy.max(positions[1])], 
        [numpy.min(positions[0]), numpy.max(positions[0])], 
    ]
    size_of_bbox = [bbox[0][1]-bbox[0][0], bbox[1][1]-bbox[1][0], bbox[2][1]-bbox[2][0]]
    center_of_bbox = [(bbox[0][1]+bbox[0][0])/2, (bbox[1][1]+bbox[1][0])/2, (bbox[2][1]+bbox[2][0])/2]
    return bbox, size_of_bbox, center_of_bbox

def calculate_radius(volume):
    # Calculate the radius using the formula derived above
    radius = ((3 * volume) / (4 * math.pi)) ** (1/3)
    return radius

def cal_results(name, counter):
    results = []
    gt_volume = nibabel.load(name)
    mask = numpy.uint8(gt_volume.get_fdata())
    lesion_mask = np.uint8(mask==label_of_interest[args.target])
    
    header = gt_volume.header
    spacing = header.get_zooms()
    per_pix_volume = spacing[0] * spacing[1] * spacing[2]
    

    if lesion_mask.sum() == 0:
        counter.increment()
        datum = {
            'case': name.split('/')[-1].rstrip('.nii.gz')+'_'+str(1),
            'lesion_size': 0,
            'lesion_radius': 0,
            'bb_p_x0': 0,
            'bb_p_x1': 0,
            'bb_p_y0': 0,
            'bb_p_y1': 0,
            'bb_p_z0': 0,
            'bb_p_z1': 0,
            'bb_p_size_x': 0,
            'bb_p_size_y': 0,
            'bb_p_size_z': 0,
            'bb_p_center_x': 0,
            'bb_p_center_y': 0,
            'bb_p_center_z': 0,
            'bb_'+args.target+'_x0': 0,
            'bb_'+args.target+'_x1': 0,
            'bb_'+args.target+'_y0': 0,
            'bb_'+args.target+'_y1': 0,
            'bb_'+args.target+'_z0': 0,
            'bb_'+args.target+'_z1': 0,
            'bb_'+args.target+'_size_x': 0,
            'bb_'+args.target+'_size_y': 0,
            'bb_'+args.target+'_size_z': 0,
            'bb_'+args.target+'_center_x': 0,
            'bb_'+args.target+'_center_y': 0,
            'bb_'+args.target+'_center_z': 0,
            'offset_of_center_pancreas_to_'+args.target+'_x': 0,
            'offset_of_center_pancreas_to_'+args.target+'_y': 0,
            'offset_of_center_pancreas_to_'+args.target+'_z': 0,
            }
        return [datum]

    organ_mask  = np.uint8(mask>0)
    gt_les_numeric, gt_les_N = ndimage.label(lesion_mask)
    bbox_organ, size_of_bbox_organ, center_of_bbox_organ = bounding_box_calculation(organ_mask)
    
    for i in range(1, gt_les_N+1):
        lesion_gt_mask = (gt_les_numeric == i)
        lesion_size = lesion_gt_mask.sum()
        lesion_volume = lesion_size * per_pix_volume
        lesion_radius = calculate_radius(lesion_volume)

        bbox_interest, size_of_bbox_interest, center_of_bbox_interest = bounding_box_calculation(lesion_gt_mask)

        datum = {
            'case': name.split('/')[-1].rstrip('.nii.gz')+'_'+str(i),
            'lesion_size': lesion_size,
            'lesion_radius': lesion_radius,
            'bb_p_x0': bbox_organ[0][0],
            'bb_p_x1': bbox_organ[0][1],
            'bb_p_y0': bbox_organ[1][0],
            'bb_p_y1': bbox_organ[1][1],
            'bb_p_z0': bbox_organ[2][0],
            'bb_p_z1': bbox_organ[2][1],
            'bb_p_size_x': size_of_bbox_organ[0],
            'bb_p_size_y': size_of_bbox_organ[1],
            'bb_p_size_z': size_of_bbox_organ[2],
            'bb_p_center_x': center_of_bbox_organ[0],
            'bb_p_center_y': center_of_bbox_organ[1],
            'bb_p_center_z': center_of_bbox_organ[2],
            'bb_'+args.target+'_x0': bbox_interest[0][0],
            'bb_'+args.target+'_x1': bbox_interest[0][1],
            'bb_'+args.target+'_y0': bbox_interest[1][0],
            'bb_'+args.target+'_y1': bbox_interest[1][1],
            'bb_'+args.target+'_z0': bbox_interest[2][0],
            'bb_'+args.target+'_z1': bbox_interest[2][1],
            'bb_'+args.target+'_size_x': size_of_bbox_interest[0],
            'bb_'+args.target+'_size_y': size_of_bbox_interest[1],
            'bb_'+args.target+'_size_z': size_of_bbox_interest[2],
            'bb_'+args.target+'_center_x': center_of_bbox_interest[0],
            'bb_'+args.target+'_center_y': center_of_bbox_interest[1],
            'bb_'+args.target+'_center_z': center_of_bbox_interest[2],
            'offset_of_center_pancreas_to_'+args.target+'_x': center_of_bbox_organ[0]-center_of_bbox_interest[0],
            'offset_of_center_pancreas_to_'+args.target+'_y': center_of_bbox_organ[1]-center_of_bbox_interest[1],
            'offset_of_center_pancreas_to_'+args.target+'_z': center_of_bbox_organ[2]-center_of_bbox_interest[2],
            }
        results.append(datum)
    counter.increment()
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='pdac')
    args = parser.parse_args()

    folder_label = '/mnt/ccvl15/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/KidneyDiff_Dataset_Cleaned/0.5mm/label_6cls/*'

    case_list = glob.glob(folder_label)
    case_list.sort()
    counter = Counter()

    with Pool() as pool:
        with tqdm(total=len(case_list)) as pbar:
            async_results = [pool.apply_async(cal_results, (id, counter)) for id in case_list]
            while True:
                completed = counter.value()
                pbar.update(completed - pbar.n)
                if completed == len(case_list):
                    break
            results = [res.get() for res in async_results]


    all_results = []
    for res in results:
        all_results.extend(res)

    result_df = pandas.DataFrame(all_results)
    result_df.to_excel('./05.bbox_'+args.target+'.xlsx')
        





