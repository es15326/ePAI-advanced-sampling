import nibabel as nib, numpy as np, glob, math, argparse, os
from scipy import ndimage
from multiprocessing import Pool, Manager
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

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

def calculate_radius(volume):
    # Calculate the radius using the formula derived above
    radius = ((3 * volume) / (4 * math.pi)) ** (1/3)
    return radius

def crop_tumor(args, path, counter):
    label_data = nib.load(path).get_fdata()
    image_volume = nib.load(path.replace('labelsTs', 'imagesTs'))
    affine, hdr, image_data = image_volume.affine, image_volume.header, image_volume.get_fdata()
    spacing = hdr.get_zooms()
    per_pix_volume = spacing[0] * spacing[1] * spacing[2]

    # Crop the tumor region
    lesion_mask = np.uint8(label_data==label_of_interest[args.target])
    if lesion_mask.sum() == 0:
        counter.increment()
        return

    gt_les_numeric, gt_les_N = ndimage.label(lesion_mask)
    
    for i in range(1, gt_les_N+1):
        
        lesion_gt_mask = np.uint8(gt_les_numeric == i)
        lesion_size = lesion_gt_mask.sum()
        lesion_volume = lesion_size * per_pix_volume
        lesion_radius = calculate_radius(lesion_volume)

        if lesion_size <= 10 or lesion_radius<=1 or lesion_radius>10:
            continue
        sigma = np.random.uniform(1, 2)
        geo_blur = gaussian_filter(lesion_gt_mask*255, sigma)
        geo_blur = (geo_blur-geo_blur.min()) / (geo_blur.max()-geo_blur.min()+1e-8)
        geo_blur_idx = np.where(geo_blur > 0)
        
        cropped_geo_blur = geo_blur[geo_blur_idx[0].min():geo_blur_idx[0].max()+1, 
                                        geo_blur_idx[1].min():geo_blur_idx[1].max()+1, 
                                        geo_blur_idx[2].min():geo_blur_idx[2].max()+1]
        cropped_lesion = image_data[geo_blur_idx[0].min():geo_blur_idx[0].max()+1, 
                                        geo_blur_idx[1].min():geo_blur_idx[1].max()+1, 
                                        geo_blur_idx[2].min():geo_blur_idx[2].max()+1]
        cropped_lesion_gt_mask = lesion_gt_mask[geo_blur_idx[0].min():geo_blur_idx[0].max()+1, 
                                                geo_blur_idx[1].min():geo_blur_idx[1].max()+1, 
                                                geo_blur_idx[2].min():geo_blur_idx[2].max()+1]
        
        # Save the cropped tumor region
        lesion_path = path.replace('labelsTs', 'labelsTs_{}_blur'.format(args.target)).replace('.nii.gz', '_lesion_'+str(i)+'_r'+str(int(lesion_radius))+'.nii.gz')
        os.makedirs(os.path.dirname(lesion_path), exist_ok=True)
        lesion_img = nib.Nifti1Image(cropped_geo_blur, affine, hdr)
        nib.save(lesion_img, lesion_path)

        lesion_path = path.replace('labelsTs', 'labelsTs_{}_mask'.format(args.target)).replace('.nii.gz', '_lesion_'+str(i)+'_r'+str(int(lesion_radius))+'.nii.gz')
        os.makedirs(os.path.dirname(lesion_path), exist_ok=True)
        lesion_img = nib.Nifti1Image(cropped_lesion_gt_mask, affine, hdr)
        nib.save(lesion_img, lesion_path)

        lesion_path = path.replace('labelsTs', 'labelsTs_{}'.format(args.target)).replace('.nii.gz', '_lesion_'+str(i)+'_r'+str(int(lesion_radius))+'.nii.gz')
        os.makedirs(os.path.dirname(lesion_path), exist_ok=True)
        lesion_img = nib.Nifti1Image(cropped_lesion, affine, hdr)
        nib.save(lesion_img, lesion_path)

    counter.increment()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='pnet')
    args = parser.parse_args()
    
    root_path = '/data/yucheng/KidneyDiff_Dataset/nnUNet_Dataset/0001.datasets-nnunet-raw/nnUNet_raw_data/Task1007-YC_3mm_6cls_Remap_dataset/labelsTs/'
    # case_list = glob.glob(root_path)
    case_list = os.listdir(root_path)

    if args.target == 'pdac':
        case_list = [root_path + x for x in case_list if (x.startswith('FELIX5') or x.startswith('FELIX-PDAC'))]
    elif args.target == 'cyst':
        case_list = [root_path + x for x in case_list if x.startswith('FELIX-C')]
    elif args.target == 'pnet':
        case_list = [root_path + x for x in case_list if x.startswith('FELIX7')]
    else:
        raise ValueError('Invalid target')
    
    case_list.sort()

    counter = Counter()
    with Pool() as pool:
            with tqdm(total=len(case_list)) as pbar:
                async_results = [pool.apply_async(crop_tumor, (args, id, counter)) for id in case_list]
                while True:
                    completed = counter.value()
                    pbar.update(completed - pbar.n)
                    if completed == len(case_list):
                        break
                
        
