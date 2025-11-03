import random, cv2, elasticdeform, numpy, os
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool, Manager

'''
    Statistics
'''
statistics_info = {
    'cyst': {
                # Parameters for Shape Genreation
                'size_histogram_cdf': [0.625866050808314, 0.7806004618937644, 0.8383371824480369, 0.8983833718244804, 0.9284064665127021, 0.9445727482678984, 0.9584295612009238, 0.9699769053117783, 0.9838337182448037, 1.0],
                'size_histogram_cutoffs': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

                # Parameters for Position Generation
                # 'offset_z_cdf': [0.06712962962962964, 0.15046296296296297, 0.3055555555555556, 0.4583333333333333, 0.6666666666666666, 0.8078703703703703, 0.9189814814814815, 0.9490740740740741, 0.9953703703703703, 1.0],
                'offset_z_cdf': [0.05555556, 0.1337963, 0.21203704, 0.30092593, 0.39259259, 0.49444444, 0.62638889, 0.825, 0.96712963, 1.],
                'offset_z_curoffs': [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],

                # Parameters for Texture Generation
                'intensity_difference_coefficient_a': 0.7569,
                'intensity_difference_coefficient_b': -3.6755,
                'intensity_difference_range': 0.05,
                'intensity_sigma': 2,
                'index': 4
            },
    'pdac': {
                # Parameters for Shape Genreation
                'size_histogram_cdf': [0.2620689655172414, 0.5300492610837438, 0.6847290640394089, 0.7980295566502463, 0.8837438423645321, 0.9389162561576355, 0.9645320197044335, 0.9852216748768473, 0.994088669950739, 1.0],
                'size_histogram_cutoffs': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

                # Parameters for Position Generation
                # 'offset_z_cdf': [0.0029644268774703555, 0.12944664031620554, 0.3310276679841897, 0.45948616600790515, 0.5464426877470355, 0.6946640316205533, 0.8201581027667985, 0.941699604743083, 0.9881422924901185, 1.0],
                'offset_z_cdf': [0.02630106, 0.06939004, 0.11807499, 0.18522664, 0.26748741, 0.37157247, 0.47621712, 0.74146614, 0.97369894, 1.],
                'offset_z_curoffs': [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],

                # Parameters for Texture Generation
                'intensity_difference_coefficient_a': 0.7898,
                'intensity_difference_coefficient_b': -41.303,
                'intensity_difference_range': 0.05,
                'intensity_sigma': 2,
                'index': 3
            },
    'pnet': {
                # Parameters for Shape Genreation
                'size_histogram_cdf': [0.02630106, 0.06939004, 0.11807499, 0.18522664, 0.26748741, 0.37157247, 0.47621712, 0.74146614, 0.97369894, 1.],
                'size_histogram_cutoffs': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

                # Parameters for Position Generation
                'offset_z_cdf': [0.02630106, 0.06939004, 0.11807499, 0.18522664, 0.26748741, 0.37157247, 0.47621712, 0.74146614, 0.97369894, 1.],
                'offset_z_curoffs': [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5],

                # Parameters for Texture Generation
                'intensity_difference_coefficient_a': 0.7898,
                'intensity_difference_coefficient_b': -41.303,
                'intensity_difference_range': 0.05,
                'intensity_sigma': 2,
                'index': 5
            },
    }

'''
    Supporting functions for Pancreatic Tumor Synthesis
'''
def generate_prob_function(mask_shape):
    sigma = numpy.random.uniform(3,15)
    # uniform noise generate
    a = numpy.random.uniform(0, 1, size=(mask_shape[0],mask_shape[1],mask_shape[2]))

    # Gaussian filter
    a_2 = gaussian_filter(a, sigma=sigma)

    scale = numpy.random.uniform(0.19, 0.21)
    base = numpy.random.uniform(0.04, 0.06)
    a =  scale * (a_2 - numpy.min(a_2)) / (numpy.max(a_2) - numpy.min(a_2)) + base

    return a

def get_texture(mask_shape):
    # get the prob function
    a = generate_prob_function(mask_shape) 

    # sample once
    random_sample = numpy.random.uniform(0, 1, size=(mask_shape[0],mask_shape[1],mask_shape[2]))

    # if a(x) > random_sample(x), set b(x) = 1
    b = (a > random_sample).astype(float)  # int type can't do Gaussian filter

    # Gaussian filter
    if numpy.random.uniform() < 0.7:
        sigma_b = numpy.random.uniform(3, 5)
    else:
        sigma_b = numpy.random.uniform(5, 8)

    # this takes some time
    b2 = gaussian_filter(b, sigma_b)

    # Scaling and clipping
    u_0 = numpy.random.uniform(0.5, 0.55)
    threshold_mask = b2 > 0.12    # this is for calculte the mean_0.2(b2)
    beta = u_0 / (numpy.sum(b2 * threshold_mask) / threshold_mask.sum())
    Bj = numpy.clip(beta*b2, 0, 1)
    
    return Bj

def random_select(mask_scan):
    
    # we first find z index and then sample point with z slice
    z_indexs = np.where(np.any(mask_scan, axis=(0, 1)))[0]
    index_length = len(z_indexs)
    
    coordinates = np.array([])
    while len(coordinates) == 0:
        z = z_indexs[round(random.uniform(0.0, 1.0) * (index_length-1))]
            
        kidney_mask_2d = mask_scan[..., z] 

        # erode the mask (we don't want the edge points)
        kernel = np.ones((5,5), dtype=np.uint8)
        kidney_mask_2d = cv2.erode(np.uint8(kidney_mask_2d), kernel, iterations=1)

        coordinates = np.argwhere(kidney_mask_2d == 1) # TODO: stop here
    
    # print("z_start", z_start, 'z_end', z_end,  "z: ", z, 'len(coordinates): ', len(coordinates))
    # TODO: z_start 0 z_end 181 z:  117 len(coordinates):  0 為什麼中間會有空缺？
    
    random_index = np.random.randint(0, len(coordinates))
    xyz = coordinates[random_index].tolist() # get x,y
    xyz.append(z)
    potential_points = xyz

    return potential_points

def get_ellipsoid(x, y, z):
    # x, y, z is the radius of this ellipsoid in x, y, z direction respectly.
    sh = (4*x, 4*y, 4*z)
    out = numpy.zeros(sh, int)
    aux = numpy.zeros(sh)
    radii = numpy.array([x, y, z])
    com = numpy.array([2*x, 2*y, 2*z])  # center point

    # calculate the ellipsoid 
    bboxl = numpy.floor(com-radii).clip(0,None).astype(int)
    bboxh = (numpy.ceil(com+radii)+1).clip(None, sh).astype(int)
    roi = out[tuple(map(slice,bboxl,bboxh))]
    roiaux = aux[tuple(map(slice,bboxl,bboxh))]
    logrid = *map(numpy.square,numpy.ogrid[tuple(
            map(slice,(bboxl-com)/radii,(bboxh-com-1)/radii,1j*(bboxh-bboxl)))]),
    dst = (1-sum(logrid)).clip(0,None)
    mask = dst>roiaux
    roi[mask] = 1
    numpy.copyto(roiaux,dst,where=mask)
    
    return out

def get_bounds_from_cdf(random_number, cdf, cutoff):
    index = 0
    for i in range(len(cdf)-1):
        if cdf[i]<=random_number and random_number<cdf[i+1]:
            index = i
            break
    
    bound_low = cutoff[index]
    bound_high = cutoff[index+1]

    return bound_low, bound_high

def bounding_box_calculation(mask):
    positions = numpy.where(mask>0)
    bbox = [
        [numpy.min(positions[0]), numpy.max(positions[0])], 
        [numpy.min(positions[1]), numpy.max(positions[1])], 
        [numpy.min(positions[2]), numpy.max(positions[2])], 
    ]

    return bbox

'''
    Shape Generation
'''
def shape_generation(mask, tumor_type, center_bbox_tumor, metadata):
    size_low, size_high = get_bounds_from_cdf(
        random_number = random.random(),
        cdf = statistics_info[tumor_type]['size_histogram_cdf'], 
        cutoff = statistics_info[tumor_type]['size_histogram_cutoffs'],
        )
    
    radius_pancreas_x = (metadata['bbox_pancreas'][0][1]-metadata['bbox_pancreas'][0][0])/2.0
    radius_pancreas_y = (metadata['bbox_pancreas'][1][1]-metadata['bbox_pancreas'][1][0])/2.0
    radius_pancreas_z = (metadata['bbox_pancreas'][2][1]-metadata['bbox_pancreas'][2][0])/2.0

    x = random.randint(int(size_low*radius_pancreas_x), int(size_high*radius_pancreas_x))
    y = random.randint(int(size_low*radius_pancreas_y), int(size_high*radius_pancreas_y))
    z = random.randint(int(size_low*radius_pancreas_z), int(size_high*radius_pancreas_z))
    
    geo = get_ellipsoid(x, y, z)

    sigma = random.randint(1, 2)
    geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,1))
    geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(1,2))
    geo = elasticdeform.deform_random_grid(geo, sigma=sigma, points=3, order=0, axis=(0,2))

    geo_mask = numpy.zeros((
        mask.shape[0] + metadata['enlarge'][0], 
        mask.shape[1] + metadata['enlarge'][1],
        mask.shape[2] + metadata['enlarge'][2]), 
        dtype=numpy.int8)
    
    geo_mask[
        center_bbox_tumor[0]-geo.shape[0]//2:center_bbox_tumor[0]+geo.shape[0]//2,
        center_bbox_tumor[1]-geo.shape[1]//2:center_bbox_tumor[1]+geo.shape[1]//2,
        center_bbox_tumor[2]-geo.shape[2]//2:center_bbox_tumor[2]+geo.shape[2]//2,
        ] += geo

    # as long as we enlarge out sapce before, we need to cut it back
    geo_mask = geo_mask[
        metadata['enlarge'][0]//2:-metadata['enlarge'][0]//2, 
        metadata['enlarge'][1]//2:-metadata['enlarge'][1]//2, 
        metadata['enlarge'][2]//2:-metadata['enlarge'][2]//2,
        ]
    geo_mask = (geo_mask * mask) >=1

    return geo_mask

'''
    Position Generation
'''
def position_generation(mask, tumor_type):
    bbox_pancreas = bounding_box_calculation(mask)
    pancreas_height = bbox_pancreas[2][1] - bbox_pancreas[2][0]
    center_bbox_pancreas_z = (bbox_pancreas[2][1] + bbox_pancreas[2][0])/2.0
    
    # we need to enlarge the sample space to avoid boundary check (which will be very annoying)
    # by enlarge the space, all we need to do is change the place point.
    enlarge_x, enlarge_y, enlarge_z = 150, 150, int(1.3*pancreas_height)
        
    offset_low, offset_high = get_bounds_from_cdf(
        random_number = random.random(),
        cdf = statistics_info[tumor_type]['offset_z_cdf'], 
        cutoff = statistics_info[tumor_type]['offset_z_curoffs'],
        )

    while True:
        center_bbox_tumor = random_select(mask)
        offset_ratio = (center_bbox_tumor[2]-center_bbox_pancreas_z)/pancreas_height
        if offset_low<=offset_ratio and offset_ratio<=offset_high:
            break

    # center_bbox_tumor = [center_bbox_tumor[0] + enlarge_x//2, center_bbox_tumor[1] + enlarge_y//2, center_bbox_tumor[2] + enlarge_z//2]
    
    metadata = {
        'enlarge': [enlarge_x, enlarge_y, enlarge_z],
        'pancreas_height': pancreas_height,
        'bbox_pancreas': bbox_pancreas,
    }
    return center_bbox_tumor, metadata

'''
    Texture Generation
'''
def texture_generation(image, mask, tumor_type, mask_generated):
    texture = get_texture(mask.shape)

    sigma = numpy.random.uniform(1, statistics_info[tumor_type]['intensity_sigma'])
    median_healthy = numpy.median(image[numpy.where((mask_generated>0) & (mask>0))])
    median_target = statistics_info[tumor_type]['intensity_difference_coefficient_a'] * median_healthy + statistics_info[tumor_type]['intensity_difference_coefficient_b']
    range = statistics_info[tumor_type]['intensity_difference_range']
    difference = median_healthy - median_target
    difference = numpy.random.uniform(int((1-range)*difference), int((1+range)*difference))

    # blur the boundary
    geo_blur = gaussian_filter(mask_generated*255, sigma)
    abnormally = (image - texture * (geo_blur/255) * difference) * mask_generated
    
    image = image * (1 - mask_generated) + abnormally
    mask = mask + mask_generated
    return image, mask


def paste_tumor(image, mask, lesion_image, geo_blur, lesion_mask, tumor_type, center_bbox_tumor):
    # TODO: 透过center来粘贴tumor
    
    image = image * (1 - geo_blur) + lesion_image * geo_blur
    mask

'''
    Entry Point Function
'''
def synthesize_pancreatic_tumor(image, mask, tumor_type):
    center_bbox_tumor, metadata = position_generation(mask, tumor_type)
    # TODO: use center to paste tumor from cropped dictionary
    mask_generated = shape_generation(mask, tumor_type, center_bbox_tumor, metadata)
    image, mask = texture_generation(image, mask, tumor_type, mask_generated)
    return image, mask

def paste_pancreatic_tumor(image, mask, tumor_type, tumor_image, tumor_mask, geo_blur):
    geo_blur = (geo_blur-geo_blur.min()) / (geo_blur.max()-geo_blur.min()+1e-8)
    padded_tumor_image = numpy.zeros_like(image)
    padded_tumor_mask = numpy.zeros_like(image)
    padded_tumor_geo_blur = numpy.zeros_like(image)
    while True:
        center_bbox_tumor, metadata = position_generation(mask, tumor_type)
        x, y, z = tumor_image.shape

        if x%2==0:
            x_min, x_max = center_bbox_tumor[0]-x//2, center_bbox_tumor[0]+x//2
        else:
            x_min, x_max = center_bbox_tumor[0]-x//2, center_bbox_tumor[0]+x//2 + 1
        if y%2==0:
            y_min, y_max = center_bbox_tumor[1]-y//2, center_bbox_tumor[1]+y//2
        else:
            y_min, y_max = center_bbox_tumor[1]-y//2, center_bbox_tumor[1]+y//2 + 1
        if z%2==0:
            z_min, z_max = center_bbox_tumor[2]-z//2, center_bbox_tumor[2]+z//2
        else:
            z_min, z_max = center_bbox_tumor[2]-z//2, center_bbox_tumor[2]+z//2 + 1
        
        if x_min>=0 and x_max<=image.shape[0] and y_min>=0 and y_max<=image.shape[1] and z_min>=0 and z_max<=image.shape[2]:
            break
        else:
            print('failed to paste tumor, retry, size:',x, y, z, 'center:', center_bbox_tumor, 'image shape:', image.shape)
    padded_tumor_image[x_min:x_max, y_min:y_max, z_min:z_max] = tumor_image
    padded_tumor_mask[x_min:x_max, y_min:y_max, z_min:z_max] = tumor_mask
    padded_tumor_geo_blur[x_min:x_max, y_min:y_max, z_min:z_max] = geo_blur

    image = image * (1 - padded_tumor_geo_blur) + padded_tumor_image * padded_tumor_geo_blur
    
    mask[np.where(np.uint8(padded_tumor_mask) > 0)] = statistics_info[tumor_type]['index']
    mask = np.uint8(mask)
    
    return image, mask

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

def crop_tumor(root, healthy, syn_pdac_cases, syn_cyst_cases, syn_pnet_cases, counter=None):
    try:
        if healthy in syn_pdac_cases:
            target = 'pdac'
        elif healthy in syn_cyst_cases:
            target = 'cyst'
        else:
            target = 'pnet'
        
        all_tumors = os.listdir(root+'labelsTs_{}/'.format(target))
        v_all_tumors = [x for x in all_tumors if 'VENOUS' in x]
        a_all_tumors = [x for x in all_tumors if 'ARTERIAL' in x]

        healthy_img_path = root+'imagesTr/'+healthy
        healthy_img = nib.load(healthy_img_path).get_fdata()
        healthy_gt_path = root+'labelsTr/'+healthy.replace('_0000', '')
        healthy_gt = nib.load(healthy_gt_path).get_fdata()

        phase = 'ARTERIAL' if 'ARTERIAL' in healthy else 'VENOUS'
        if phase == 'ARTERIAL':
            phase_all_tumors = a_all_tumors
        elif phase == 'VENOUS':
            phase_all_tumors = v_all_tumors

        tumor_num = random.randint(1, 5)
        tumors = random.choices(phase_all_tumors, k=tumor_num)
        
        # tumors = ['FELIX5165_VENOUS_lesion_1_r9.nii.gz']
        for tumor in tumors:
            # print(healthy, tumor_num, 'USE tumor: ', tumor)
            tumor_img_path = root+'labelsTs_{}/'.format(target) + tumor
            tumor_volume = nib.load(tumor_img_path)
            affine = tumor_volume.affine
            hdr = tumor_volume.header
            tumor_img = tumor_volume.get_fdata()
            tumor_gt_path = root+'labelsTs_{}_mask/'.format(target) + tumor
            tumor_gt = nib.load(tumor_gt_path).get_fdata()
            geo_blur_path = root+'labelsTs_{}_blur/'.format(target) + tumor
            geo_blur = nib.load(geo_blur_path).get_fdata()

            healthy_img, healthy_gt = paste_pancreatic_tumor(healthy_img, healthy_gt, target, tumor_img, tumor_gt, geo_blur)

        lesion_path = healthy_img_path.replace('imagesTr', 'Syn/imagesTr_{}'.format(target))
        os.makedirs(os.path.dirname(lesion_path), exist_ok=True)
        lesion_img = nib.Nifti1Image(healthy_img, affine, hdr)
        nib.save(lesion_img, lesion_path)

        lesion_path = healthy_img_path.replace('imagesTr', 'Syn/labelsTr_{}'.format(target)).replace('_0000', '')
        os.makedirs(os.path.dirname(lesion_path), exist_ok=True)
        lesion_img = nib.Nifti1Image(healthy_gt, affine, hdr)
        nib.save(lesion_img, lesion_path)
    except Exception as e:
        print(healthy, e, tumor)
        counter.increment()
        return
    
    counter.increment()
    return 


if __name__ == '__main__':
    import nibabel as nib, numpy as np, shutil
    from tqdm import tqdm
    root = '/mnt/ccvl15/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0001.datasets-nnunet-raw/nnUNet_raw_data/Task1007-YC_3mm_6cls_Remap_dataset/'
    targets = ['pdac', 'cyst', 'pnet']
    # 0.28 0.24 0.18 0.30

    healthy_cases = [x for x in os.listdir(root+'imagesTr/') if not (x.startswith('FELIX5') or x.startswith('FELIX-PDAC') or x.startswith('FELIX-C') or x.startswith('FELIX7'))]
    
    syn_pdac_num = int(len(healthy_cases)*0.28)
    syn_cyst_num = int(len(healthy_cases)*0.24)
    syn_pnet_num = int(len(healthy_cases)*0.18)
    healthy_num = int(len(healthy_cases)*0.30)
    print(syn_pdac_num, syn_cyst_num, syn_pnet_num, healthy_num)
    random.shuffle(healthy_cases)
    
    syn_pdac_cases = healthy_cases[:syn_pdac_num]
    syn_cyst_cases = healthy_cases[syn_pdac_num:syn_pdac_num+syn_cyst_num]
    syn_pnet_cases = healthy_cases[syn_pdac_num+syn_cyst_num:syn_pdac_num+syn_cyst_num+syn_pnet_num]
    syn_healthy_cases = healthy_cases[syn_pdac_num+syn_cyst_num+syn_pnet_num:]
    syn_cases = syn_pdac_cases + syn_cyst_cases + syn_pnet_cases
    
    for healthy_case in tqdm(syn_healthy_cases):
        os.makedirs(root+'Syn/imagesTr_healthy', exist_ok=True)
        os.makedirs(root+'Syn/labelsTr_healthy', exist_ok=True)
        shutil.copy(root+'imagesTr/'+healthy_case, root+'Syn/imagesTr_healthy/'+healthy_case)
        shutil.copy(root+'labelsTr/'+healthy_case.replace('_0000', ''), root+'Syn/labelsTr_healthy/'+healthy_case.replace('_0000', ''))
        
    # syn_cases = ['FELIX0277_VENOUS_0000.nii.gz']
    # 包含V和A，因为我没办法同时在V和A找到一样的tumor然后贴上去
    counter = Counter()
    with Pool() as pool:
        with tqdm(total=len(syn_cases)) as pbar:
            async_results = [pool.apply_async(crop_tumor, (root, id, syn_pdac_cases, syn_cyst_cases, syn_pnet_cases, counter)) for id in syn_cases]
            while True:
                completed = counter.value()
                pbar.update(completed - pbar.n)
                if completed == len(syn_cases):
                    break
    # for x in tqdm(syn_cases):
    #     crop_tumor(root, x, syn_pdac_cases, syn_cyst_cases, syn_pnet_cases)


    