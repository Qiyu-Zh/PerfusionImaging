import SimpleITK as sitk
import ipywidgets as widgets
from IPython.display import display
import numpy as np
from ipywidgets import interact
from ipywidgets.widgets import IntSlider
import matplotlib.pyplot as plt
import matplotlib
import cv2
import os 
import shutil
from totalsegmentator.python_api import totalsegmentator
from skimage.metrics import structural_similarity 
from scipy import ndimage
import ants
import monai
import monai.metrics
import nibabel as nib
import nibabel.processing
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform


def as_closest_canonical_nifti(path_in, path_out):
    """
    Convert the given nifti file to the closest canonical nifti file.
    """
    img_in = nib.load(path_in)
    img_out = nib.as_closest_canonical(img_in)
    nib.save(img_out, path_out)


def undo_canonical(img_can, img_orig):
    img_ornt = io_orientation(img_orig.affine)
    ras_ornt = axcodes2ornt("RAS")

    to_canonical = img_ornt  # Same as ornt_transform(img_ornt, ras_ornt)
    from_canonical = ornt_transform(ras_ornt, img_ornt)

    # Same as as_closest_canonical
    # img_canonical = img_orig.as_reoriented(to_canonical)

    return img_can.as_reoriented(from_canonical)


def undo_canonical_nifti(path_in_can, path_in_orig, path_out):
    img_can = nib.load(path_in_can)
    img_orig = nib.load(path_in_orig)
    img_out = undo_canonical(img_can, img_orig)
    nib.save(img_out, path_out)
    
def compute_hu_distance(n_classes, y_pred, y):
    data = []
    for i in range(1, n_classes):
        data.append(monai.metrics.compute_hausdorff_distance((y_pred == i), (y == i), include_background=False, distance_metric='euclidean', percentile=None, directed=False, spacing=None))
    return data
    
    

def get_HU_error(y_pred, y, onehot = False, n_classes = None):
    def get_onehot(label, n_classes):
        one_hot = np.eye(n_classes)[label]  # Shape: (1, 32, 32, 32, 6)
        one_hot = one_hot.transpose(3, 0, 1, 2)
        one_hot = np.expand_dims(one_hot, axis=0)

        return one_hot
        
    if onehot is False:
        y = get_onehot(y, n_classes)  # Shape: (1, 32, 32, 32, 6)
        y_pred = get_onehot(y_pred, n_classes) 
    return monai.metrics.compute_hausdorff_distance(y_pred, y, include_background=False, distance_metric='euclidean', percentile=None, directed=False, spacing=None)    
    
    
def erode3d(mask=np.ones((6, 6, 6)), size=2):
    # Ensure mask is boolean
    mask = mask[:]
    
    # Create a 3D structuring element
    structure = np.ones((2 * size + 1, 2 * size + 1, 2 * size + 1), dtype=bool)
    
    # Perform binary erosion
    eroded_mask = ndimage.binary_erosion(mask, structure=structure).astype(mask.dtype)
    
    return eroded_mask

def plot3d(CFR_crop, mask = None, vmax = 2, sample_rate = 5, save_path = None, title = "Interactive 3D CFR Visualization"):
    matplotlib.use('module://ipympl.backend_nbagg')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    x, y, z = np.indices(CFR_crop.shape)

    # Filter the coordinates and CFR values using the mask
    if mask is not None:
        mask = mask[:]
        mask = mask - erode3d(mask, size = 1)
        mask = mask.astype(bool)
        mask_indices = np.where(mask)

    else:
        mask_indices = CFR_crop[:].nonzero()
    x_masked = x[mask_indices]
    y_masked = y[mask_indices]
    z_masked = z[mask_indices]
    CFR_masked = CFR_crop[mask_indices]

    x_masked = x_masked[::sample_rate]
    y_masked = y_masked[::sample_rate]
    z_masked = z_masked[::sample_rate]
    CFR_masked = CFR_masked[::sample_rate]

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title(title)
    # Plot only the masked values
    scatter = ax.scatter(
        x_masked,
        y_masked,
        z_masked,
        c=CFR_masked,
        cmap="jet",
        vmin=0,
        vmax=vmax,
        s=1
    )
    ax.set_xticks([])  
    ax.set_yticks([])  
    ax.set_zticks([])
    ax.set_frame_on(False)

    # Add a colorbar and adjust its position
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    colorbar.set_label("CFR Intensity")
    if save_path:

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    
    
def get_contour(binary_mask):

    # Multiply by 255 to convert it to the format expected by OpenCV (0s and 255s)
    mask_for_cv = binary_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_for_cv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_only_mask = np.zeros_like(binary_mask).copy()
    cv2.drawContours(contour_only_mask, contours, -1, 1, 1)  
    return contour_only_mask


def resize(image_path, target_size, output_path = None):
    image = sitk.ReadImage(image_path)
    original_size = image.GetSize()  
    original_spacing = image.GetSpacing()  
    target_size = list(target_size)
    for i in range(len(target_size)):
    	if target_size[i] == -1:
    	    target_size[i] = original_size[i]	
    new_spacing = [original_spacing[i] * (original_size[i] / target_size[i]) for i in range(len(target_size))]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetInterpolator(sitk.sitkBSpline2)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resized_image = resampler.Execute(image)
    if output_path:
        sitk.WriteImage(resized_image, output_path)
    return resized_image
  
def crop_with_bbox(image_path, bbox, output_path = None):
    bbox = np.array(bbox, dtype= int).tolist()
    image = sitk.ReadImage(image_path)
    # Extract start index and size from the bounding box
    adjusted_bbox = []
    for i in range(3):
        start = bbox[i][0] if bbox[i][0] >= 0 else image_size[i] + bbox[i][0]
        end = bbox[i][1] if bbox[i][1] >= 0 else image_size[i] + bbox[i][1]
        adjusted_bbox.append((start, end))
    bbox = adjusted_bbox
    start_index = [bbox[i][0] for i in range(3)]  # Start index (x_min, y_min, z_min)
    size = [bbox[i][1] - bbox[i][0] for i in range(3)]  # Size (x_max - x_min, y_max - y_min, z_max - z_min)
    
    # Crop the image using RegionOfInterest
    cropped_image = sitk.RegionOfInterest(image, size=size, index=start_index)
    
    # Update the origin based on the start index
    original_origin = image.GetOrigin()
    original_spacing = image.GetSpacing()
    
    # Calculate the new origin based on the cropping
    new_origin = [original_origin[i] + start_index[i] * original_spacing[i] for i in range(3)]
    
    # Set the origin, direction, and spacing of the cropped image
    cropped_image.SetOrigin(new_origin)
    cropped_image.SetDirection(image.GetDirection())
    cropped_image.SetSpacing(image.GetSpacing())
    if output_path:
        sitk.WriteImage(cropped_image, output_path)
    return cropped_image
  
def load_2d_3d(dicom_folder, output_path = None):
    # Use SimpleITK to read the DICOM series
    reader = sitk.ImageSeriesReader()
    # Get the list of DICOM file names from the directory
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_folder)
    # Load the DICOM series
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    if output_path:
        sitk.WriteImage(image, output_path)
    return image

def plot_contour(mask_for_cv, ax, color = "red", label = None):
    contours, _ = cv2.findContours(mask_for_cv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ax.plot([], [], color = color, linewidth = 2, label = label)
    for c in contours:
        
        close_c = np.vstack([c, c[0:1]])
        ax.plot(close_c[:,0,0], close_c[:,0,1], color = color, linewidth = 2)

    return contours
    


def plot_mask(mask_for_cv, ax, color = "Blues", alpha = 0.5, label = None):
    masked_mask = np.ma.masked_where(mask_for_cv == 0, mask_for_cv)
    ax.imshow(masked_mask, alpha=alpha, cmap = color, label = None, vmin = 0, vmax=1)
    return masked_mask


def list_display(img_list, name = "", vmax = 300, cmap = 'jet', read_dcm = False):

    
    if read_dcm:
        img_list = [sitk.GetArrayFromImage(sitk.ReadImage(dcm_file)) for dcm_file in img_list]
    max_slices_contrast = max(img_list, key = lambda x: x.shape[2]).shape[2]

    contrast_slice_slider = widgets.IntSlider(min=0, max=max_slices_contrast-1, step=1, value=0, description='Slice:')
    def display_slice(contrast_slice_index, img_list, name):
        n = len(img_list)
        fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
        fig.suptitle(name + f'  Slice {contrast_slice_index}')
        for i, img in enumerate(img_list):
            if img is not None:
                axes[i].imshow(img[:,:,min(img.shape[2] - 1, contrast_slice_index)], cmap=cmap, vmin = 0, vmax = vmax)
                axes[i].axis('off')
        plt.show()
        
    def update(contrast_slice_index):
        display_slice(contrast_slice_index, img_list, name)

    widgets.interact(update, contrast_slice_index=contrast_slice_slider)


def folders_display(img_list, name = "", read_dcm = False):

    if read_dcm:
        img_list = [sitk.GetArrayFromImage(sitk.ReadImage(dcm_file)) for dcm_file in img_list]
    max_slices_contrast = max(img_list, key = lambda x: x.shape[0]).shape[0]

    contrast_slice_slider = widgets.IntSlider(min=0, max=max_slices_contrast-1, step=1, value=0, description='Slice:')

    def update(contrast_slice_index):

        display_slice(contrast_slice_index, img_list, name)

    widgets.interact(update, contrast_slice_index=contrast_slice_slider)
def display_slice(contrast_slice_index, img_list, name):
    n = len(img_list)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    fig.suptitle(name + f'  Slice {contrast_slice_index}')
    for i, img in enumerate(img_list):
        if img is not None:
            axes[i].imshow(img[min(img.shape[0] - 1, contrast_slice_index),:,:], cmap='jet', vmax = 300)
            axes[i].axis('off')
    plt.show()


def lists_display(img_lists, cmap = 'jet', name = None, titles = None):
    if name == None:
        name = ""
    if titles == None:
        titles = [""] * len(img_lists)
    if isinstance(img_lists[0], str):
        img_lists = [sorted([os.path.join(folder, path) for path in os.listdir(folder) if path.endswith('.nii')]) for folder in img_lists]
        
    if isinstance(img_lists[0][0], str):
        img_lists = [[sitk.GetArrayFromImage(sitk.ReadImage(dcm_file)) for dcm_file in img_list] for img_list in img_lists]
    vmaxs = [300] * len(img_lists)
    for i in range(len(img_lists)):
        if np.max(img_lists[i][0]) < 50:
            vmaxs[i] = np.max(img_lists[i][0])
    max_slices_contrast = max(img_lists[0], key = lambda x: x.shape[0]).shape[0]
    max_num = len(max(img_lists, key = lambda x: len(x)))

    contrast_slice_slider = widgets.IntSlider(min=0, max=max_slices_contrast-1, step=1, value=0, description='Slice:')
    list_idx_slider = widgets.IntSlider(min=0, max=max_num-1, step=1, value=0, description='img_idx:')
    def display_slice(contrast_slice_index, list_idx_slider):
        n = max(len(img_lists), 2)
        fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
        fig.suptitle(name + f'  Slice {contrast_slice_index}')
        for i, imgs in enumerate(img_lists):
            if imgs is not None:
                
                img = imgs[min(len(imgs),list_idx_slider)]
                axes[i].imshow(img[min(img.shape[0] - 1, contrast_slice_index),:,:], cmap=cmap, vmax = vmaxs[i])
                axes[i].axis('off')
                axes[i].set_title(titles[i])
        plt.show()

    def update(list_idx, contrast_slice_index):

        display_slice(contrast_slice_index, list_idx)

    widgets.interact(update, contrast_slice_index=contrast_slice_slider, list_idx = list_idx_slider)
    

def make_if_dont_exist(folder_path,overwrite=False):
    """
    creates a folder if it does not exists
    input: 
    folder_path : relative path of the folder which needs to be created
    over_write :(default: False) if True overwrite the existing folder 
    """
    if os.path.exists(folder_path):
        if overwrite:
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)
    else:
        os.makedirs(folder_path)

def load_3d_2d(dcm_file, output_path):
    make_if_dont_exist(output_path)
    # Read the 3D DICOM image using SimpleITK
    image_3d = sitk.ReadImage(dcm_file)
    # Convert the 3D image to a NumPy array
    image_array_3d = sitk.GetArrayFromImage(image_3d)
    # The shape of the array is (slices, height, width)
    print(f"Shape of the 3D image array: {image_array_3d.shape}")
    spacing = image_3d.GetSpacing()
    # Loop through the slices and process each 2D slice
    for i in range(image_array_3d.shape[0]):
        # Extract 2D slice from the NumPy array
        slice_2d = image_array_3d[i, :, :]
        # Convert the 2D NumPy array back to a SimpleITK image
        slice_image = sitk.GetImageFromArray(slice_2d)
        origin = list(image_3d.GetOrigin())
        origin[2] += i * image_3d.GetSpacing()[2]  # Adjust the Z-axis position for each slice
        slice_image.SetOrigin(origin)
        slice_image.SetSpacing((spacing[0], spacing[1]))
        # Set the Instance Number (or other tags) to reflect the correct slice
        instance_number = i + 1  # Instance numbers usually start at 1
        # Save the slice as a 2D DICOM file
        sitk.WriteImage(slice_image, os.path.join(output_path, os.path.splitext(os.path.basename(dcm_file))[0] + f"_{i}.nii"))


def ssim(np_image1, np_image2):
    
    data_range = np.max([np_image1.max(), np_image2.max()]) - np.min([np_image1.min(), np_image2.min()])
    return structural_similarity(np_image1, np_image2, data_range=data_range)
    
def delete_if_exist(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File {file_path} has been deleted.")
    else:
        print(f"File {file_path} does not exist.")
        
def sitk2ant(img, reverse = False):
    if reverse == True:
        ants.image_write(img, "tmp.nii")
        dcm = sitk.ReadImage("tmp.nii")
        delete_if_exist("tmp.nii")
        return dcm
    else:
        sitk.WriteImage(img, "tmp.nii")
        dcm = ants.image_read("tmp.nii")
        delete_if_exist("tmp.nii")
        return dcm
    
def erode(mask = np.ones((6, 6)), size = 2):
    mask = mask[:]
    structure = np.ones((2*size + 1, 2*size + 1))
    # Erode the mask
    eroded_mask = ndimage.binary_erosion(mask, structure).astype(mask.dtype)
    return eroded_mask
	
