import SimpleITK as sitk
import ipywidgets as widgets
from IPython.display import display
import numpy as np
from ipywidgets import interact
from ipywidgets.widgets import IntSlider
import matplotlib.pyplot as plt
import matplotlib
from scipy import ndimage
import pydicom
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
import ants
from skimage.metrics import structural_similarity 
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from tqdm import tqdm

from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from tqdm import tqdm
def classify_CFR_region(stress_values, cfr_values, visualization=False, sample_rate = 1):
    labels = ["Normal Flow", "Minimally Reduced", "Mildly Reduced", 
              "Moderately Reduced Flow Capacity", "Myocardial Steal", 
              "Definite Ischemia", "Predominantly Transmural Myocardial Scar"]
    print("The labels order will be:", labels)
    colors = {
        "Normal Flow": "#FF0000",  # Brown
        "Minimally Reduced": "#FF8C00",  # Orange
        "Mildly Reduced": "#FFD700",  # Yellow
        "Moderately Reduced Flow Capacity": "#008000",  # Green
        "Definite Ischemia": "#0000FF",  # Blue
        "Myocardial Steal": "#4B0082",  # Dark Blue
        "Predominantly Transmural Myocardial Scar": "#000000"  # Black
    }
    
    # Define the CFR regions as polygons
    regions = {
        "Normal Flow": Polygon([(3.37/5, 3.37), (2.39, 3.37), (2.39, 2.39/2), (10, 5), (10, 10), (2, 10)]),
        "Minimally Reduced": Polygon([(1.76, 0.88), (1.76, 2.7), (0.54, 2.7), (3.37/5, 3.37), (2.39, 3.37), (2.39, 2.39/2)]),
        "Mildly Reduced": Polygon([(0.406, 2.03), (1.12, 2.03), (1.12, 0.56), (1.76, 0.88), (1.76, 2.7), (0.54, 2.7)]),
        "Moderately Reduced Flow Capacity": Polygon([(1.74/5, 1.74), (0.91, 1.74), (0.91, 0.91/2), (1.12, 0.56), (1.12, 2.03), (0.406, 2.03)]),
        "Myocardial Steal": Polygon([(1/5, 1), (1.74/5, 1.74), (0.91, 1.74), (0.91, 1)]),
        "Definite Ischemia": Polygon([(0, 0), (1/5, 1), (0.91, 1), (0.91, 0.91/2)]),
        "Predominantly Transmural Myocardial Scar": Polygon([(0, 0), (2, 10), (0, 10)]),
    }

    # Ensure inputs are NumPy arrays and handle 3D arrays
    stress_values = np.array(stress_values)
    cfr_values = np.array(cfr_values)

    # Validate input shapes
    if stress_values.shape != cfr_values.shape:
        raise ValueError("stress_values and cfr_values must have the same shape")

    # Initialize 3D arrays for labels and indices
    label_names = np.empty(stress_values.shape, dtype=object)
    label_indices = stress_values.copy()
    valid_indices = np.argwhere(~np.isnan(stress_values) & ~np.isnan(cfr_values))

    # Iterate only over valid indices
    
    # Iterate over each voxel in the 3D arrays
    if visualization:
        fig, ax = plt.subplots(figsize=(10, 10))
        for label, polygon in regions.items():
            x, y = polygon.exterior.xy  # Get polygon boundary coordinates
            ax.fill(x, y, alpha=1, label=label, color=colors[label])
            
    for i in tqdm(range(len(valid_indices))):
        idx = tuple(valid_indices[i])  # Convert index array to tuple for proper indexing
        stress = stress_values[idx]
        cfr = cfr_values[idx]
        point = Point(stress, cfr)  # Create a point object

        # Check if the point is inside any region
        min_distance = float("inf")
        closest_region = "Outside Defined Regions"
        closest_index = -1

        for label, polygon in regions.items():
            if polygon.contains(point):
                label_names[idx] = label
                label_indices[idx] = labels.index(label)
                break  # Stop checking if a match is found
            else:
                # Compute distance from point to polygon
                distance = polygon.exterior.distance(point)
                if distance < min_distance:
                    min_distance = distance
                    closest_region = label
                    closest_index = labels.index(label)
        else:
            label_names[idx] = closest_region  # Assign the closest region
            label_indices[idx] = closest_index
        if visualization and i % sample_rate == 0:
            ax.scatter(stress, cfr, color='red', marker='o', s=10)

    # Complete visualization setup
    if visualization:
        ax.set_xlabel("Stress Perfusion (cc/min/gm)")
        ax.set_ylabel("Coronary Flow Reserve (CFR)")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.legend(loc="lower right")  # Move legend to lower-left corner
        ax.set_title("CFR Classification Regions")
        plt.show()
    return label_indices, label_names

def scan_time_vector(dcm_files):
    def dcm_time_to_sec(dcm_time):
        hr = float(dcm_time[0:2])
        minute = float(dcm_time[2:4])
        sec = float(dcm_time[4:6])
        ms = float(dcm_time[6:]) if len(dcm_time) > 6 else 0.0  # Handle cases without milliseconds
        return hr * 3600 + minute * 60 + sec + ms
    scan_times = []
    for dcm_file in dcm_files:
        # Read DICOM file
        dcm = pydicom.dcmread(dcm_file)
        # Extract Content Time and convert to seconds
        content_time = dcm.get("ContentTime")  # Default to "000000.00" if missing
        # print(content_time)
        scan_time = dcm_time_to_sec(content_time)
        scan_times.append(scan_time)
    # Sort scan times
    scan_times.sort()
    return scan_times

def get_voxel_size(dicom_file):

    dcm = pydicom.dcmread(dicom_file)

    # Extract the Pixel Spacing and Slice Thickness
    pixel_spacing = dcm.get("PixelSpacing", [1.0, 1.0])  # Default to [1.0, 1.0] if missing
    slice_thickness = dcm.get("SliceThickness", 1.0)     # Default to 1.0 if missing

    # Combine into a single list
    voxel_size = list(pixel_spacing) + [slice_thickness]

    return voxel_size

def compute_aif(dcm, x, y, r):
    
    mean_values = []
    for z in range(dcm.shape[2]):
        # Get the 2D slice
        slice_data = dcm[:, :, z]

        # Create a boolean mask for the circle
        rows, cols = slice_data.shape
        y_indices, x_indices = np.ogrid[:rows, :cols]
        mask = (x_indices - x)**2 + (y_indices - y)**2 <= r**2

        # Extract pixel values inside the circle
        pixel_values = slice_data[mask]

        # Compute the mean value for the circle
        mean_value = np.mean(pixel_values) if np.any(mask) else 0.0  # Handle empty mask case

        # Store the mean value
        mean_values.append(mean_value)

    return mean_values

def gamma(x, p, time_vec_end, aif_vec_end):
    p1, p2 = p
    eps = np.finfo(float).eps  # Small epsilon to avoid division by zero
    # Compute r1
    r1 = (aif_vec_end - p2) / (((time_vec_end / (time_vec_end + eps)) ** p1) * np.exp(p1 * (1 - time_vec_end / (time_vec_end + eps))))
    # Compute r2
    r2 = np.where(
        x == 0,
        0,
        (x / (time_vec_end + eps)) ** p1 * np.exp(p1 * (1 - x / (time_vec_end + eps)))
    )
    return r1 * r2 + p2

def gamma_curve_fit(time_vec_gamma, aif_vec_gamma, time_vec_end, aif_vec_end, p0, lower_bounds=(-100.0, 0.0), upper_bounds=(100.0, 200.0)):
    def gamma_model(x, p1, p2):
        return gamma(x, [p1, p2], time_vec_end, aif_vec_end)
    # Perform curve fitting
    bounds = (lower_bounds, upper_bounds)
    fit, pcov = curve_fit(gamma_model, time_vec_gamma, aif_vec_gamma, p0=p0, bounds=bounds)
    return fit

def gamma_plot(ax1, times2, ss_rest_value2, triger = False , triger_gap = 80, TTP = 7.5):
    baseline_hu = np.mean(ss_rest_value2[0])
    p0 = [0.0, baseline_hu]  # Initial guess (0, blood pool offset)
    idx = np.argmax(ss_rest_value2)
    time_vec_end, aif_vec_end = times2[idx], ss_rest_value2[idx]
    opt_params = gamma_curve_fit(times2, ss_rest_value2, time_vec_end, aif_vec_end, p0)
    x_fit = np.linspace(np.min(times2), times2[-1], 1000)
    y_fit = gamma(x_fit, opt_params, time_vec_end, aif_vec_end)
    
    if triger:
        
        triger_xfit_idx = np.argwhere(y_fit - y_fit[0] - triger_gap >= 0)[0, 0]
        
        
        idx2 = np.argmin(np.abs(times2 - x_fit[triger_xfit_idx] - TTP))
        time_vec_end, aif_vec_end = times2[idx2], ss_rest_value2[idx2]
  
        triger_idx = np.argwhere(times2 - x_fit[triger_xfit_idx] >= 0)[0, 0]
        times2 = times2[:triger_idx] + [x_fit[triger_xfit_idx], time_vec_end]
        ss_rest_value2 = ss_rest_value2[:triger_idx] + [y_fit[triger_xfit_idx], aif_vec_end]
        opt_params = gamma_curve_fit(times2, ss_rest_value2, time_vec_end, aif_vec_end, p0)
        x_fit = np.linspace(np.min(times2), times2[-1], 1000)
        y_fit = gamma(x_fit, opt_params, time_vec_end, aif_vec_end)
        idx = len(times2) - 1
    y_fit_idx = np.argmin(np.abs(y_fit - ss_rest_value2[idx])) 
    baseline_hu = y_fit[0]

    # Adjust y values by subtracting baseline and taking max(0, y)
    dense_y_fit_adjusted = np.maximum(y_fit[:y_fit_idx+1] - baseline_hu, 0)
    
    # Compute the area under the curve using trapezoidal integration
    area_under_curve = trapezoid(dense_y_fit_adjusted, x_fit[:y_fit_idx+1])
    # Generate a dense time vector
    # Compute input concentration
    
      # Adjust for Python indexing (0-based)
    ax1.set_title(f"Fitted AIF Curve with AUC as {area_under_curve:.2f}")
    ax1.set_xlabel("Time Point (s)")
    ax1.set_ylabel("Intensity (HU)")
    # Plot the scatter points
    ax1.scatter(times2, ss_rest_value2, label="Data Points", color="blue")
    # Plot the fitted curve
    ax1.plot(x_fit, y_fit, label="Fitted Curve", color="red")
    # Highlight specific points                
    if triger:
        ax1.scatter(times2[-2], ss_rest_value2[-2], label="Triger", color="green")
    # Add legend
    ax1.legend(loc="upper left")
    # Generate AUC plot
    time_temp = np.linspace(times2[0], times2[-1], int(np.max(times2) * 1))
    # Create a denser AUC plot
    n_points = 1000  # Number of points for denser interpolation
    time_temp_dense = np.linspace(time_temp[0], time_temp[-1], n_points)
    auc_area_dense = gamma(time_temp_dense, opt_params, time_vec_end, aif_vec_end) - baseline_hu

    for i in range(y_fit_idx+1):
        ax1.plot(
            [time_temp_dense[i], time_temp_dense[i]],
            [baseline_hu, auc_area_dense[i] + baseline_hu],
            color="cyan",
            linewidth=1,
            alpha=0.2
        )
    return area_under_curve, idx2 if triger else idx 

def calculate_mean_hu(dcm_rest, dcm_mask_rest, bolus_rest_init, erode_size = 2, visual = False):
    def erode(mask = np.ones((6, 6)), size = 2):
        structure = np.ones((2*size + 1, 2*size + 1))
        # Erode the mask
        eroded_mask = ndimage.binary_erosion(mask, structure).astype(mask.dtype)
        return eroded_mask
    
    def ssim(np_image1, np_image2):
        data_range = np.max([np_image1.max(), np_image2.max()]) - np.min([np_image1.min(), np_image2.min()])
        return structural_similarity(np_image1, np_image2, data_range=data_range)
        
    idxes =  [i for i in range(dcm_rest.shape[2]) if np.sum(dcm_mask_rest[:, :, i]) > 100]
    slice_idx = max([(ssim(dcm_rest[:,:,i], bolus_rest_init), i) for i in idxes])[1]
    reg_ss_rest = ants.registration(fixed = ants.from_numpy(dcm_rest[:, :, slice_idx]) , moving = ants.from_numpy(bolus_rest_init), type_of_transform ='SyNAggro')['warpedmovout']
    
    mask = erode(dcm_mask_rest[:, :, slice_idx], size = erode_size).astype(bool)
    if visual:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 7))
        print(f"The slice number of {slice_idx} is chosen")
        ax0.imshow(dcm_rest[:, :, slice_idx], cmap='gray', vmin=0, vmax=300)
        ax0.imshow(mask[:], alpha= 0.5)
        ax0.set_title("V2 with Mask")

        ax1.imshow(reg_ss_rest[:], cmap='gray', vmin=0, vmax=300)
        ax1.imshow(mask[:], alpha= 0.5)
        ax1.set_title("Registed Surestart with Mask")
    HD_rest = np.mean(reg_ss_rest[:][mask])
    return HD_rest

def compute_organ_metrics(dcm_rest, dcm_mask_rest, v1, input_conc, tissue_rho=1.053):
    dcm_mask_rest = dcm_mask_rest[:].astype(bool)
    try:
        v1_arr = dcm_rest.copy()
        v1_arr[dcm_mask_rest] = v1
    except:
        v1_arr = v1.copy()
    
    voxel_size = dcm_rest.spacing
    v1_arr[~dcm_mask_rest[:].astype(bool)] = np.nan
    dcm_rest[~dcm_mask_rest[:].astype(bool)] = np.nan

    # Compute organ mass (g)
    organ_mass = (
        np.sum(dcm_mask_rest[:]) *
        tissue_rho *
        voxel_size[0] *
        voxel_size[1] *
        voxel_size[2] / 1000
    )

    # Compute delta HU
    delta_hu = np.mean(dcm_rest[dcm_mask_rest]) - np.mean(v1_arr[dcm_mask_rest])

    # Compute organ volume in-plane (cm^2)
    organ_vol_inplane = voxel_size[0] * voxel_size[1] * voxel_size[2]/ 1000

    # Compute V1 and V2 mass
    v1_mass = np.sum(v1_arr[dcm_mask_rest]) * organ_vol_inplane
    v2_mass = np.sum(dcm_rest[dcm_mask_rest]) * organ_vol_inplane

    # Compute flow (mL/min)
    flow = (60 / input_conc) * (v2_mass - v1_mass)

    # Compute flow map (mL/min/g)
    flow_map = (dcm_rest - v1_arr) / (np.mean(dcm_rest[dcm_mask_rest]) - np.mean(v1_arr[dcm_mask_rest])) * flow
    
    flow_std = np.std(flow_map[dcm_mask_rest])

    # Compute perfusion map (normalized by organ mass)
    perf_map = flow_map / organ_mass

    # Compute perfusion standard deviation
    perf_std = np.std(perf_map[dcm_mask_rest])

    # Compute perfusion
    perf = flow / organ_mass

    # Return computed metrics
    metrics = {
        "organ_mass": organ_mass,
        "delta_hu": delta_hu,
        "organ_vol_inplane": organ_vol_inplane,
        "v1_mass": v1_mass,
        "v2_mass": v2_mass,
        "flow": flow,
        "flow_map": flow_map[:],
        "flow_std": flow_std,
        "perf_map": perf_map[:],
        "perf_std": perf_std,
        "perf": perf,
    }
    
    return metrics

def plot3d(CFR_crop, vmax = 2, sample_rate = 5):
    matplotlib.use('module://ipympl.backend_nbagg')

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    x, y, z = np.indices(CFR_crop.shape)

    # Filter the coordinates and CFR values using the mask
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
    ax.set_title("Interactive 3D CFR Visualization")
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


    # Add a colorbar and adjust its position
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    colorbar.set_label("CFR Intensity")
    plt.show()
    
    
    
def mask_fun(img):
    x = img[:].copy()
    x[x > -400] = 1
    x[x < 0] = 0
    return x
