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
    for z in range(dcm.shape[0]):
        # Get the 2D slice
        slice_data = dcm[z, :, :]

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

def compute_organ_metrics(dcm_rest, dcm_mask_rest, v1_arr, time_vec_gamma_rest, HU_sure_mean, input_conc, tissue_rho=1.053):
    voxel_size = dcm_rest.spacing

    # Compute delta time
    delta_time = time_vec_gamma_rest[-1] - time_vec_gamma_rest[-2]

    # Compute heart rate
    heart_rate = round(1 / (np.mean(np.diff(time_vec_gamma_rest)) / 60))

    # Compute organ mass (g)
    organ_mass = (
        np.sum(dcm_mask_rest[:]) *
        tissue_rho *
        voxel_size[0] *
        voxel_size[1] *
        voxel_size[2] / 1000
    )

    # Compute delta HU
    delta_hu = np.mean(dcm_rest[dcm_mask_rest]) - HU_sure_mean

    # Compute organ volume in-plane (cm^2)
    organ_vol_inplane = voxel_size[0] * voxel_size[1] / 1000

    # Compute V1 and V2 mass
    v1_mass = np.sum(v1_arr[dcm_mask_rest]) * organ_vol_inplane
    v2_mass = np.sum(dcm_rest[dcm_mask_rest]) * organ_vol_inplane

    # Compute flow (mL/min)
    flow = (1 / input_conc[0]) * ((v2_mass - v1_mass) / (delta_time / 60))

    # Compute flow map (mL/min/g)
    flow_map = (dcm_rest - v1_arr) / (np.mean(dcm_rest[dcm_mask_rest]) - HU_sure_mean) * flow

    # Compute perfusion map (normalized by organ mass)
    perf_map = flow_map / organ_mass

    # Compute perfusion standard deviation
    perf_std = np.std(perf_map[dcm_mask_rest])

    # Compute perfusion
    perf = flow / organ_mass

    # Return computed metrics
    metrics = {
        "delta_time": delta_time,
        "heart_rate": heart_rate,
        "organ_mass": organ_mass,
        "delta_hu": delta_hu,
        "organ_vol_inplane": organ_vol_inplane,
        "v1_mass": v1_mass,
        "v2_mass": v2_mass,
        "flow": flow,
        "flow_map": flow_map,
        "perf_map": perf_map,
        "perf_std": perf_std,
        "perf": perf,
    }
    
    return metrics
