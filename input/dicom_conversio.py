import os
import numpy as np
import pydicom
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, gaussian_filter

def load_dicom_series(directory):
    """Load a DICOM CT series using SimpleITK."""
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(directory)
    if not series_IDs:
        raise ValueError("No DICOM series found in directory.")
    series_file_names = reader.GetGDCMSeriesFileNames(directory, series_IDs[0])
    reader.SetFileNames(series_file_names)
    image = reader.Execute()
    return image

def load_dicom_file(filepath):
    """Load a single DICOM file using pydicom."""
    return pydicom.dcmread(filepath)

def extract_structures(rtstruct, reference_image):
    """Extract contours from RTStruct and convert to binary masks."""
    structure_masks = {}
    for roi in rtstruct.StructureSetROISequence:
        roi_number = roi.ROINumber
        roi_name = roi.ROIName
        for item in rtstruct.ROIContourSequence:
            if item.ReferencedROINumber == roi_number:
                mask = sitk.Image(reference_image.GetSize(), sitk.sitkUInt8)
                mask.CopyInformation(reference_image)
                for contour in item.ContourSequence:
                    points = np.array(contour.ContourData).reshape(-1, 3)
                    indices = [reference_image.TransformPhysicalPointToIndex(tuple(p)) for p in points]
                    for idx in indices:
                        try:
                            mask[idx] = 1
                        except:
                            continue
                structure_masks[roi_name] = mask
    return structure_masks

def load_dose_image(rtdose_path):
    """Load RTDose as a SimpleITK image."""
    dose_ds = pydicom.dcmread(rtdose_path)
    dose_array = dose_ds.pixel_array * dose_ds.DoseGridScaling
    dose_image = sitk.GetImageFromArray(dose_array)
    dose_image.SetSpacing(dose_ds.PixelSpacing + [dose_ds.SliceThickness])
    dose_image.SetOrigin([float(x) for x in dose_ds.ImagePositionPatient])
    return dose_image

def resample_to_reference(image, reference):
    """Resample image to match reference image grid."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if image.GetPixelID() == sitk.sitkUInt8 else sitk.sitkLinear)
    return resampler.Execute(image)

def vol_to_gridpoints(vol, affine):
    dims = np.array(vol.shape)
    center_voxel = (dims + 1) / 2
    center = np.dot(affine, np.append(center_voxel, 1))[:3]
    extents = np.dot(affine[:3, :3], dims)
    signs = np.sign(np.diag(affine[:3, :3]))
    x = np.linspace(center[0] - 0.5 * extents[0], center[0] + 0.5 * extents[0], dims[0]) * signs[0]
    y = np.linspace(center[1] - 0.5 * extents[1], center[1] + 0.5 * extents[1], dims[1]) * signs[1]
    z = np.linspace(center[2] - 0.5 * extents[2], center[2] + 0.5 * extents[2], dims[2]) * signs[2]
    return x, y, z

def save_hedos_inputs(ct_image, structure_masks, dose_image, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    affine = np.eye(4)
    spacing = ct_image.GetSpacing()
    direction = np.array(ct_image.GetDirection()).reshape(3, 3)
    origin = np.array(ct_image.GetOrigin())
    affine[:3, :3] = direction * spacing
    affine[:3, 3] = origin

    dose_array = sitk.GetArrayFromImage(dose_image)
    np.save(os.path.join(output_dir, 'dose.npy'), dose_array)
    np.save(os.path.join(output_dir, 'affine.npy'), affine)

    seg_arrays = {}
    for organ, mask in structure_masks.items():
        mask_array = sitk.GetArrayFromImage(mask)
        seg_arrays[organ] = mask_array.astype(np.uint8)
    np.savez_compressed(os.path.join(output_dir, 'compressed_segs.npz'), **seg_arrays)

# Example usage (update paths accordingly)
ct_dir = "path/to/CT"
rtstruct_path = "path/to/RTStruct.dcm"
rtdose_path = "path/to/RTDose.dcm"
output_dir = "hedos_patient_data"

ct_image = load_dicom_series(ct_dir)
rtstruct = load_dicom_file(rtstruct_path)
rtdose_image = load_dose_image(rtdose_path)

structure_masks = extract_structures(rtstruct, ct_image)

# Resample dose and masks to CT grid
rtdose_image = resample_to_reference(rtdose_image, ct_image)
for organ in structure_masks:
    structure_masks[organ] = resample_to_reference(structure_masks[organ], ct_image)

save_hedos_inputs(ct_image, structure_masks, rtdose_image, output_dir)
print("HEDOS-compatible files saved to:", output_dir)

