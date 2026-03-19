from imaris_ims_file_reader.ims import ims
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import skimage as sk
import tifffile
import xarray as xr
import pandas as pd
import argparse

def read_image(file):

    if not (isinstance(file, str) or isinstance(file, Path)):
        raise ValueError(f"file input must be a str or Path.")
    elif isinstance(file, str):
        file = Path(file)
    
    if not file.is_file():
        raise FileNotFoundError("'{file}' does not appear to exist or is not a valid file.")
    
    img = ims(file)

    nucl_img = img[0, 0, :, :, :]
    protein_img = img[0, 1, :, :, :]

    return nucl_img, protein_img

def get_mip_image(a, channel):

    mip = np.max(a[0, channel, 7:, :, :], axis=0)

    print(mip.shape)
    return mip

def segment_cells_3d(img):

    # Pre-process the image to normalize intensity
    img_norm = normalize_image_prctile(img, upper=99.5, lower=10)

    thresh = sk.filters.threshold_otsu(img_norm)

    mask = img_norm > thresh

    mask = sk.morphology.opening(mask, sk.morphology.footprint_rectangle((1, 3, 3)))

    for iZ in range(mask.shape[0]):
        mask = sk.morphology.remove_small_holes(mask, max_size=1000)

    labels = sk.measure.label(mask)

    return labels

def segment_blobs(img):

    img_norm = normalize_image_prctile(img, lower=5, upper=99.5)

    # diff_of_gaussians = sk.filters.difference_of_gaussians(img_norm, low_sigma=2, high_sigma=10)

    # thresh = sk.filters.threshold_otsu(diff_of_gaussians)

    # blobs_mask = diff_of_gaussians > thresh

    thresh = sk.filters.threshold_otsu(img_norm)

    blobs_mask = img_norm > thresh

    blobs_mask = sk.morphology.opening(blobs_mask, sk.morphology.footprint_rectangle((1, 3, 3)))

    for iZ in range (blobs_mask.shape[0]):
        blobs_mask[iZ, :, :] = sk.morphology.remove_small_holes(blobs_mask[iZ, :, :],
                                                    max_size=500)
        blobs_mask[iZ, :, :] = sk.morphology.remove_small_objects(blobs_mask[iZ, :, :], max_size=50)
        
    # overlay = sk.segmentation.mark_boundaries(img_norm[9, :, :], blobs_mask[9, :, :])

    # plt.imshow(overlay)
    # plt.show()
    return blobs_mask


def normalize_image(img, max_factor=0.98, min_factor=0.2):

    if not img.dtype == np.float32:
        img = img.astype(np.float32)

    Imax = np.max(img)
    Imin = np.min(img)

    img_norm = (img - Imin)/(Imax - Imin)

    img_norm[img_norm > max_factor] = 1.0
    img_norm[img_norm < min_factor] = 0.0

    return img_norm

def normalize_image_prctile(img, upper=100, lower=2):

    if not img.dtype == np.float32:
        img = img.astype(np.float32)

    upper_value = np.percentile(img, upper)
    lower_value = np.percentile(img, lower)

    img_norm = (img - lower_value)/(upper_value - lower_value)

    img_norm[img_norm > 1.0] = 1.0
    img_norm[img_norm < 0.0] = 0.0

    return img_norm

def analyze_image(file, output_dir, save_data=True):

    if isinstance(file, str):
        file = Path(file)
    elif isinstance(file, Path):
        pass
    else:
        raise ValueError(f"Expected file to be a str or Path. Instead it is a {type(file)}.")

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    elif isinstance(output_dir, Path):
        pass
    else:
        raise ValueError(f"Expected output_dir to be a str or Path. Instead it is a {type(output_dir)}.")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        
    if not output_dir.is_dir():
        raise ValueError(f"The output path {output_dir} does not seem to point to a valid directory")
    
    nucl_img, protein_img = read_image(file)

    cell_labels = segment_cells_3d(nucl_img)
    blob_labels = segment_blobs(protein_img)

    # Measure properties
    props = sk.measure.regionprops(cell_labels, protein_img)

    props = [p for p in props if p.area >= 300]

    red_voxels = np.zeros(len(props))
    volume_ratio = np.zeros(len(props))
    sphericity = np.zeros(len(props))
    for cnt, cell in enumerate(props):

        # Measure the volume of red in the cell
        num_voxels = np.count_nonzero(cell_labels == cell.label)
        num_red_voxels = np.count_nonzero(blob_labels[cell_labels==cell.label])

        volume_ratio[cnt] = num_red_voxels/num_voxels
        red_voxels[cnt] = num_red_voxels

        try:
            # Calculate the sphericity
            verts, faces, _, _ = sk.measure.marching_cubes(cell.image)
            surf_area = sk.measure.mesh_surface_area(verts, faces)
            
            # Apply Wadell Sphericity formula
            sphericity[cnt] = (np.pi**(1/3) * (6 * cell.area)**(2/3)) / surf_area
        except ValueError:
            sphericity[cnt] = 0

    # Generate an xarray Dataset
    ds = xr.Dataset(
        data_vars={"volume": ("index", [p.area for p in props]),
                   "red_volume": ("index", red_voxels),
                   "mean_red_intensity": ("index", [p.intensity_mean for p in props]),
                   "ratio_red_volume": ("index", volume_ratio),
                   "sphericity": ("index", sphericity)
        },
        coords={
            "image": ("index", [file.stem] * len(props)),
            "label": ("index", [p.label for p in props])
        }
    )

    # ---Measure image-based statistics---

    # Measure the volume of red OUTSIDE the green cells
    red_vol_outside_green = blob_labels[cell_labels == 0]
    
    num_red_voxels_outside_green = np.count_nonzero(red_vol_outside_green)

    red_vol_inside_green = blob_labels[cell_labels > 0]
    num_red_voxels_inside_green = np.count_nonzero(red_vol_inside_green)

    #image, num_cells, red_voxels_in_green, red_voxels_outside_Green, green_voxels

    ds_image = xr.Dataset(
        data_vars={"num_cells": ("index", [len(props)]),
                   "num_red_voxels_in_green": ("index", [num_red_voxels_inside_green]),
                   "num_red_voxels_outside_green": ("index", [num_red_voxels_outside_green]),
                   "total_green_voxels": ("index", [np.count_nonzero(cell_labels > 0)])
        },
        coords={
            "image": ("index", [file.stem])
        }
    )

    fn = "results_" + str(file.stem)

    export_tiff_stack(nucl_img, cell_labels, protein_img, blob_labels, props, 
                    (output_dir / (fn + "_labels.tif")))
    
    if save_data:
        save_datasets(ds, ds_image, output_dir=output_dir, filename_prefix=fn)
    else:
        return ds, ds_image

def save_datasets(ds, ds_image, output_dir, filename_prefix=None):

    # Validate inputs
    if not isinstance(ds, xr.Dataset):
        raise ValueError(f"Expected ds to be an xarray Dataset. Instead it is a {type(ds)}.")
    
    if not isinstance(ds_image, xr.Dataset):
        raise ValueError(f"Expected ds_image to be an xarray Dataset. Instead it is a {type(ds_image)}.")
    
    if not isinstance(output_dir, Path):
        raise ValueError(f"Expected output_dir to be a Path. Instead it is a {type(output_dir)}.")
    
    if filename_prefix:
        fn = filename_prefix
    else:
        fn = ""
    
    # Save the main results file
    ds.to_netcdf(output_dir / ("results" + fn + ".nc"))

    df = ds.to_dataframe().reset_index()
    col_order = [
        "image",
        "label",
        "volume",
        "red_volume",
        "ratio_red_volume",
        "mean_red_intensity",
        "sphericity"]
    headers = [
        "Image",
        "Cell ID",
        "Cell Volume (voxel)",
        "Protein Volume (voxel)",
        "Volume Ratio (Protein/Cell)",
        "Mean Protein Intensity",
        "Cell Sphericity"]
    
    df[col_order].to_csv(output_dir / ("results" + fn + ".csv"), header=headers, index=False)

    # Save the image summary file
    ds_image.to_netcdf(output_dir / ("summary" + fn + ".nc"))
                                        
    df_image = ds_image.to_dataframe().reset_index()
    col_order = ["image", 
                "num_cells", 
                "total_green_voxels",
                "num_red_voxels_in_green", 
                "num_red_voxels_outside_green"
                ]
    headers = [
        "Image",
        "Number of Cells",
        "Cell Volume (voxel)",
        "Protein Volume Inside Cell (voxel)",
        "Protein Volume Outside Cell (voxel)"
        ]

    df_image[col_order].to_csv(output_dir / ("summary" + fn + ".csv"), header=headers, index=False)

        



def export_tiff_stack(nucl_img, nucl_labels, protein_img, protein_labels, props, output_fn):

    # Normalize the images
    # nucl_img = normalize_image(nucl_img, max_factor=0.9, min_factor=0.0)
    # protein_img = normalize_image(protein_img, max_factor=0.8, min_factor=0.0)

    nucl_img = normalize_image_prctile(nucl_img, upper=99.9, lower=10)
    protein_img = normalize_image_prctile(protein_img, upper=99.9, lower=10)

    output_slices = []
    for iZ in range(nucl_img.shape[0]):

        # Initialize an empty ndarray for the RGB image
        im_rgb = np.zeros((nucl_img.shape[1], nucl_img.shape[2], 3), dtype=np.float32)

        # Combine the two channels into a single RGB image        
        im_rgb[..., 0] = protein_img[iZ, :, :]
        im_rgb[..., 1] = nucl_img[iZ, :, :]

        # Draw the boundaries
        im_rgb = sk.segmentation.mark_boundaries(im_rgb, nucl_labels[iZ, :, :], color=(1, 1, 0))

        im_rgb = sk.segmentation.mark_boundaries(im_rgb, protein_labels[iZ, :, :], color=(0, 1, 1))

        # plt.imshow(im_rgb)
        # plt.show()

        # Mark each visible object
        fig, ax = plt.subplots(figsize=(im_rgb.shape[1]/100, im_rgb.shape[0]/100), dpi=100)
        plt.subplots_adjust(0,0,1,1)
        ax.axis('off')
        ax.imshow(im_rgb) 

        for prop in props:
            z_min, y_min, x_min, z_max, y_max, x_max = prop.bbox

            if z_min <= iZ < z_max:
                zyx = prop.centroid
                ax.text(zyx[2], zyx[1], str(prop.label), color='white', 
                    fontsize=8, fontweight='bold', ha='center', va='center')

        fig.canvas.draw()

        w, h = fig.canvas.get_width_height()
        # Extract the RGBA buffer and convert to RGB uint8
        rgba_buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgba_array = rgba_buffer.reshape(h, w, 4)

        rgb_array = rgba_array[:, :, :3].copy()
        
        plt.close(fig)

        output_slices.append(rgb_array)
    
    final_stack_image = np.stack(output_slices, axis=0)
    # final_stack_image = (final_stack_image * 255).astype(np.uint8)
    tifffile.imwrite(output_fn, final_stack_image, photometric='rgb')

def analyze_images_in_dir(data_dir, output_dir):

    # Check that the data_dir is a valid directory
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    elif isinstance(data_dir, Path):
        pass
    else:
        raise ValueError(f'Expected data_dir argument to be a str or Path. Instead it is a {type(data_dir)}.')

    if not data_dir.exists():
        raise ValueError(f"The input path '{data_dir}' does not seem to exist.")
    elif not data_dir.is_dir():
        raise ValueError(f"The input path '{data_dir}' does not seem to be a valid directory. To process a single image file, use analyze_image instead.")
    
    # Validate the output directory
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    elif isinstance(output_dir, Path):
        pass
    else:
        raise ValueError(f'Expected output_dir argument to be a str or Path. Instead it is a {type(output_dir)}.')

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    elif not output_dir.is_dir():
        raise ValueError(f"The output path '{output_dir}' does not seem to be a valid directory.")
    
    # Get all image files in directory
    files = list(data_dir.glob("*.ims"))

    ds_list = []
    ds_image_list = []

    for f in files:
        try:
            ds, ds_image = analyze_image(f, output_dir, save_data=False)
        except Exception as e:
            print(f"Error processing file {f}. Error details: {e}.")
            continue

        ds_list.append(ds)
        ds_image_list.append(ds_image)

    # Concatenate and save
    all_ds = xr.concat(ds_list, dim="index", join="outer")
    all_ds_image = xr.concat(ds_image_list, dim="index", join="outer")

    save_datasets(all_ds, all_ds_image, output_dir)

def main():

    parser = argparse.ArgumentParser(description="Process 3D images of macrophages within zebrafish CHT.")

    parser.add_argument("input", help="Path of image file or directory of files.")
    parser.add_argument("output", help="Path to the output destination.")

    args = parser.parse_args()

    ip = Path(args.input)
    
    if ip.exists():

        if ip.is_dir():
            analyze_images_in_dir(ip, args.output)
        elif ip.is_file():
            analyze_image(ip, args.output)
        else:
            raise ValueError('Unknown input type.')
        
    else:
        raise FileNotFoundError(f"The input path '{input}' is not a file or a directory. If the input was a file path, check that you have included the extension.")

if __name__ == "__main__":
    main()
    #analyze_images_in_dir(r"D:\Projects\OIC-264 Magarita\data\2-13-26 GA rapamycin\2026-02-13", r"../processed/2026-03-18 test")

