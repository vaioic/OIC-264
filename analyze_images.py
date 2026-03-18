from imaris_ims_file_reader.ims import ims
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import skimage as sk
import tifffile
import xarray as xr
import pandas as pd

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

    # mask = sk.morphology.opening(mask, sk.morphology.ball(2))
    #mask = sk.morphology.remove_small_holes(mask, max_size=1000)

    labels = sk.measure.label(mask)

    # overlay = sk.segmentation.mark_boundaries(img_norm[9, :, :], labels[9, :, :])
    # plt.imshow(overlay)
    # plt.show()
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

def analyze_image(file, output_dir):

    if isinstance(file, str):
        file = Path(file)
    else:
        raise ValueError(f"Expected file to be a string.")

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    else:
        raise ValueError(f"Expected output_dir to be a string.")
    
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

    volume_ratio = np.zeros(len(props))
    sphericity = np.zeros(len(props))
    for cnt, cell in enumerate(props):

        # Measure the volume of red in the cell
        num_voxels = np.count_nonzero(cell_labels == cell.label)
        num_red_voxels = np.count_nonzero(blob_labels[cell_labels==cell.label])

        volume_ratio[cnt] = num_red_voxels/num_voxels

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
                   "mean_red_intensity": ("index", [p.intensity_mean for p in props]),
                   "ratio_red_volume": ("index", volume_ratio),
                   "sphericity": ("index", sphericity)
        },
        coords={
            "image": ("index", [file.stem] * len(props)),
            "label": ("index", [p.label for p in props])
        }
    )
    
    # TODO: Measure the volume of red objects OUTSIDE the green cells
    # I think something similar to the way I calculated the ratio previously

    # # Filter out objects which are too small
    # ds_filtered = ds.sel(index=(ds.volume >= 1000))

    # Write data
    fn = "results_" + str(file.stem)
    ds.to_netcdf(output_dir / (fn + ".nc"))

    df = ds.to_dataframe()
    df.to_csv(output_dir / (fn + ".csv"))
                                         
    # in each macrophage, how much red signal?
   
    export_tiff_stack(nucl_img, cell_labels, protein_img, blob_labels, props, 
                      (output_dir / (fn + "_labels.tif")))

    # output_slices = []
    # for iZ in range(cell_labels.shape[0]):
    #     im = normalize_image(nucl_img)
    #     overlay = sk.segmentation.mark_boundaries(im[iZ, :, :], cell_labels[iZ, :, :])

    #     output_slices.append(overlay)

    # final_stack_image = np.stack(output_slices, axis=0)
    # tifffile.imwrite('test_stack.tif', output_slices, photometric='rgb')
    
    # plt.imshow(overlay)
    # plt.show()

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


if __name__ == "__main__":
    analyze_image(r"D:\Projects\OIC-264 Magarita\data\2-13-26 GA rapamycin\2026-02-13\1 uM rapa_1.ims", r"../processed/2026-03-18 test")

