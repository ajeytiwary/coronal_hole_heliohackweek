import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
from skimage.transform import rescale
import numpy as np
from sunpy.map.maputils import all_coordinates_from_map
import sunpy.map
import numpy.ma as ma



def match_disk(fits_file, ref_fits):
  # fits_file is a file path
  # ref_fits is a file path for reference data

  # preps and matches cropped images of EUV data
  # matched_iamge = match_disk('euvi.fits', 'eit.fits')

    sun_map = sunpy.map.Map(fits_file)
    hpc_coords = all_coordinates_from_map(sun_map)
    r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / sun_map.rsun_obs
    mask = ma.masked_less_equal(r, 1)

    sun_map.data[:] -= sun_map.data.min()
    sun_map.data[:] += 10e-6

    if sun_map.detector == 'EUVI':
        sq_img = sun_map.data * sun_map.data
        sun_map.data[:] = sq_img[:]

    if sun_map.detector == 'EIT':
        sq_img = np.sqrt(sun_map.data)
        sun_map.data[:] = sq_img[:]

    masked = sun_map.data * mask.mask
    masked[masked == 0] = np.nan


        # force scale of 1 arcsec per pix
    scale  = 1 / sun_map.scale[1].value

    masked = rescale(masked, sun_map.scale[1].value, anti_aliasing=False)

    crpix = masked.shape[0]/2

    crop = masked[int(crpix - 600):int(crpix + 600), int(crpix - 600):int(crpix + 600)]

    # ------------------------------------
    sun_map = sunpy.map.Map(ref_fits)
    hpc_coords = all_coordinates_from_map(sun_map)
    r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / sun_map.rsun_obs
    mask = ma.masked_less_equal(r, 1)

    sun_map.data[:] -= sun_map.data.min()
    sun_map.data[:] += 10e-6

    if sun_map.detector == 'EUVI':
        sq_img = sun_map.data * sun_map.data
        sun_map.data[:] = sq_img[:]

    if sun_map.detector == 'EIT':
        sq_img = np.sqrt(sun_map.data)
        sun_map.data[:] = sq_img[:]

    masked = sun_map.data * mask.mask
    masked[masked == 0] = np.nan


        # force scale of 1 arcsec per pix
    scale  = 1 / sun_map.scale[1].value

    masked = rescale(masked, sun_map.scale[1].value, anti_aliasing=False)

    crpix = masked.shape[0]/2

    crop_ref = masked[int(crpix - 600):int(crpix + 600), int(crpix - 600):int(crpix + 600)]

    matched = match_histograms(crop, crop_ref)

    fig = plt.figure()

    ax = plt.subplot("131")
    im = ax.imshow(crop)
    ax.set_title('EUVI squared')
    fig.colorbar(im, ax=ax)

    ax = plt.subplot("132")
    im = ax.imshow(matched)
    ax.set_title('EUVI matched')
    fig.colorbar(im, ax=ax)

    ax = plt.subplot("133")
    im = ax.imshow(crop_ref)
    ax.set_title('EIT Square Root')
    fig.colorbar(im, ax=ax)

    return matched
