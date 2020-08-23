# Imports
import numpy as np
import numpy.ma as ma
import sunpy.map
import sunpy.sun
from sunpy.map.maputils import all_coordinates_from_map
from sunpy.coordinates import get_horizons_coord
import glob
import hamada_hist_matching
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

def wavelet_enhancement(img):
    
    # incomplete
    img_enhanced = img
    
    return img_enhanced

# Masking function
def mask_outside_disk(inst_map):
    # Find coordinates and radius
    hpc_coords = all_coordinates_from_map(inst_map)
    r = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2) / inst_map.rsun_obs
    
    # Mask everything outside of the solar disk
    mask = ma.masked_greater_equal(r, 1)
    ma.set_fill_value(mask, np.nan)
    where_disk = np.where(mask.mask == 1)
    
    return where_disk


def make_hist(norm_log_inst, bins_inst):
    
    where_mask = np.isfinite(norm_log_inst)
    arr = norm_log_inst[where_mask]
    hist, bins = np.histogram(arr, bins=bins_inst, density=True)
    width = bins[1] - bins[0]
    
    return hist*width


# Nb. of channels
nb_channels = 3
list_channels = [171, 195, 304]

# Find sample data
path_to_files = '/home/btremblay/Documents/dir.HelioHackWeek/sampledata/'
filenames_eit_0 = sorted(glob.glob(path_to_files+'eit_l1*'))
filenames_euvil_0 = sorted(glob.glob(path_to_files+'*eu_L.fts'))
filenames_euvir_0 = sorted(glob.glob(path_to_files+'*eu_R.fts'))
# Benoit: Needs fixing
filenames_eit_1 = sorted(glob.glob(path_to_files+'eit_l1*'))
filenames_euvil_1 = sorted(glob.glob(path_to_files+'*eu_L.fts'))
filenames_euvir_1 = sorted(glob.glob(path_to_files+'*eu_R.fts'))
# Benoit: Needs fixing
filenames_eit_2 = sorted(glob.glob(path_to_files+'eit_l1*'))
filenames_euvil_2 = sorted(glob.glob(path_to_files+'*eu_L.fts'))
filenames_euvir_2 = sorted(glob.glob(path_to_files+'*eu_R.fts'))
# Nb. of samples to work with
nb_files = len(filenames_euvil_0)

# Hamada class for homogenization
filename = 'cumulative_hist.npz'
hamada_model = hamada_hist_matching.hamada(filename='cumulative_hist.npz')

# Output
output_path = ''

for file_nb in range(nb_files):
    
    # Make map objects for one channel
    eit_maps = sunpy.map.Map(filenames_eit_0[file_nb])
    euvil_maps = sunpy.map.Map(filenames_euvil_0[file_nb])
    euvir_maps = sunpy.map.Map(filenames_euvir_0[file_nb])
    filename_extract = filenames_eit_0[file_nb].split(path_to_files)
    filename_output = output_path + 'composite_' + filename_extract[1]

    if eit_maps.observatory in ['SOHO']:
        new_coords = get_horizons_coord(eit_maps.observatory.replace(' ', '-'),
                                        eit_maps.date)
        eit_maps.meta['HGLN_OBS'] = new_coords.lon.to('deg').value
        eit_maps.meta['HGLT_OBS'] = new_coords.lat.to('deg').value
        eit_maps.meta['DSUN_OBS'] = new_coords.radius.to('m').value
        eit_maps.meta.pop('hec_x')
        eit_maps.meta.pop('hec_y')
        eit_maps.meta.pop('hec_z')
    
    # Check positioning of instruments in order to cover full Sun
    long_eit = eit_maps.observer_coordinate.lon.to('degree')
    print(long_eit.value)
    long_euvil = euvil_maps.observer_coordinate.lon.to('degree')-long_eit
    print(long_euvil.value)
    long_euvir = euvir_maps.observer_coordinate.lon.to('degree')-long_eit
    print(long_euvir.value)
    if -90 < long_euvil.value < 90 and -90 < long_euvir.value < 90:
        position_condition = 0
    elif 90 < long_euvil.value < 180 and long_euvil.value-180 < long_euvir.value < 180:
        position_condition = 0
    elif 90 < long_euvir.value < 180 and long_euvir.value-180 < long_euvil.value < 180:
        position_condition = 0
    elif -180 < long_euvil.value < -90 and -180 < long_euvir.value < long_euvil.value+180:
        position_condition = 0
    elif -180 < long_euvir.value < -90 and -180 < long_euvil.value < long_euvir.value+180:
        position_condition = 0
    else:
        position_condition = 1
    
    # If positioning is ideal, continue # MODIFY
    if position_condition == 1:
        
        # continue
        for channel_nb in range(nb_channels):
            # Make map objects
            if channel_nb == 1:
                eit_maps = sunpy.map.Map(filenames_eit_1[file_nb])
                euvil_maps = sunpy.map.Map(filenames_euvil_1[file_nb])
                euvir_maps = sunpy.map.Map(filenames_euvir_1[file_nb])
                filename_extract = filenames_eit_1[file_nb].split(path_to_files)
                filename_output = output_path + 'composite_' + filename_extract[1]
            elif channel_nb == 2:
                eit_maps = sunpy.map.Map(filenames_eit_2[file_nb])
                euvil_maps = sunpy.map.Map(filenames_euvil_2[file_nb])
                euvir_maps = sunpy.map.Map(filenames_euvir_2[file_nb])
                filename_output = output_path+'composite_' + filenames_eit_2[file_nb]
                filename_extract = filenames_eit_2[file_nb].split(path_to_files)
                filename_output = output_path + 'composite_' + filename_extract[1]
                
            # Properties
            nx_eit, ny_eit = eit_maps.data.shape
            nx_euvil, ny_euvil = euvil_maps.data.shape
            nx_euvir, ny_euvir = euvir_maps.data.shape
            
            # Adjust SoHO if not already done
            if channel_nb > 0 and eit_maps.observatory in ['SOHO']:
                new_coords = get_horizons_coord(eit_maps.observatory.replace(' ', '-'),
                                                eit_maps.date)
                eit_maps.meta['HGLN_OBS'] = new_coords.lon.to('deg').value
                eit_maps.meta['HGLT_OBS'] = new_coords.lat.to('deg').value
                eit_maps.meta['DSUN_OBS'] = new_coords.radius.to('m').value
                eit_maps.meta.pop('hec_x')
                eit_maps.meta.pop('hec_y')
                eit_maps.meta.pop('hec_z')

            # Mask everything outside the solar disk
            where_mask = mask_outside_disk(eit_maps)
            eit_maps.data[where_mask] = np.nan
            where_mask = mask_outside_disk(euvil_maps)
            euvil_maps.data[where_mask] = np.nan
            where_mask = mask_outside_disk(euvir_maps)
            euvir_maps.data[where_mask] = np.nan
            
            # Wavelet enchancement of the EIT data for improved contrast
            # Missing Step ######################
            # arr_tmp = eit_maps.data
            # eit_maps.data = wavelet_enhancement(arr_tmp)
            # Homogenization of the EIT data /w respect to EUVI
            arr_tmp = eit_maps.data
            where_mask = np.isfinite(arr_tmp)
            arr_tmp = hamada_model.hist_matching(arr_tmp, channel_nb)
            eit_maps.data[where_mask] = arr_tmp[where_mask]
            
            # Combined maps
            maps = [eit_maps, euvil_maps, euvir_maps]
            
            # Combined maps
            shape_out = (180, 360)  # This is set deliberately low to reduce memory consumption
            header = sunpy.map.make_fitswcs_header(shape_out,
                                                   SkyCoord(0, 0, unit=u.deg,
                                                            frame="heliographic_stonyhurst",
                                                            obstime=maps[0].date),
                                                   scale=[180 / shape_out[0],
                                                          360 / shape_out[1]] * u.deg / u.pix,
                                                   wavelength=int(maps[0].meta['wavelnth']) * u.AA,
                                                   projection_code="CAR")
            out_wcs = WCS(header)
            coordinates = tuple(map(sunpy.map.all_coordinates_from_map, maps))
            weights = [coord.transform_to("heliocentric").z.value for coord in coordinates]
            weights = [(w / np.nanmax(w)) ** 3 for w in weights]
            for w in weights:
                w[np.isnan(w)] = 0
            
            array, _ = reproject_and_coadd(maps, out_wcs, shape_out,
                                           input_weights=weights,
                                           reproject_function=reproject_interp,
                                           match_background=True,
                                           background_reference=0)
        
            outmap = sunpy.map.Map((array, header))
            outmap.plot_settings = maps[0].plot_settings
            outmap.nickname = 'AIA + EUVI/A + EUVI/B'
        
            # Output
            outmap.save(filename_output, filetype='fits', overwrite=True)
        
#############
# End of file
#############
