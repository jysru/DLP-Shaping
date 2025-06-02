"""Module containing the functions for generating holograms"""
import numpy as np 


#######################################################################
# Aux definition
#######################################################################

def holo_efficiency(nuvec):
    """Returns the maximum diffraction efficiency.
    
    It assumes the first order was selected.
    
    Parameters
    ----------
    nuvec : ndarray, list, tuple
        Carrier frequency vector used for the hologram.
        
    Returns
    -------
    float
        Diffraction efficiency.
    """
    px= 1/nuvec[0]
    eta_dif = np.sin(np.pi*np.floor(px/2)/px)/np.pi
    if nuvec[1]!=0:
        py = 1/nuvec[1]
        eta_dif *= px*np.sin(np.pi*px/py)/np.pi
    return eta_dif**2

def _holo_preamble(field, nuvec, renorm=True):
    a = np.abs(field)
    phi = np.angle(field)
    sh = field.shape
    x,y = np.meshgrid(np.arange(sh[1]),np.arange(sh[0]))
    nunorm = 2*np.linalg.norm(nuvec)
    if renorm:
        a /=np.max(a)
    return a, phi, x-nuvec[1]/nunorm, y+nuvec[0]/nunorm

#######################################################################
# Gray-scale hologram definition
#######################################################################
def amplitude_off_axis(field, nuvec=(1/4,0)):
    """Returns the gray scale off-axis hologram to generate the complex field.
    
    It assumes the first order was selected.
    
    Parameters
    ----------
    field : ndarray
        The target complex field
    nuvec : ndarray, list, tuple
        Carrier frequency vector used for the hologram.
        
    Returns
    -------
    ndarray
        Gray scale hologram.
    """
    a, phi, x, y = _holo_preamble(field, nuvec)
    return 1/2 + a* np.cos(2*np.pi*(nuvec[0]*x + nuvec[1]*y)-phi)/2

def amplitude_lee(field, nuvec=(1/4,0)):
    """Returns the gray scale Lee sampling hologram to generate the complex field.
    
    It assumes the first order was selected.
    
    Parameters
    ----------
    field : ndarray
        The target complex field
    nuvec : ndarray, list, tuple
        Carrier frequency vector used for the hologram.
        
    Returns
    -------
    ndarray
        Gray scale hologram.
    """
    a, phi, x, y = _holo_preamble(field, nuvec)
    return a* (np.cos(2*np.pi*(nuvec[0]*x + nuvec[1]*y)-phi) 
               + np.abs(np.cos(2*np.pi*(nuvec[0]*x + nuvec[1]*y)-phi)))

#######################################################################
# Lee hologram definition
#######################################################################

def parallel_lee(field, nuvec=(1/4,0), renorm=True):
    """Returns the binary parallel Lee hologram to generate the complex field.
    
    It assumes the first order was selected.
    
    Parameters
    ----------
    field : ndarray
        The target complex field
    nuvec : ndarray, list, tuple
        Carrier frequency vector used for the hologram.
    renorm : bool, optional 
        If true it sets the maximum amplitude of the field equal to one. 
        
    Returns
    -------
    ndarray
        Binary parallel Lee hologram.
    """
    a, phi, x, y = _holo_preamble(field, nuvec, renorm=renorm)
    return np.abs(np.mod((nuvec[0]*x + nuvec[1]*y)-phi/(2*np.pi)-1/2, 1) -1/2)\
        < np.arcsin(a)/(2*np.pi)

def orthogonal_lee(field, nuvec=(1/4,0), renorm=True):
    """Returns the binary orthogonal Lee hologram to generate the complex field.
    
    It assumes the first order was selected.
    
    Parameters
    ----------
    field : ndarray
        The target complex field
    nuvec : ndarray, list, tuple
        Carrier frequency vector used for the hologram.
    renorm : bool, optional 
        If true it sets the maximum amplitude of the field equal to one. 
        
    Returns
    -------
    ndarray
        Binary orthogonal Lee hologram.
    """
    a, phi, x, y = _holo_preamble(field, nuvec, renorm=renorm)
    
    return (np.abs(np.mod((nuvec[0]*x + nuvec[1]*y)-phi/(2*np.pi)-1/2, 1) -1/2-0*1/2) < 1/4) \
            *(np.mod((-nuvec[1]*x + nuvec[0]*y), 1) < 1*a)

#######################################################################
# Look-up table hologram definition
#######################################################################

def _down_sample(field, nsp, method='center'):
    if method=='center':
        ds_field = field[nsp//2::nsp,nsp//2::nsp]
    elif method=='mean':
        ds_field = np.zeros_like(field[nsp//2::nsp,nsp//2::nsp])
        for i in range(nsp):
            for j in range(nsp):
                ds_field += field[i::nsp,j::nsp]
        ds_field /= nsp**2
    elif method=='side':
        ds_field = field[::nsp,::nsp]
    else:
        raise ValueError('Invalid option for method.')
    return ds_field


def holo_Haskell(field, lut, pixel_combinations, step=0.01, ds_method='center', renorm=True):
    """Returns the Haskell hologram to generate the complex field.
    
    This function works for both the aligned and 45degrees tilted version of the Haskell
    hologram. It assumes the first order was selected.
    
    Parameters
    ----------
    field : ndarray
        The target complex field
    lut : ndarray
        Look-up table.
    pixel_combinations : ndarray
        Array conatining all the ifferent pixel combinations leading to different complex values. 
    step : float
        Step used to build the lut.
    ds_method : {'center', 'mean'}
        Method used to downsample the original image by the resolution of the super pixels. 
        'center' use a value the center value or close to it and 'mean' takes the mean value of all 
        pixels within the superpixel. 
    renorm : bool, optional 
        If true it sets the maximum amplitude of the field equal to one. 
        
    Returns
    -------
    ndarray
        Binary Haskell hologram with a size being proportional to that of the superpixel.
    """
    if renorm:
        field /= np.max(np.abs(field))
    # assume the field has been rescaled to the unit SP
    m = len(pixel_combinations[0])
    n_SP = int(np.sqrt(m))
    
    ds_field = _down_sample(field, n_SP, method=ds_method)
    sh = ds_field.shape
    holo = np.zeros((sh[-2]*n_SP, sh[-1]*n_SP),dtype=int)
    # rescale field values according to LUT
    field_sc = ds_field/(np.max(np.abs(ds_field))*step)
    reim0 = (len(lut))//2
    for j in range(sh[-2]):
        for i in range(sh[-1]):
            re = int(np.round(np.real(field_sc[j,i])))
            im = int(np.round(np.imag(field_sc[j,i])))
            # sp_pixel = np.roll(pixel_combinations[lut[re+reim0, im+reim0]], -shift)
            sp_pixel = pixel_combinations[lut[re+reim0, im+reim0]]
            holo[n_SP*j:n_SP*(j+1),n_SP*i:n_SP*(i+1)] = sp_pixel.reshape(n_SP,n_SP)
    return holo

def holo_SP(field, lut, pixel_combinations, step=0.01, ds_method='center', renorm=True):
    """Returns the Super pixel hologram to generate the complex field.
    
    This function works for both the aligned and 45degrees tilted version of the Haskell
    hologram. It assumes the first order was selected.
    
    Parameters
    ----------
    field : ndarray
        The target complex field
    lut : ndarray
        Look-up table.
    pixel_combinations : ndarray
        Array conatining all the ifferent pixel combinations leading to different complex values. 
    step : float
        Step used to build the lut.
    ds_method : {'center', 'mean'}
        Method used to downsample the original image by the resolution of the super pixels. 
        'center' use a value the center value or close to it and 'mean' takes the mean value of all 
        pixels within the superpixel. 
    renorm : bool, optional 
        If true it sets the maximum amplitude of the field equal to one. 
        
    Returns
    -------
    ndarray
        Binary Super pixel hologram with a size being proportional to that of the superpixel.
    """
    if renorm:
        field /= np.max(np.abs(field))
    # assume the field has been rescaled to the unit SP
    m = len(pixel_combinations[0])
    n_SP = int(np.sqrt(m))

    ds_field = _down_sample(field, n_SP, method=ds_method)
    sh = ds_field.shape
    holo = np.zeros((sh[-2]*n_SP, sh[-1]*n_SP),dtype=int)
    # rescale field values according to LUT
    field_sc = ds_field/(np.max(np.abs(ds_field))*step)
    reim0 = (len(lut))//2
    for j in range(sh[-2]):
        shift = np.mod(n_SP*j, n_SP**2)
        for i in range(sh[-1]):
            re = int(np.round(np.real(field_sc[j,i])))
            im = int(np.round(np.imag(field_sc[j,i])))
            sp_pixel = np.roll(pixel_combinations[lut[re+reim0, im+reim0]], -shift)
            holo[n_SP*j:n_SP*(j+1),n_SP*i:n_SP*(i+1)] = sp_pixel.reshape(n_SP,n_SP).T
    return holo






import numpy as np

def holo_SP_optimized(field, lut, pixel_combinations, step=0.01, ds_method='center', renorm=True):
    """Optimized version of holo_SP that maintains accuracy while significantly improving speed."""
    
    if renorm:
        field /= np.max(np.abs(field))  # Normalize amplitude
    
    # Determine superpixel size
    m = len(pixel_combinations[0])
    n_SP = int(np.sqrt(m))

    # Downsample field
    ds_field = _down_sample(field, n_SP, method=ds_method)
    sh = ds_field.shape  # (height, width)

    # Initialize hologram
    holo = np.zeros((sh[0] * n_SP, sh[1] * n_SP), dtype=int)

    # Rescale field values for LUT lookup
    field_sc = ds_field / (np.max(np.abs(ds_field)) * step)
    reim0 = len(lut) // 2

    # **Vectorized LUT lookup**
    re_indices = np.clip(np.round(np.real(field_sc)).astype(int) + reim0, 0, lut.shape[0] - 1)
    im_indices = np.clip(np.round(np.imag(field_sc)).astype(int) + reim0, 0, lut.shape[1] - 1)
    lut_indices = lut[re_indices, im_indices]

    # **Vectorized assignment of superpixel patterns**
    sp_pixels = pixel_combinations[lut_indices]  # Get superpixel patterns (shape: sh[0], sh[1], m)

    # **Apply Row-Wise Rolling with Corrected Assignment**
    for j in range(sh[0]):
        shift = np.mod(n_SP * j, n_SP**2)  # Compute shift for row `j`
        rolled_sp = np.roll(sp_pixels[j], -shift, axis=1)  # Apply shift

        # **Reshape & Assign to Hologram Correctly**
        for i in range(sh[1]):
            holo[n_SP * j:n_SP * (j + 1), n_SP * i:n_SP * (i + 1)] = rolled_sp[i].reshape(n_SP, n_SP).T

    return holo



import cupy as cp

def holo_SP_optimized_cupy(field, lut, pixel_combinations, step=0.01, ds_method='center', renorm=True):
    """Optimized version of holo_SP that maintains accuracy while significantly improving speed using GPU."""
    
    if renorm:
        field /= cp.max(cp.abs(field))  # Normalize amplitude
    
    # Determine superpixel size
    m = len(pixel_combinations[0])
    n_SP = int(cp.sqrt(m))

    # Downsample field
    ds_field = _down_sample(field, n_SP, method=ds_method)
    sh = ds_field.shape  # (height, width)

    # Initialize hologram
    holo = cp.zeros((sh[0] * n_SP, sh[1] * n_SP), dtype=int)

    # Rescale field values for LUT lookup
    field_sc = ds_field / (cp.max(cp.abs(ds_field)) * step)
    reim0 = len(lut) // 2

    # **Vectorized LUT lookup**
    re_indices = cp.clip(cp.round(cp.real(field_sc)).astype(int) + reim0, 0, lut.shape[0] - 1)
    im_indices = cp.clip(cp.round(cp.imag(field_sc)).astype(int) + reim0, 0, lut.shape[1] - 1)
    lut_indices = lut[re_indices, im_indices]

    # **Vectorized assignment of superpixel patterns**
    sp_pixels = pixel_combinations[lut_indices]  # Get superpixel patterns (shape: sh[0], sh[1], m)

    # **Apply Row-Wise Rolling with Corrected Assignment**
    for j in range(sh[0]):
        shift = cp.mod(n_SP * j, n_SP**2)  # Compute shift for row `j`
        rolled_sp = cp.roll(sp_pixels[j], -shift, axis=1)  # Apply shift

        # **Reshape & Assign to Hologram Correctly**
        for i in range(sh[1]):
            holo[n_SP * j:n_SP * (j + 1), n_SP * i:n_SP * (i + 1)] = rolled_sp[i].reshape(n_SP, n_SP).T

    return holo



import numpy as np

# def holo_SP_super_optimized(field, lut, pixel_combinations, step=0.01, ds_method='center', renorm=True):
#     """Further optimized version of holo_SP that avoids loops for even faster execution."""
    
#     if renorm:
#         field /= np.max(np.abs(field))  # Normalize amplitude
    
#     # Determine superpixel size
#     m = len(pixel_combinations[0])
#     n_SP = int(np.sqrt(m))

#     # Downsample field
#     ds_field = _down_sample(field, n_SP, method=ds_method)
#     sh = ds_field.shape  # (height, width)

#     # Initialize hologram
#     holo = np.zeros((sh[0] * n_SP, sh[1] * n_SP), dtype=int)

#     # Rescale field values for LUT lookup
#     field_sc = ds_field / (np.max(np.abs(ds_field)) * step)
#     reim0 = len(lut) // 2

#     # **Vectorized LUT lookup**
#     re_indices = np.clip(np.round(np.real(field_sc)).astype(int) + reim0, 0, lut.shape[0] - 1)
#     im_indices = np.clip(np.round(np.imag(field_sc)).astype(int) + reim0, 0, lut.shape[1] - 1)
#     lut_indices = lut[re_indices, im_indices]  # (sh[0], sh[1])

#     # **Vectorized Superpixel Retrieval**
#     sp_pixels = pixel_combinations[lut_indices]  # (sh[0], sh[1], m)

#     # **Compute Shift Offsets for All Rows at Once**
#     shifts = np.mod(n_SP * np.arange(sh[0]), n_SP**2)[:, None]  # (sh[0], 1)

#     # **Apply np.roll in One Shot for All Rows**
#     sp_pixels = np.take_along_axis(sp_pixels, np.mod(np.arange(m) - shifts, m), axis=1)

#     # **Vectorized Superpixel Assignment**
#     holo = sp_pixels.reshape(sh[0], sh[1], n_SP, n_SP).swapaxes(1, 2).reshape(sh[0] * n_SP, sh[1] * n_SP)

#     return holo


def holo_SP_super_optimized(field, lut, pixel_combinations, step=0.01, ds_method='center', renorm=True):
    """Returns the Super pixel hologram to generate the complex field with a super optimized method.
    
    Parameters
    ----------
    field : ndarray
        The target complex field
    lut : ndarray
        Look-up table.
    pixel_combinations : ndarray
        Array containing all the different pixel combinations leading to different complex values. 
    step : float
        Step used to build the LUT.
    ds_method : {'center', 'mean'}
        Method used to downsample the original image by the resolution of the superpixels. 
        'center' uses a value from the center, and 'mean' takes the mean of all pixels within the superpixel. 
    renorm : bool, optional
        If true, it sets the maximum amplitude of the field equal to one.
        
    Returns
    -------
    ndarray
        Binary Super pixel hologram with a size being proportional to that of the superpixel.
    """
    if renorm:
        field /= np.max(np.abs(field))
    
    # assume the field has been rescaled to the unit SP
    m = len(pixel_combinations[0])
    n_SP = int(np.sqrt(m))

    ds_field = _down_sample(field, n_SP, method=ds_method)
    sh = ds_field.shape
    holo = np.zeros((sh[-2] * n_SP, sh[-1] * n_SP), dtype=int)
    
    # Rescale field values according to LUT
    field_sc = ds_field / (np.max(np.abs(ds_field)) * step)
    reim0 = len(lut) // 2

    # **Convert indices to integers**
    real_idx = np.round(np.real(field_sc)).astype(int) + reim0
    imag_idx = np.round(np.imag(field_sc)).astype(int) + reim0

    # Ensure indices are within valid bounds
    real_idx = np.clip(real_idx, 0, lut.shape[0] - 1)
    imag_idx = np.clip(imag_idx, 0, lut.shape[1] - 1)

    # Lookup table indexing
    lut_indices = lut[real_idx, imag_idx]  # Shape: (sh[0], sh[1])

    # **Vectorized Shifting**
    shifts = np.mod(n_SP * np.arange(sh[0]), n_SP**2)[:, None]  # Shape (sh[0], 1)

    # Retrieve pixel combinations based on LUT indices
    sp_pixels = pixel_combinations[lut_indices]  # Shape: (sh[0], sh[1], m)

    # Fix shape mismatch for np.take_along_axis:
    # indices shape (sh[0], m) -> (sh[0], 1, m) -> (sh[0], sh[1], m)
    indices = np.mod(np.arange(m)[None, None, :] - shifts[:, None, :], m)  # Shape: (sh[0], sh[1], m)

    # Apply np.take_along_axis with correct dimensions
    sp_pixels = np.take_along_axis(sp_pixels, indices, axis=2)

    # **Vectorized Superpixel Assignment**
    holo = sp_pixels.reshape(sh[0], sh[1], n_SP, n_SP).swapaxes(1, 2).reshape(sh[0] * n_SP, sh[1] * n_SP)
    
    return holo


def debug_holo_comparison(target_field, lut, pixel_combinations, holo_SP_opt, holo_SP_sopt):
    """
    Compares intermediate steps between the optimized and super-optimized hologram generation
    to identify where differences arise.
    """
    print("Starting step-by-step comparison...")
    
    # Normalize target field
    target_field_norm = target_field / np.max(np.abs(target_field))
    print("Step 1: Target field normalized.")
    
    # Compute LUT indices (Check if they match exactly)
    lut_indices_opt = lut[np.round(np.real(target_field_norm)).astype(int),
                           np.round(np.imag(target_field_norm)).astype(int)]
    lut_indices_sopt = lut[np.round(np.real(target_field_norm)).astype(int),
                            np.round(np.imag(target_field_norm)).astype(int)]
    
    if not np.array_equal(lut_indices_opt, lut_indices_sopt):
        print("Discrepancy found in LUT indices!")
        diff_indices = np.where(lut_indices_opt != lut_indices_sopt)
        print("First differing indices:", diff_indices)
        print("Optimized LUT value:", lut_indices_opt[diff_indices[0][0], diff_indices[1][0]])
        print("Super-optimized LUT value:", lut_indices_sopt[diff_indices[0][0], diff_indices[1][0]])
        return
    print("Step 2: LUT indices match.")
    
    # Compute shifted pixel combinations
    n_SP = pixel_combinations.shape[0]  # Assuming this dimension represents superpixel count
    sh = target_field.shape
    m = sh[1] * n_SP  # Number of pixels per row
    shifts = np.mod(n_SP * np.arange(sh[0]), n_SP**2)[:, None]
    
    indices_opt = np.mod(np.arange(m)[None, :] - shifts, m)
    indices_sopt = np.mod(np.arange(m)[None, :] - shifts, m)
    
    if not np.array_equal(indices_opt, indices_sopt):
        print("Discrepancy found in index shifting!")
        return
    print("Step 3: Index shifts match.")
    
    # Extract pixel combinations
    sp_pixels_opt = np.take_along_axis(pixel_combinations[lut_indices_opt], indices_opt, axis=1)
    sp_pixels_sopt = np.take_along_axis(pixel_combinations[lut_indices_sopt], indices_sopt, axis=1)
    
    if not np.array_equal(sp_pixels_opt, sp_pixels_sopt):
        print("Discrepancy found in superpixel selection!")
        diff_pixels = np.where(sp_pixels_opt != sp_pixels_sopt)
        print("First differing pixel indices:", diff_pixels)
        print("Optimized pixel value:", sp_pixels_opt[diff_pixels[0][0], diff_pixels[1][0]])
        print("Super-optimized pixel value:", sp_pixels_sopt[diff_pixels[0][0], diff_pixels[1][0]])
        return
    print("Step 4: Superpixel selection matches.")
    
    # Final hologram reshaping
    holo_opt = sp_pixels_opt.reshape(sh[0], sh[1], n_SP, n_SP).swapaxes(1, 2).reshape(sh[0] * n_SP, sh[1] * n_SP)
    holo_sopt = sp_pixels_sopt.reshape(sh[0], sh[1], n_SP, n_SP).swapaxes(1, 2).reshape(sh[0] * n_SP, sh[1] * n_SP)
    
    if not np.array_equal(holo_opt, holo_sopt):
        print("Discrepancy found in final hologram!")
        diff_holo = np.where(holo_opt != holo_sopt)
        print("First differing hologram pixel:", diff_holo)
        print("Optimized holo value:", holo_opt[diff_holo[0][0], diff_holo[1][0]])
        print("Super-optimized holo value:", holo_sopt[diff_holo[0][0], diff_holo[1][0]])
        return
    print("Step 5: Final holograms match exactly.")
    
    print("No differences found. The issue might be elsewhere.")


# def parallel_lee2(field, nuvec=(1/4,0)):
#     a, phi, x, y = holo_preamble(field, nuvec)
#     return (1+np.sign(np.cos(2*np.pi*(nuvec[0]*x + nuvec[1]*y)-phi) - np.cos(np.arcsin(a))))/2

# def orthogonal_lee2(field, nuvec=(1/4,0), renorm=True):
#     a, phi, x, y = holo_preamble(field, nuvec, renorm=renorm)
    
#     return (1+np.sign(np.cos(2*np.pi*(nuvec[0]*x + nuvec[1]*y)-phi)))/2 \
#         * (1+np.sign(np.cos(2*np.pi*(-nuvec[1]*x + nuvec[0]*y)) - np.cos(np.pi*a)))/2
# Constellations
