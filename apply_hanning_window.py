import numpy as np
import xarray as xr

def apply_hann_window_lon(da, center_lon, window_size):
    """
    Applies a Hann window along the 'lon' dimension of an xarray DataArray.

    Parameters:
    - da (xarray DataArray): Input data in the form da(time, lat, lon).
    - center_lon (float): Value of longitude where the Hann window is centered.
    - window_size (int): Length of the Hann window in longitude.

    Returns:
    - windowed_da (xarray DataArray): DataArray with Hann window applied along 'lon' dimension.
    """

    # Calculate left and right extensions of the window
    left_ext = center_lon - int(window_size / 2)
    right_ext = center_lon + int(window_size / 2)

    # Calculate the resolution of longitude
    lon_resolution = abs(da.lon[0] - da.lon[1])

    # Generate a Hann window of specified size
    hann_window = np.hanning(window_size)
    hann_filter = np.zeros(da.lon.shape[0])

    # Find indices of the longitude values within the window
    lon_indices = np.searchsorted(da.lon, np.arange(left_ext, right_ext + lon_resolution, lon_resolution))

    # Create the Hann filter with non-zero values within the window
    hann_filter[lon_indices[0]:lon_indices[-1] + 1] = hann_window

    # Extend the filter to match the shape of the input data
    hann_filter_full = np.tile(hann_filter, (da.shape[0], da.shape[1], 1))

    # Apply the windowed filter to the input data
    windowed_da = da.copy() * hann_filter_full

    # Create a new DataArray with the windowed data
    windowed_da = xr.DataArray(data=windowed_da, name=da.name, dims=da.dims, coords=da.coords)

    # Update attributes to describe the applied Hanning window
    windowed_da.attrs.update({
        'description': f'Hanning window of size {window_size} applied at {center_lon}E' if center_lon >= 0
        else f'Hanning window of size {window_size} applied at {abs(center_lon)}W'
    })

    return windowed_da
