# Distributed under the MIT License.
# See LICENSE.txt for details.


def mass_flux(
    rest_mass_density,
    spatial_velocity,
    lorentz_factor,
    lapse,
    shift,
    sqrt_det_spatial_metric,
):
    return (
        rest_mass_density
        * lorentz_factor
        * sqrt_det_spatial_metric
        * (lapse * spatial_velocity - shift)
    )
