# Distributed under the MIT License.
# See LICENSE.txt for details.

import Fluxes, Sources


def non_flux_terms_dt_tilde_e(tilde_e, tilde_s, tilde_p, lapse, shift,
                              spatial_metric, inv_spatial_metric, source_n,
                              source_i, d_lapse, d_shift, d_spatial_metric,
                              extrinsic_curvature):
    return Sources.source_tilde_e(tilde_e, tilde_s, tilde_p, source_n,
                                  source_i, lapse, d_lapse, d_shift,
                                  d_spatial_metric, inv_spatial_metric,
                                  extrinsic_curvature)


def non_flux_terms_dt_tilde_s(tilde_e, tilde_s, tilde_p, lapse, shift,
                              spatial_metric, inv_spatial_metric, source_n,
                              source_i, d_lapse, d_shift, d_spatial_metric,
                              extrinsic_curvature):
    return Sources.source_tilde_s(tilde_e, tilde_s, tilde_p, source_n,
                                  source_i, lapse, d_lapse, d_shift,
                                  d_spatial_metric, inv_spatial_metric,
                                  extrinsic_curvature)


def tilde_e_flux(tilde_e, tilde_s, tilde_p, lapse, shift, spatial_metric,
                 inv_spatial_metric, source_n, source_i, d_lapse, d_shift,
                 d_spatial_metric, extrinsic_curvature):
    return Fluxes.tilde_e_flux(tilde_e, tilde_s, tilde_p, lapse, shift,
                               spatial_metric, inv_spatial_metric)


def tilde_s_flux(tilde_e, tilde_s, tilde_p, lapse, shift, spatial_metric,
                 inv_spatial_metric, source_n, source_i, d_lapse, d_shift,
                 d_spatial_metric, extrinsic_curvature):
    return Fluxes.tilde_s_flux(tilde_e, tilde_s, tilde_p, lapse, shift,
                               spatial_metric, inv_spatial_metric)
