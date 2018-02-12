// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/StrahlkorperGr.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"

namespace StrahlkorperGr {

template <typename Frame>
tnsr::i<DataVector, 3, Frame> unit_normal_one_form(
    const tnsr::i<DataVector, 3, Frame>& normal_one_form,
    const DataVector& one_over_one_form_magnitude) noexcept {
  auto unit_normal_one_form = normal_one_form;
  for (size_t i = 0; i < 3; ++i) {
    unit_normal_one_form.get(i) *= one_over_one_form_magnitude;
  }
  return unit_normal_one_form;
}

template <typename Frame>
tnsr::ii<DataVector, 3, Frame> grad_unit_normal_one_form(
    const tnsr::i<DataVector, 3, Frame>& r_hat, const DataVector& radius,
    const tnsr::i<DataVector, 3, Frame>& unit_normal_one_form,
    const tnsr::ii<DataVector, 3, Frame>& d2x_radius,
    const DataVector& one_over_one_form_magnitude,
    const tnsr::Ijj<DataVector, 3, Frame>& christoffel_2nd_kind) noexcept {
  const DataVector one_over_radius = 1.0 / radius;
  tnsr::ii<DataVector, 3, Frame> grad_normal(radius.size(), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {  // symmetry
      grad_normal.get(i, j) = -one_over_one_form_magnitude *
                              (r_hat.get(i) * r_hat.get(j) * one_over_radius +
                               d2x_radius.get(i, j));
      for (size_t k = 0; k < 3; ++k) {
        grad_normal.get(i, j) -=
            unit_normal_one_form.get(k) * christoffel_2nd_kind.get(k, i, j);
      }
    }
    grad_normal.get(i, i) += one_over_radius * one_over_one_form_magnitude;
  }
  return grad_normal;
}

template <typename Frame>
Scalar<DataVector> expansion(
    const tnsr::ii<DataVector, 3, Frame>& grad_normal,
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
    const tnsr::II<DataVector, 3, Frame>& upper_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature) noexcept {
  // Form inverse surface metric
  tnsr::II<DataVector, 3, Frame> inv_surf_metric(
      get<0>(unit_normal_vector).size(), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      inv_surf_metric.get(i, j) =
          upper_spatial_metric.get(i, j) -
          unit_normal_vector.get(i) * unit_normal_vector.get(j);
    }
  }

  // If you want the future *ingoing* null expansion,
  // the formula is the same as here except you
  // change the sign on grad_normal just before you
  // subtract the extrinsic curvature.
  // That is, if GsBar is the value of grad_normal
  // at this point in the code, and S^i is the unit
  // spatial normal to the surface,
  // the outgoing expansion is
  // (g^ij - S^i S^j) (GsBar_ij - K_ij)
  // and the ingoing expansion is
  // (g^ij - S^i S^j) (-GsBar_ij - K_ij)

  Scalar<DataVector> expansion(get<0>(unit_normal_vector).size(), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      get(expansion) += inv_surf_metric.get(i, j) *
                        (grad_normal.get(i, j) - extrinsic_curvature.get(i, j));
    }
  }

  return expansion;
}

template <typename Frame>
tnsr::ii<DataVector, 3, Frame> extrinsic_curvature(
    const tnsr::ii<DataVector, 3, Frame>& grad_normal,
    const tnsr::i<DataVector, 3, Frame>& unit_normal_one_form,
    const tnsr::II<DataVector, 3, Frame>& upper_spatial_metric) noexcept {

  // Form projector
  tnsr::Ij<DataVector, 3, Frame> projector(
      get<0>(unit_normal_one_form).size(), 0.0);
  const auto& unit_normal_vector = raise_or_lower_index(unit_normal_one_form,
                                                        upper_spatial_metric);
  for (size_t i = 0; i < 3; ++i) {
    projector.get(i,i) += 1.0;
    for (size_t j = 0; j < 3; ++j) {
      projector.get(i, j) -=
          unit_normal_vector.get(i) * unit_normal_one_form.get(j);
    }
  }

  // Project symmetrized gradient
  tnsr::ii<DataVector, 3, Frame> extrinsic_curvature(
      get<0>(unit_normal_one_form).size(), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j ) {
      for (size_t k = 0; k < 3; ++k ) {
        for (size_t l =0; l < 3; ++l ) {
          extrinsic_curvature.get(i,j) +=
              //Note: grad_normal is already symmetric, so
              //no need to symmetrize it
              projector.get(k,i) * projector.get(l,j) * grad_normal.get(k,l);
        }
      }
    }
  }

  return extrinsic_curvature;
}

template <typename Frame>
Scalar<DataVector> ricci_scalar(
    const tnsr::ii<DataVector, 3, Frame>& spatial_ricci_tensor,
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
    const tnsr::II<DataVector, 3, Frame>& upper_spatial_metric) noexcept {

  Scalar<DataVector> ricci_scalar(get<0>(unit_normal_vector).size(), 0.0);

  for(size_t i = 0; i < 3; ++i) {
    for(size_t j = 0; j < 3; ++j) {
      ricci_scalar.get() -= 2.0 * spatial_ricci_tensor.get(i,j)
                            * unit_normal_vector.get(i)
                            * unit_normal_vector.get(j);

      for(size_t k = 0; k < 3; ++k) {
        for(size_t l = 0; l < 3; ++l) {
          // K^{ij} K_{ij} = g^{ik} g^{jl} K_{kl} K_{ij}
          get(ricci_scalar) -= upper_spatial_metric.get(i,k)
                               * upper_spatial_metric.get(j,l)
                               * extrinsic_curvature.get(k,l)
                               * extrinsic_curvature.get(i,j);
        }
      }
    }
  }

  get(ricci_scalar) += square(get(trace(extrinsic_curvature,
                                        upper_spatial_metric)));
  get(ricci_scalar) += get(trace(spatial_ricci_tensor, upper_spatial_metric));
  return ricci_scalar;
}
}  // namespace StrahlkorperGr

template tnsr::i<DataVector, 3, Frame::Inertial>
StrahlkorperGr::unit_normal_one_form<Frame::Inertial>(
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_one_form,
    const DataVector& one_over_one_form_magnitude) noexcept;

template tnsr::ii<DataVector, 3, Frame::Inertial>
StrahlkorperGr::grad_unit_normal_one_form<Frame::Inertial>(
    const tnsr::i<DataVector, 3, Frame::Inertial>& r_hat,
    const DataVector& radius,
    const tnsr::i<DataVector, 3, Frame::Inertial>& unit_normal_one_form,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& d2x_radius,
    const DataVector& one_over_one_form_magnitude,
    const tnsr::Ijj<DataVector, 3, Frame::Inertial>&
        christoffel_2nd_kind) noexcept;

template Scalar<DataVector> StrahlkorperGr::expansion<Frame::Inertial>(
    const tnsr::ii<DataVector, 3, Frame::Inertial>& grad_normal,
    const tnsr::I<DataVector, 3, Frame::Inertial>& unit_normal_vector,
    const tnsr::II<DataVector, 3, Frame::Inertial>& upper_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>&
        extrinsic_curvature) noexcept;

template tnsr::ii<DataVector, 3, Frame::Inertial>
StrahlkorperGr::extrinsic_curvature<Frame::Inertial>(
    const tnsr::ii<DataVector, 3, Frame::Inertial>& grad_normal,
    const tnsr::i<DataVector, 3, Frame::Inertial>& unit_normal_one_form,
    const tnsr::II<DataVector, 3, Frame::Inertial>&
    upper_spatial_metric) noexcept;


template Scalar<DataVector> StrahlkorperGr::ricci_scalar<Frame::Inertial>(
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_ricci_tensor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& unit_normal_vector,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& extrinsic_curvature,
    const tnsr::II<DataVector, 3, Frame::Inertial>&
    upper_spatial_metric) noexcept;
