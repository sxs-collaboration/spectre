// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace hydro {


enum class HalfPlaneIntegralMask {None, PositiveXOnly, NegativeXOnly};

std::string name(const HalfPlaneIntegralMask mask);

/// Tag containing TildeD * SpecificInternalEnergy
/// Useful as a diagnostics tool, as input to volume
/// integral.
namespace Tags {
template <typename DataType>
struct MassWeightedInternalEnergy : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// Tag containing TildeD * (LorentzFactor - 1.0)
/// Useful as a diagnostics tool, as input to volume
/// integral.
template <typename DataType>
struct MassWeightedKineticEnergy : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// Contains TildeD restricted to regions marked as
/// unbound, using the u_t < -1 criterion.
template <typename DataType>
struct TildeDUnboundUtCriterion : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// Contains TildeD restricted to x>0 or x<0. This provides the
/// normalization factor for integrals over the half plane
/// weighted by tildeD
template <typename DataType, HalfPlaneIntegralMask IntegralMask>
struct TildeDInHalfPlane : db::SimpleTag {
  using type = Scalar<DataType>;
  static std::string name() {
    return "TildeDMask(" + ::hydro::name(IntegralMask) + ")";
  }
};

/// Contains TildeD * (coordinates in frame Fr).
/// IntegralMask allows us to restrict the data to the x>0 or x<0
/// plane in grid coordinates (useful for NSNS).
template <typename DataType, size_t Dim, HalfPlaneIntegralMask IntegralMask,
          typename Fr = Frame::Inertial>
struct MassWeightedCoords : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Fr>;
  static std::string name() {
    return "MassWeightedCoordsMask(" + ::hydro::name(IntegralMask) + ")";
  }
};
}  // namespace Tags


template <typename DataType, size_t Dim, typename Frame>
void u_lower_t(const gsl::not_null<Scalar<DataType>*> result,
               const Scalar<DataType>& lorentz_factor,
               const tnsr::I<DataType, Dim, Frame>& spatial_velocity,
               const tnsr::ii<DataType, Dim, Frame>& spatial_metric,
               const Scalar<DataType>& lapse,
               const tnsr::I<DataType, Dim, Frame>& shift);

/// Compute tilde_d * specific_internal_energy
/// Result of the calculation stored in result.
template <typename DataType>
void mass_weighted_internal_energy(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& tilde_d,
    const Scalar<DataType>& specific_internal_energy);

/// Compute tilde_d * (lorentz_factor - 1.0)
/// Result of the calculation stored in result.
template <typename DataType>
void mass_weighted_kinetic_energy(const gsl::not_null<Scalar<DataType>*> result,
                                  const Scalar<DataType>& tilde_d,
                                  const Scalar<DataType>& lorentz_factor);

/// Returns tilde_d in regions where u_t < -1 and 0 in regions where
/// u_t > -1 (approximate criteria for unbound matter, theoretically
/// valid for particles following geodesics of a time-independent metric).
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
void tilde_d_unbound_ut_criterion(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& tilde_d, const Scalar<DataType>& lorentz_factor,
    const tnsr::I<DataType, Dim, Fr>& spatial_velocity,
    const tnsr::ii<DataType, Dim, Fr>& spatial_metric,
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Fr>& shift);

/// Returns tilde_d in one half plane and zero in the other
/// IntegralMask allows us to restrict the data to the x>0 or x<0
/// plane in grid coordinates (useful for NSNS).
template <HalfPlaneIntegralMask IntegralMask, typename DataType, size_t Dim>
void tilde_d_in_half_plane(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& tilde_d,
    const tnsr::I<DataType, Dim, Frame::Grid>& grid_coords);

/// Returns tilde_d * compute_coords
/// IntegralMask allows us to restrict the data to the x>0 or x<0
/// plane in grid coordinates (useful for NSNS).
template <HalfPlaneIntegralMask IntegralMask, typename DataType, size_t Dim,
          typename Fr = Frame::Inertial>
void mass_weighted_coords(
    const gsl::not_null<tnsr::I<DataType, Dim, Fr>*> result,
    const Scalar<DataType>& tilde_d,
    const tnsr::I<DataType, Dim, Frame::Grid>& grid_coords,
    const tnsr::I<DataType, Dim, Fr>& compute_coords);

namespace Tags {
/// Compute item for mass-weighted internal energy
///
/// Can be retrieved using `hydro::Tags::MassWeightedInternalEnergy'
template <typename DataType>
struct MassWeightedInternalEnergyCompute : MassWeightedInternalEnergy<DataType>,
                                           db::ComputeTag {
  using argument_tags = tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
                                   SpecificInternalEnergy<DataType>>;

  using return_type = Scalar<DataType>;

  using base = MassWeightedInternalEnergy<DataType>;

  static constexpr auto function =
      static_cast<void (*)(const gsl::not_null<Scalar<DataType>*> result,
                           const Scalar<DataType>& tilde_d,
                           const Scalar<DataType>& specific_internal_energy)>(
          &mass_weighted_internal_energy<DataType>);
};

/// Compute item for mass-weighted internal energy
///
/// Can be retrieved using `hydro::Tags::MassWeightedKineticEnergy'
template <typename DataType>
struct MassWeightedKineticEnergyCompute : MassWeightedKineticEnergy<DataType>,
                                          db::ComputeTag {
  using argument_tags = tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
                                   LorentzFactor<DataType>>;

  using return_type = Scalar<DataType>;

  using base = MassWeightedKineticEnergy<DataType>;

  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<Scalar<DataType>*> result,
      const Scalar<DataType>& tilde_d, const Scalar<DataType>& lorentz_factor)>(
      &mass_weighted_kinetic_energy<DataType>);
};

/// Compute item for TildeD limited to unbound material (u_t<-1 criteria)
///
/// Can be retrieved using `hydro::Tags::TildeDUnboundUtCriterion'
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct TildeDUnboundUtCriterionCompute : TildeDUnboundUtCriterion<DataType>,
                                         db::ComputeTag {
  using argument_tags =
      tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD, LorentzFactor<DataType>,
                 SpatialVelocity<DataType, Dim, Fr>,
                 gr::Tags::SpatialMetric<DataType, Dim, Fr>,
                 gr::Tags::Lapse<DataType>, gr::Tags::Shift<DataType, Dim, Fr>>;

  using return_type = Scalar<DataType>;

  using base = TildeDUnboundUtCriterion<DataType>;

  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<Scalar<DataType>*> result,
      const Scalar<DataType>& tilde_d, const Scalar<DataType>& lorentz_factor,
      const tnsr::I<DataType, Dim, Fr>& spatial_velocity,
      const tnsr::ii<DataType, Dim, Fr>& spacial_metric,
      const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Fr>& shift)>(
      &tilde_d_unbound_ut_criterion<DataType, Dim, Fr>);
};

/// Compute tag for TildeD limited to the x>0 or x<0 half plane
template <typename DataType, size_t Dim, HalfPlaneIntegralMask IntegralMask,
          typename GridCoordsTag>
struct TildeDInHalfPlaneCompute : TildeDInHalfPlane<DataType, IntegralMask>,
                                  db::ComputeTag {
  using argument_tags =
      tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD, GridCoordsTag>;

  using return_type = Scalar<DataType>;

  using base = TildeDInHalfPlane<DataType, IntegralMask>;

  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<Scalar<DataType>*> result,
      const Scalar<DataType>& tilde_d,
      const tnsr::I<DataType, Dim, Frame::Grid>& grid_coords)>(
        &tilde_d_in_half_plane<IntegralMask, DataType, Dim>);
};

/// Compute item for TildeD * (coordinates in frame Fr).
/// IntegralMask allows us to restrict the data to the x>0 or x<0
/// plane in grid coordinates (useful for NSNS).
///
/// Can be retrieved using `hydro::Tags::MassWeightedCoords'
template <typename DataType, size_t Dim, HalfPlaneIntegralMask IntegralMask,
          typename GridCoordsTag, typename OutputCoordsTag,
          typename Fr = Frame::Inertial>
struct MassWeightedCoordsCompute : MassWeightedCoords<DataType, Dim,
                                   IntegralMask>, db::ComputeTag {
  using argument_tags = tmpl::list<grmhd::ValenciaDivClean::Tags::TildeD,
                                   GridCoordsTag, OutputCoordsTag>;

  using return_type = tnsr::I<DataType, Dim, Fr>;

  using base = MassWeightedCoords<DataType, Dim, IntegralMask, Fr>;

  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<tnsr::I<DataType, Dim, Fr>*> result,
      const Scalar<DataType>& tilde_d,
      const tnsr::I<DataType, Dim, Frame::Grid>& grid_coords,
      const tnsr::I<DataType, Dim, Fr>& compute_coords)>(
      &mass_weighted_coords<IntegralMask, DataType, Dim, Fr>);
};
}  // namespace Tags
}  // namespace hydro
