// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp" // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp" // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare gr::Tags::Lapse
// IWYU pragma: no_forward_declare gr::Tags::Shift
// IWYU pragma: no_forward_declare gr::Tags::SqrtDetSpatialMetric
// IWYU pragma: no_forward_declare hydro::Tags::LorentzFactor
// IWYU pragma: no_forward_declare hydro::Tags::RestMassDensity
// IWYU pragma: no_forward_declare hydro::Tags::SpatialVelocity

namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl

namespace hydro {
//@{
/// Computes the vector \f$J^i\f$ in \f$\dot{M} = -\int J^i s_i d^2S\f$,
/// representing the mass flux through a surface with normal \f$s_i\f$.
///
/// Note that the integral is understood
/// as a flat-space integral: all metric factors are included in \f$J^i\f$.
/// In particular, if the integral is done over a Strahlkorper, the
/// `StrahlkorperGr::euclidean_area_element` of the Strahlkorper should be used,
/// and \f$s_i\f$ is
/// the normal one-form to the Strahlkorper normalized with the flat metric,
/// \f$s_is_j\delta^{ij}=1\f$.
///
/// The formula is
/// \f$ J^i = \rho W \sqrt{\gamma}(\alpha v^i-\beta^i)\f$,
/// where \f$\rho\f$ is the mass density, \f$W\f$ is the Lorentz factor,
/// \f$v^i\f$ is the spatial velocity of the fluid,
/// \f$\gamma\f$ is the determinant of the 3-metric \f$\gamma_{ij}\f$,
/// \f$\alpha\f$ is the lapse, and \f$\beta^i\f$ is the shift.
template <typename DataType, size_t Dim, typename Frame>
void mass_flux(gsl::not_null<tnsr::I<DataType, Dim, Frame>*> result,
               const Scalar<DataType>& rest_mass_density,
               const tnsr::I<DataType, Dim, Frame>& spatial_velocity,
               const Scalar<DataType>& lorentz_factor,
               const Scalar<DataType>& lapse,
               const tnsr::I<DataType, Dim, Frame>& shift,
               const Scalar<DataType>& sqrt_det_spatial_metric) noexcept;

template <typename DataType, size_t Dim, typename Frame>
tnsr::I<DataType, Dim, Frame> mass_flux(
    const Scalar<DataType>& rest_mass_density,
    const tnsr::I<DataType, Dim, Frame>& spatial_velocity,
    const Scalar<DataType>& lorentz_factor, const Scalar<DataType>& lapse,
    const tnsr::I<DataType, Dim, Frame>& shift,
    const Scalar<DataType>& sqrt_det_spatial_metric) noexcept;
//@}

namespace Tags {
/// Compute item for mass flux vector \f$J^i\f$.
///
/// Can be retrieved using `hydro::Tags::MassFlux`
template <typename DataType, size_t Dim, typename Frame>
struct MassFluxCompute : MassFlux<DataType, Dim, Frame>,
                               db::ComputeTag {
  using argument_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataType>,
                 hydro::Tags::SpatialVelocity<DataType, Dim, Frame>,
                 hydro::Tags::LorentzFactor<DataType>,
                 ::gr::Tags::Lapse<DataType>,
                 ::gr::Tags::Shift<Dim, Frame, DataType>,
                 ::gr::Tags::SqrtDetSpatialMetric<DataType>>;

  using return_type = tnsr::I<DataType, Dim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::I<DataType, Dim, Frame>*>, const Scalar<DataType>&,
      const tnsr::I<DataType, Dim, Frame>&, const Scalar<DataType>&,
      const Scalar<DataType>&, const tnsr::I<DataType, Dim, Frame>&,
      const Scalar<DataType>&) noexcept>(&mass_flux<DataType, Dim, Frame>);

  using base = MassFlux<DataType, Dim, Frame>;
};
}  // namespace Tags
}  // namespace hydro
