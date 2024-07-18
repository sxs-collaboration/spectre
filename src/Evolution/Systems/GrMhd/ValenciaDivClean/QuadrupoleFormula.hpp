// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/TagsDeclarations.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/QuadrupoleFormula.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace grmhd::ValenciaDivClean::Tags {
/// \brief Compute tag for the quadrupole moment.
///
/// Compute item for the quadrupole moment, using
/// \f$\tilde{D} x^i x^j\f$ (equation 21 of \cite Shibata2003),
/// with \f$\tilde{D}=\sqrt{\gamma}\rho W\f$, $W$ being the Lorentz factor,
/// $\gamma$ being the determinant of the spatial metric, and
/// \f$x\f$ the coordinates in Frame Fr.
template <typename DataType, size_t Dim, typename OutputCoordsTag,
          typename Fr = Frame::Inertial>
struct QuadrupoleMomentCompute
    : hydro::Tags::QuadrupoleMoment<DataType, Dim, Fr>,
      db::ComputeTag {
  using argument_tags = tmpl::list<TildeD, OutputCoordsTag>;

  using base = hydro::Tags::QuadrupoleMoment<DataType, Dim, Fr>;
  using return_type = typename base::type;

  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<tnsr::ii<DataType, Dim, Fr>*> result,
      const Scalar<DataType>& tilde_d,
      const tnsr::I<DataType, Dim, Fr>& coordinates)>(
      &hydro::quadrupole_moment<DataType, Dim, Fr>);
};

/// \brief Compute tag for the first time derivative of the quadrupole moment.
///
/// Compute item for the first time derivative of the quadrupole moment, using
/// \f$\tilde{D} (v^i x^j + x^i v^j)\f$ (equation 23 of \cite Shibata2003),
/// with \f$\tilde{D}=\sqrt{\gamma}\rho W\f$, $W$ being the Lorentz factor,
/// $\gamma$ being the determinant of the spatial metric, \f$x\f$ the
/// coordinates in Frame Fr, and \f$v\f$ the corresponding spatial velocity.
template <typename DataType, size_t Dim, typename OutputCoordsTag,
          typename VelocityTag, typename Fr = Frame::Inertial>
struct QuadrupoleMomentDerivativeCompute
    : hydro::Tags::QuadrupoleMomentDerivative<DataType, Dim, Fr>,
      db::ComputeTag {
  using argument_tags = tmpl::list<TildeD, OutputCoordsTag, VelocityTag>;

  using base = hydro::Tags::QuadrupoleMomentDerivative<DataType, Dim, Fr>;
  using return_type = typename base::type;

  static std::string name() {
    return "QuadrupoleMomentDerivative(" + db::tag_name<VelocityTag>() + ")";
  }

  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<tnsr::ii<DataType, Dim, Fr>*> result,
      const Scalar<DataType>& tilde_d,
      const tnsr::I<DataType, Dim, Fr>& coordinates,
      const tnsr::I<DataType, Dim, Fr>& spatial_velocity)>(
      &hydro::quadrupole_moment_derivative<DataType, Dim, Fr>);
};
}  // namespace grmhd::ValenciaDivClean::Tags
