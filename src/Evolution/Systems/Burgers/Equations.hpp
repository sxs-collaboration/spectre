// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/Options.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
// IWYU pragma: no_forward_declare Tensor
template <typename>
class Variables;
namespace Burgers {
namespace Tags {
struct U;
}  // namespace Tags
}  // namespace Burgers
namespace PUP {
class er;
}  // namespace PUP
namespace Tags {
template <typename>
struct NormalDotFlux;
}  // namespace Tags
/// \endcond

namespace Burgers {
/// \ingroup NumericalFluxesGroup
struct LocalLaxFriedrichsFlux {
  using options = tmpl::list<>;
  static constexpr OptionString help{
      "Computes the Local LF flux for the Burgers system."};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) noexcept {}

  using package_tags = tmpl::list<::Tags::NormalDotFlux<Tags::U>, Tags::U>;

  using argument_tags = tmpl::list<::Tags::NormalDotFlux<Tags::U>, Tags::U>;

  void package_data(
      gsl::not_null<Variables<package_tags>*> packaged_data,
      const Scalar<DataVector>& normal_dot_flux_u,
      const Scalar<DataVector>& u) const noexcept;

  void operator()(
      gsl::not_null<Scalar<DataVector>*> normal_dot_numerical_flux_u,
      const Scalar<DataVector>& normal_dot_flux_u_interior,
      const Scalar<DataVector>& u_interior,
      const Scalar<DataVector>& minus_normal_dot_flux_u_exterior,
      const Scalar<DataVector>& u_exterior) const noexcept;
};

struct ComputeLargestCharacteristicSpeed {
  using argument_tags = tmpl::list<Tags::U>;
  static double apply(const Scalar<DataVector>& u) noexcept;
};
}  // namespace Burgers
