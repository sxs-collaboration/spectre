// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
// IWYU pragma: no_forward_declare Tensor
namespace Burgers {
namespace Tags {
struct U;
}  // namespace Tags
}  // namespace Burgers
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace Burgers {
/// The flux of \f$U\f$ is \f$\frac{1}{2} U^2\f$.
struct Fluxes {
  using argument_tags = tmpl::list<Tags::U>;
  using return_tags =
      tmpl::list<::Tags::Flux<Tags::U, tmpl::size_t<1>, Frame::Inertial>>;
  static void apply(gsl::not_null<tnsr::I<DataVector, 1>*> flux,
                    const Scalar<DataVector>& u) noexcept;
};
}  // namespace Burgers
