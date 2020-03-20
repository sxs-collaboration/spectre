// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/Initialize/InverseCubic.hpp"

#include <cstddef>
#include <memory>
#include <string>

#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/Initialize/InitializeJ.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class ComplexDataVector;
/// \endcond

namespace Cce {
namespace InitializeJ {

/*!
 * \brief Initialize \f$J\f$ on the first hypersurface from provided boundary
 * values of \f$J\f$, \f$R\f$, and \f$\partial_r J\f$.
 *
 * \details This initial data is chosen to take the function:
 *
 * \f[ J = \frac{A}{r} + \frac{B}{r^3},\f]
 *
 * where
 *
 * \f{align*}{
 * A &= R \left( \frac{3}{2} J|_{r = R} + \frac{1}{2} R \partial_r J|_{r =
 * R}\right) \notag\\
 * B &= - \frac{1}{2} R^3 (J|_{r = R} + R \partial_r J|_{r = R})
 * \f}
 */
struct InverseCubic : InitializeJ {
  using options = tmpl::list<>;
  static constexpr OptionString help = {
      "Initialization process where J is set to a simple Ansatz with a\n"
      " A/r + B/r^3 piece such that it is smooth with the Cauchy data at the \n"
      "worldtube"};

  WRAPPED_PUPable_decl_template(InverseCubic);  // NOLINT
  explicit InverseCubic(CkMigrateMessage* /*unused*/) noexcept {}

  InverseCubic() = default;

  std::unique_ptr<InitializeJ> get_clone() const noexcept override;

  void operator()(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> j,
      gsl::not_null<tnsr::i<DataVector, 3>*> cartesian_cauchy_coordinates,
      gsl::not_null<
          tnsr::i<DataVector, 2, ::Frame::Spherical<::Frame::Inertial>>*>
          angular_cauchy_coordinates,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_j,
      const Scalar<SpinWeighted<ComplexDataVector, 2>>& boundary_dr_j,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, size_t l_max,
      size_t number_of_radial_points) const noexcept override;

  void pup(PUP::er& /*p*/) noexcept override;
};
}  // namespace InitializeJ
}  // namespace Cce
