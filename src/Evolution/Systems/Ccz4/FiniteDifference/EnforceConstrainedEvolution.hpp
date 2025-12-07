// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
/*!
 * \brief Mutator to enforce constrained evolution after every time step
 *
 * Togglable option in input file via option tag defined in Ccz4::fd::Tags.
 * If constrained_evolution is true, enforce
 * the unit determinant constraint on the conformal spatial metric
 * and traceless constraint on the ATilde
 */
struct EnforceConstrainedEvolution : tt::ConformsTo<db::protocols::Mutator> {
  static constexpr size_t dim = System::volume_dim;
  using return_tags = tmpl::list<::Ccz4::Tags::ConformalMetric<DataVector, dim>,
                                 ::Ccz4::Tags::ATilde<DataVector, dim>>;
  using argument_tags = tmpl::list<::Ccz4::fd::Tags::ConstrainedEvolution>;

  static void apply(
      gsl::not_null<tnsr::ii<DataVector, dim>*> conformal_spatial_metric,
      gsl::not_null<tnsr::ii<DataVector, dim>*> a_tilde,
      bool constrained_evolution);
};
}  // namespace Ccz4::fd
