// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarAdvection {
/*!
 * \brief Computes the time derivative terms needed for the ScalarAdvection
 * system, which are just the fluxes.
 */
template <size_t Dim>
struct TimeDerivativeTerms {
  using temporary_tags = tmpl::list<Tags::VelocityField<Dim>>;
  using argument_tags = tmpl::list<Tags::U, Tags::VelocityField<Dim>>;

  static void apply(
      // Time derivatives returned by reference. No source terms or
      // nonconservative products, so not used. All the tags in the
      // variables_tag in the system struct.
      gsl::not_null<Scalar<DataVector>*> /*non_flux_terms_dt_vars*/,

      // Fluxes returned by reference. Listed in the system struct as
      // flux_variables.
      gsl::not_null<tnsr::I<DataVector, Dim>*> flux,

      // Temporary tags
      gsl::not_null<tnsr::I<DataVector, Dim>*> temp_velocity_field,

      // Arguments listed in argument_tags above
      const Scalar<DataVector>& u,
      const tnsr::I<DataVector, Dim>& velocity_field) noexcept;
};
}  // namespace ScalarAdvection
