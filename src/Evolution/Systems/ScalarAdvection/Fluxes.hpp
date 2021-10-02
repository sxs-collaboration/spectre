// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarAdvection {

/*!
 * \brief Compute the fluxes of the ScalarAdvection system \f$F^i = v^iU\f$
 * where \f$v^i\f$ is the velocity field.
 */
template <size_t Dim>
struct Fluxes {
  using argument_tags = tmpl::list<Tags::U, Tags::VelocityField<Dim>>;
  using return_tags =
      tmpl::list<::Tags::Flux<Tags::U, tmpl::size_t<Dim>, Frame::Inertial>>;
  static void apply(gsl::not_null<tnsr::I<DataVector, Dim>*> u_flux,
                    const Scalar<DataVector>& u,
                    const tnsr::I<DataVector, Dim>& velocity_field);
};
}  // namespace ScalarAdvection
