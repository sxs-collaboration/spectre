// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/Formulation.hpp"

#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"

namespace dg::Tags {
dg::Formulation Formulation::create_from_options(
    const dg::Formulation& formulation) {
  return formulation;
}
}  // namespace dg::Tags
