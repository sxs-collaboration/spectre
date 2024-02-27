// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Policies/EnforcePolicies.hpp"

#include "Domain/Amr/Flag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "ParallelAlgorithms/Amr/Policies/Isotropy.hpp"
#include "ParallelAlgorithms/Amr/Policies/Limits.hpp"
#include "ParallelAlgorithms/Amr/Policies/Policies.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace amr {
template <size_t Dim>
void enforce_policies(const gsl::not_null<std::array<Flag, Dim>*> amr_decision,
                      const amr::Policies& amr_policies,
                      const ElementId<Dim>& element_id, const Mesh<Dim>& mesh) {
  if (amr_policies.isotropy() == amr::Isotropy::Isotropic) {
    *amr_decision = make_array<Dim>(*alg::max_element(*amr_decision));
  }

  const auto& limits = amr_policies.limits();
  const auto& levels = element_id.refinement_levels();
  for (size_t d = 0; d < Dim; ++d) {
    if (gsl::at(*amr_decision, d) == Flag::Join and
        gsl::at(levels, d) <= limits.minimum_refinement_level()) {
      gsl::at(*amr_decision, d) = Flag::DoNothing;
    }
    if (gsl::at(*amr_decision, d) == Flag::Split and
        gsl::at(levels, d) >= limits.maximum_refinement_level()) {
      gsl::at(*amr_decision, d) = Flag::DoNothing;
    }
    const size_t minimum_resolution = std::max(
        limits.minimum_resolution(), Spectral::detail::minimum_number_of_points(
                                         mesh.basis(d), mesh.quadrature(d)));
    if (gsl::at(*amr_decision, d) == Flag::DecreaseResolution and
        mesh.extents(d) <= minimum_resolution) {
      gsl::at(*amr_decision, d) = Flag::DoNothing;
    }
    if (gsl::at(*amr_decision, d) == Flag::IncreaseResolution and
        mesh.extents(d) >= limits.maximum_resolution()) {
      gsl::at(*amr_decision, d) = Flag::DoNothing;
    }
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                    \
  template void enforce_policies(                               \
      gsl::not_null<std::array<Flag, DIM(data)>*> amr_decision, \
      const amr::Policies& amr_policies,                        \
      const ElementId<DIM(data)>& element_id, const Mesh<DIM(data)>& mesh);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace amr
