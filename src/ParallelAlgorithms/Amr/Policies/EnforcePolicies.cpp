// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Policies/EnforcePolicies.hpp"

#include "Domain/Amr/Flag.hpp"
#include "ParallelAlgorithms/Amr/Policies/Policies.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace amr {
template <size_t Dim>
void enforce_policies(const gsl::not_null<std::array<Flag, Dim>*> amr_decision,
                      const amr::Policies& amr_policies) {
  if (amr_policies.isotropy() == amr::Isotropy::Isotropic) {
    *amr_decision = make_array<Dim>(*alg::max_element(*amr_decision));
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                    \
  template void enforce_policies(                               \
      gsl::not_null<std::array<Flag, DIM(data)>*> amr_decision, \
      const amr::Policies& amr_policies);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace amr
