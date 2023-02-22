// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Tags/InitialExtents.hpp"

#include <array>
#include <cstddef>
#include <memory>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace domain::Tags {
template <size_t Dim>
std::vector<std::array<size_t, Dim>> InitialExtents<Dim>::create_from_options(
    const std::unique_ptr<::DomainCreator<Dim>>& domain_creator) {
  return domain_creator->initial_extents();
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                          \
  template std::vector<std::array<size_t, DIM(data)>> \
  InitialExtents<DIM(data)>::create_from_options(     \
      const std::unique_ptr<::DomainCreator<DIM(data)>>& domain_creator);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace domain::Tags
