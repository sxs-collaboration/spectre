// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Tags/Domain.hpp"

#include <cstddef>
#include <memory>

#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace domain::Tags {
template <size_t VolumeDim>
::Domain<VolumeDim> Domain<VolumeDim>::create_from_options(
    const std::unique_ptr<::DomainCreator<VolumeDim>>& domain_creator) {
  return domain_creator->create_domain();
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                           \
  template ::Domain<DIM(data)> Domain<DIM(data)>::create_from_options( \
      const std::unique_ptr<::DomainCreator<DIM(data)>>& domain_creator);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace domain::Tags
