// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/FunctionsOfTime/Tags.hpp"

#include <memory>

#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace domain {
namespace Tags {
template <size_t Dim>
auto InitialFunctionsOfTime<Dim>::create_from_options(
    const std::unique_ptr<::DomainCreator<Dim>>& domain_creator) noexcept
    -> type {
  return domain_creator->functions_of_time();
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                            \
  template auto InitialFunctionsOfTime<DIM(data)>::create_from_options( \
      const std::unique_ptr<::DomainCreator<DIM(data)>>&                \
          domain_creator) noexcept->type;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace Tags
}  // namespace domain
