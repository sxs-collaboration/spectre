// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <memory>

#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Domain/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace domain {
namespace Tags {
template <size_t VolumeDim>
::Domain<VolumeDim> Domain<VolumeDim>::create_from_options(
    const std::unique_ptr<::DomainCreator<VolumeDim>>& domain_creator) {
  return domain_creator->create_domain();
}

template <size_t Dim>
std::vector<std::array<size_t, Dim>> InitialExtents<Dim>::create_from_options(
    const std::unique_ptr<::DomainCreator<Dim>>& domain_creator) {
  return domain_creator->initial_extents();
}

template <size_t Dim>
std::vector<std::array<size_t, Dim>>
InitialRefinementLevels<Dim>::create_from_options(
    const std::unique_ptr<::DomainCreator<Dim>>& domain_creator) {
  return domain_creator->initial_refinement_levels();
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                              \
  template ::Domain<DIM(data)> Domain<DIM(data)>::create_from_options(    \
      const std::unique_ptr<::DomainCreator<DIM(data)>>& domain_creator); \
  template std::vector<std::array<size_t, DIM(data)>>                     \
  InitialExtents<DIM(data)>::create_from_options(                         \
      const std::unique_ptr<::DomainCreator<DIM(data)>>& domain_creator); \
  template std::vector<std::array<size_t, DIM(data)>>                     \
  InitialRefinementLevels<DIM(data)>::create_from_options(                \
      const std::unique_ptr<::DomainCreator<DIM(data)>>& domain_creator);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace Tags
}  // namespace domain
