// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <memory>
#include <pup.h>

#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/TimeDependence/CubicScale.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace domain {
namespace creators {
namespace time_dependence {
namespace {
template <typename TimeDep>
struct get_maps {
  using type = typename TimeDep::maps_list;
};

template <size_t Dim>
void register_maps_with_charm() noexcept {
  using maps_to_register = tmpl::remove_duplicates<tmpl::flatten<
      tmpl::transform<typename TimeDependence<Dim>::creatable_classes,
                      get_maps<tmpl::_1>>>>;

  Parallel::register_classes_in_list<maps_to_register>();
}
}  // namespace

void register_derived_with_charm() noexcept {
  register_maps_with_charm<1>();
  register_maps_with_charm<2>();
  register_maps_with_charm<3>();
}
}  // namespace time_dependence
}  // namespace creators
}  // namespace domain
