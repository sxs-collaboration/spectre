// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/BulgedCube.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/CylindricalEndcap.hpp"
#include "Domain/CoordinateMaps/CylindricalFlatEndcap.hpp"
#include "Domain/CoordinateMaps/CylindricalFlatSide.hpp"
#include "Domain/CoordinateMaps/CylindricalSide.hpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/EquatorialCompression.hpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/Frustum.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Rotation.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Shape.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/RegisterDerivedWithCharm.hpp"
#include "Domain/CoordinateMaps/TimeDependent/SphericalCompression.hpp"
#include "Domain/CoordinateMaps/UniformCylindricalEndcap.hpp"
#include "Domain/CoordinateMaps/UniformCylindricalFlatEndcap.hpp"
#include "Domain/CoordinateMaps/UniformCylindricalSide.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/Factory.hpp"
#include "Domain/Creators/Factory1D.hpp"
#include "Domain/Creators/Factory2D.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::creators {
namespace {
template <typename Creator>
struct get_maps {
  using type = typename Creator::maps_list;
};

template <typename Map>
struct to_grid_map {
  using type = void;
};

template <typename... Maps>
struct to_grid_map<
    CoordinateMap<Frame::BlockLogical, Frame::Inertial, Maps...>> {
  using type = CoordinateMap<Frame::BlockLogical, Frame::Grid, Maps...>;
};

template <size_t Dim>
using maps_from_creators = tmpl::remove_duplicates<
    tmpl::flatten<tmpl::transform<domain_creators<Dim>, get_maps<tmpl::_1>>>>;
}  // namespace

void register_derived_with_charm() {
  using all_maps = tmpl::remove_duplicates<tmpl::append<
      maps_from_creators<1>, maps_from_creators<2>, maps_from_creators<3>>>;
  using maps_to_grid =
      tmpl::remove<tmpl::transform<all_maps, to_grid_map<tmpl::_1>>, void>;
  using maps_to_register =
      tmpl::remove_duplicates<tmpl::append<all_maps, maps_to_grid>>;

  Parallel::register_classes_with_charm(maps_to_register{});

  domain::CoordinateMaps::ShapeMapTransitionFunctions::
      register_derived_with_charm();
}
}  // namespace domain::creators
