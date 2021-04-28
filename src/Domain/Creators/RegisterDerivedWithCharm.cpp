// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
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
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/AlignedLattice.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Cylinder.hpp"
#include "Domain/Creators/Disk.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/FrustalCloak.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Creators/RotatedBricks.hpp"
#include "Domain/Creators/RotatedIntervals.hpp"
#include "Domain/Creators/RotatedRectangles.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Creators/Sphere.hpp"
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
struct to_grid_map<CoordinateMap<Frame::Logical, Frame::Inertial, Maps...>> {
  using type = CoordinateMap<Frame::Logical, Frame::Grid, Maps...>;
};

template <size_t Dim>
using maps_from_creators =
    tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
        typename DomainCreator<Dim>::creatable_classes, get_maps<tmpl::_1>>>>;
}  // namespace

void register_derived_with_charm() {
  using all_maps = tmpl::remove_duplicates<tmpl::append<
      maps_from_creators<1>, maps_from_creators<2>, maps_from_creators<3>>>;
  using maps_to_grid =
      tmpl::remove<tmpl::transform<all_maps, to_grid_map<tmpl::_1>>, void>;
  using maps_to_register =
      tmpl::remove_duplicates<tmpl::append<all_maps, maps_to_grid>>;

  Parallel::register_classes_in_list<maps_to_register>();
}
}  // namespace domain::creators
