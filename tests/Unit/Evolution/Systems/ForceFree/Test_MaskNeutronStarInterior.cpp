// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/Systems/ForceFree/MaskNeutronStarInterior.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticData/ForceFree/RotatingDipole.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree {
namespace {

struct MetavariablesForTest {
  using component_list = tmpl::list<>;
  using initial_data_list = tmpl::list<ForceFree::AnalyticData::RotatingDipole>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<evolution::initial_data::InitialData, initial_data_list>>;
  };
};

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ForceFree.MaskNsInterior",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ForceFree"};

  const size_t num_dg_pts = 5;

  const Mesh<3> dg_mesh{num_dg_pts, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  const auto logical_to_grid_map = ElementMap<3, Frame::Grid>{
      ElementId<3>{0},
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
          domain::CoordinateMaps::Identity<3>{})};

  const auto grid_to_inertial_map =
      domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
          domain::CoordinateMaps::Identity<3>{});

  const auto dg_logical_coords = logical_coordinates(dg_mesh);
  const auto subcell_logical_coords = logical_coordinates(subcell_mesh);

  const auto dg_inertial_coords =
      (*grid_to_inertial_map)(logical_to_grid_map(dg_logical_coords));
  const auto subcell_inertial_coords =
      (*grid_to_inertial_map)(logical_to_grid_map(subcell_logical_coords));

  auto box = db::create<db::AddSimpleTags<
      domain::Tags::Coordinates<3, Frame::Inertial>,
      evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>,
      Tags::NsInteriorMask,
      evolution::dg::subcell::Tags::Inactive<Tags::NsInteriorMask>,
      evolution::initial_data::Tags::InitialData>>(
      dg_inertial_coords, subcell_inertial_coords,
      std::optional<Scalar<DataVector>>{}, std::optional<Scalar<DataVector>>{},
      AnalyticData::RotatingDipole{1.0, 0.1, 0.1, 0.1, 0.5}.get_clone());

  // Apply the mutator and check the values of masking variables on both grids.
  db::mutate_apply<MaskNeutronStarInterior<MetavariablesForTest, false>>(
      make_not_null(&box));
  db::mutate_apply<MaskNeutronStarInterior<MetavariablesForTest, true>>(
      make_not_null(&box));

  {
    const auto dg_mask = get<Tags::NsInteriorMask>(box);
    const Scalar<DataVector> dg_mask_from_python{pypp::call<Scalar<DataVector>>(
        "TestFunctions", "compute_ns_interior_mask", dg_inertial_coords)};
    CHECK(dg_mask == dg_mask_from_python);

    const auto subcell_mask =
        get<evolution::dg::subcell::Tags::Inactive<Tags::NsInteriorMask>>(box);
    const Scalar<DataVector> subcell_mask_from_python{
        pypp::call<Scalar<DataVector>>("TestFunctions",
                                       "compute_ns_interior_mask",
                                       subcell_inertial_coords)};
    CHECK(subcell_mask == subcell_mask_from_python);
  }

  // Initialize the mask variables back to `null`, push the coordinates far
  // away from the NS surface, then run the mutator again.
  db::mutate<domain::Tags::Coordinates<3, Frame::Inertial>,
             evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>,
             Tags::NsInteriorMask,
             evolution::dg::subcell::Tags::Inactive<Tags::NsInteriorMask>>(
      [](const auto dg_coords, const auto subcell_coords, const auto dg_mask,
         const auto subcell_mask) {
        for (size_t d = 0; d < 3; ++d) {
          (*dg_coords).get(d) += 5.0;
          (*subcell_coords).get(d) += 5.0;
        }
        *dg_mask = std::optional<Scalar<DataVector>>{};
        *subcell_mask = std::optional<Scalar<DataVector>>{};
      },
      make_not_null(&box));

  db::mutate_apply<MaskNeutronStarInterior<MetavariablesForTest, false>>(
      make_not_null(&box));
  db::mutate_apply<MaskNeutronStarInterior<MetavariablesForTest, true>>(
      make_not_null(&box));

  CHECK(get<Tags::NsInteriorMask>(box).has_value() == false);
  CHECK(get<evolution::dg::subcell::Tags::Inactive<Tags::NsInteriorMask>>(box)
            .has_value() == false);
}

}  // namespace
}  // namespace ForceFree
