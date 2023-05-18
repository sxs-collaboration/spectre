// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Filters.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/PrimReconstructor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim>
void set_polynomial(
    const gsl::not_null<std::vector<DataVector>*> vars_ptr,
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& local_logical_coords,
    const size_t degree) {
  for (auto& var : *vars_ptr) {
    var = 0.0;
    for (size_t i = 0; i < Dim; ++i) {
      var += pow(local_logical_coords.get(i), degree);
    }
  }
}

void set_solution(
    const gsl::not_null<Variables<
        typename grmhd::GhValenciaDivClean::System::variables_tag::tags_list>*>
        volume_vars,
    const gsl::not_null<FixedHashMap<
        maximum_number_of_neighbors(3), std::pair<Direction<3>, ElementId<3>>,
        evolution::dg::subcell::GhostData,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>*>
        neighbor_data,
    const Mesh<3>& mesh,
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& logical_coords,
    const size_t deriv_order, const size_t degree) {
  const auto set_data_vectors =
      [](const gsl::not_null<std::vector<DataVector>*> local_dvs,
         const auto local_vars) {
        for (size_t i = 0; i < 10; ++i) {
          (*local_dvs)[i].set_data_ref(make_not_null(
              &get<gr::Tags::SpacetimeMetric<DataVector, 3>>(*local_vars)[i]));
        }
        for (size_t i = 0; i < 10; ++i) {
          (*local_dvs)[i + 10].set_data_ref(
              make_not_null(&get<gh::Tags::Pi<DataVector, 3>>(*local_vars)[i]));
        }
        for (size_t i = 0; i < 30; ++i) {
          (*local_dvs)[i + 20].set_data_ref(make_not_null(
              &get<gh::Tags::Phi<DataVector, 3>>(*local_vars)[i]));
        }
      };
  std::vector<DataVector> vars(50);
  set_data_vectors(make_not_null(&vars), volume_vars);
  set_polynomial(&vars, logical_coords, degree);

  for (const auto& direction : Direction<3>::all_directions()) {
    auto neighbor_logical_coords = logical_coords;
    neighbor_logical_coords.get(direction.dimension()) +=
        direction.sign() * 2.0;
    std::vector<DataVector> neighbor_dvs(50);
    Variables<grmhd::GhValenciaDivClean::Tags::
                  primitive_grmhd_and_spacetime_reconstruction_tags>
        neighbor_vars{mesh.number_of_grid_points()};
    set_data_vectors(make_not_null(&neighbor_dvs),
                     make_not_null(&neighbor_vars));
    set_polynomial(&neighbor_dvs, neighbor_logical_coords, degree);

    const auto sliced_data = evolution::dg::subcell::detail::slice_data_impl(
        gsl::make_span(neighbor_vars), mesh.extents(), deriv_order / 2 + 1,
        std::unordered_set{direction.opposite()}, 0);
    CAPTURE(deriv_order / 2 + 1);
    REQUIRE(sliced_data.size() == 1);
    REQUIRE(sliced_data.contains(direction.opposite()));
    const auto key = std::pair{direction, ElementId<3>{0}};
    (*neighbor_data)[key] = evolution::dg::subcell::GhostData{1};
    (*neighbor_data)[key].neighbor_ghost_data_for_reconstruction() =
        sliced_data.at(direction.opposite());
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GrMhd.GhValenciaDivClean.Fd.Filters",
                  "[Unit][Evolution]") {
  const size_t points_per_dimension = 5;
  const Mesh<3> subcell_mesh{points_per_dimension,
                             Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};
  const auto logical_coords = TestHelpers::grmhd::GhValenciaDivClean::fd::
      detail::set_logical_coordinates(subcell_mesh);

  Variables<
      typename grmhd::GhValenciaDivClean::System::variables_tag::tags_list>
      result{subcell_mesh.number_of_grid_points(), 0.0};
  Variables<
      typename grmhd::GhValenciaDivClean::System::variables_tag::tags_list>
      volume_evolved_variables{subcell_mesh.number_of_grid_points()};

  FixedHashMap<maximum_number_of_neighbors(3),
               std::pair<Direction<3>, ElementId<3>>,
               evolution::dg::subcell::GhostData,
               boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      neighbor_data_for_reconstruction{};

  set_solution(&volume_evolved_variables, &neighbor_data_for_reconstruction,
               subcell_mesh, logical_coords, 4, 3);

  grmhd::GhValenciaDivClean::fd::spacetime_kreiss_oliger_filter(
      make_not_null(&result), volume_evolved_variables,
      neighbor_data_for_reconstruction, subcell_mesh, 4, 1.0);

  tmpl::for_each<
      grmhd::GhValenciaDivClean::Tags::spacetime_reconstruction_tags>(
      [&result, &volume_evolved_variables](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        auto& result_tensor = get<tag>(result);
        auto& volume_tensor = get<tag>(volume_evolved_variables);
        Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
        for (size_t tensor_index = 0; tensor_index < result_tensor.size();
             ++tensor_index) {
          result_tensor[tensor_index] -= volume_tensor[tensor_index];
          CHECK_ITERABLE_CUSTOM_APPROX(
              result_tensor[tensor_index],
              DataVector(result_tensor[tensor_index].size(), 0.0),
              custom_approx);
        }
      });
}
}  // namespace
