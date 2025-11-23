// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/ApplyFilter.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Filter.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/Ccz4/PrimReconstructor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
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

template <typename System>
void set_solution(
    const gsl::not_null<Variables<typename System::variables_tag::tags_list>*>
        volume_vars,
    const gsl::not_null<DirectionalIdMap<3, evolution::dg::subcell::GhostData>*>
        neighbor_data,
    const Mesh<3>& mesh,
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& logical_coords,
    const size_t deriv_order, const size_t degree) {
  const auto set_data_vectors = [](const gsl::not_null<std::vector<DataVector>*>
                                       local_dvs,
                                   const auto local_vars) {
    for (size_t i = 0; i < 6; ++i) {
      (*local_dvs)[i].set_data_ref(make_not_null(
          &get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(*local_vars)[i]));
    }
    for (size_t i = 0; i < 1; ++i) {
      (*local_dvs)[i + 10].set_data_ref(make_not_null(
          &get<::Ccz4::Tags::ConformalFactor<DataVector>>(*local_vars)[i]));
    }
    for (size_t i = 0; i < 6; ++i) {
      (*local_dvs)[i + 11].set_data_ref(make_not_null(
          &get<::Ccz4::Tags::ATilde<DataVector, 3>>(*local_vars)[i]));
    }
    for (size_t i = 0; i < 1; ++i) {
      (*local_dvs)[i + 17].set_data_ref(make_not_null(
          &get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(*local_vars)[i]));
    }
    for (size_t i = 0; i < 1; ++i) {
      (*local_dvs)[i + 18].set_data_ref(
          make_not_null(&get<::Ccz4::Tags::Theta<DataVector>>(*local_vars)[i]));
    }
    for (size_t i = 0; i < 3; ++i) {
      (*local_dvs)[i + 19].set_data_ref(make_not_null(
          &get<::Ccz4::Tags::GammaHat<DataVector, 3>>(*local_vars)[i]));
    }
    for (size_t i = 0; i < 1; ++i) {
      (*local_dvs)[i + 6].set_data_ref(
          make_not_null(&get<gr::Tags::Lapse<DataVector>>(*local_vars)[i]));
    }
    for (size_t i = 0; i < 3; ++i) {
      (*local_dvs)[i + 7].set_data_ref(
          make_not_null(&get<gr::Tags::Shift<DataVector, 3>>(*local_vars)[i]));
    }
    for (size_t i = 0; i < 3; ++i) {
      (*local_dvs)[i + 22].set_data_ref(make_not_null(
          &get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(*local_vars)[i]));
    }
  };

  std::vector<DataVector> vars(25);
  set_data_vectors(make_not_null(&vars), volume_vars);
  set_polynomial(&vars, logical_coords, degree);

  for (const auto& direction : Direction<3>::all_directions()) {
    auto neighbor_logical_coords = logical_coords;
    neighbor_logical_coords.get(direction.dimension()) +=
        direction.sign() * 2.0;
    std::vector<DataVector> neighbor_dvs(25);
    Variables<::Ccz4::fd::System::variables_tag_list> neighbor_vars{
        mesh.number_of_grid_points()};

    set_data_vectors(make_not_null(&neighbor_dvs),
                     make_not_null(&neighbor_vars));
    set_polynomial(&neighbor_dvs, neighbor_logical_coords, degree);

    const auto sliced_data = evolution::dg::subcell::detail::slice_data_impl(
        gsl::make_span(neighbor_vars), mesh.extents(), deriv_order / 2 + 1,
        std::unordered_set{direction.opposite()}, 0, {});
    CAPTURE(deriv_order / 2 + 1);
    REQUIRE(sliced_data.size() == 1);
    REQUIRE(sliced_data.contains(direction.opposite()));
    const auto key = DirectionalId<3>{direction, ElementId<3>{0}};
    (*neighbor_data)[key] = evolution::dg::subcell::GhostData{1};
    (*neighbor_data)[key].neighbor_ghost_data_for_reconstruction() =
        sliced_data.at(direction.opposite());
  }
}

void test(const bool evolve_lapse_and_shift) {
  const size_t points_per_dimension = 6;
  const Mesh<3> subcell_mesh{points_per_dimension,
                             Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};
  const auto logical_coords =
      TestHelpers::Ccz4::fd::detail::set_logical_coordinates(subcell_mesh);

  using System = ::Ccz4::fd::System;

  Variables<System::variables_tag_list> volume_evolved_variables{
      subcell_mesh.number_of_grid_points()};

  DirectionalIdMap<3, evolution::dg::subcell::GhostData>
      neighbor_data_for_reconstruction{};

  set_solution<System>(&volume_evolved_variables,
                       &neighbor_data_for_reconstruction, subcell_mesh,
                       logical_coords, 6, 5);

  // Store the original variables for comparison
  Variables<System::variables_tag_list> original_variables =
      volume_evolved_variables;

  // Set up the DataBox with all required tags
  const double kreiss_oliger_epsilon = 0.5;  // Use valid epsilon value

  auto box = db::create<db::AddSimpleTags<
      System::variables_tag, evolution::dg::subcell::Tags::Mesh<3>,
      Ccz4::fd::Tags::EvolveLapseAndShift, Ccz4::fd::Tags::KreissOligerEpsilon,
      evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>>(
      volume_evolved_variables, subcell_mesh,
      evolve_lapse_and_shift, kreiss_oliger_epsilon,
      neighbor_data_for_reconstruction);

  // Apply the filter through the mutator
  db::mutate_apply<Ccz4::fd::ApplyFilter>(make_not_null(&box));

  // Get the result from the box
  const auto& result = get<System::variables_tag>(box);

  if (not evolve_lapse_and_shift) {
    CHECK(get<gr::Tags::Lapse<DataVector>>(result) ==
          get<gr::Tags::Lapse<DataVector>>(original_variables));
    CHECK(get<gr::Tags::Shift<DataVector, 3>>(result) ==
          get<gr::Tags::Shift<DataVector, 3>>(original_variables));
    CHECK(
        get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(result) ==
        get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(original_variables));
  }

  tmpl::for_each<System::variables_tag_list>([&result,
                                              &original_variables](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    const auto& result_tensor = get<tag>(result);
    const auto& volume_tensor = get<tag>(original_variables);
    CAPTURE(pretty_type::name<tag>());
    const Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
    for (size_t tensor_index = 0; tensor_index < result_tensor.size();
         ++tensor_index) {
      CHECK_ITERABLE_CUSTOM_APPROX(result_tensor[tensor_index],
                                   volume_tensor[tensor_index], custom_approx);
    }
  });
}

void test_error_when_epsilon_out_of_range() {
  const size_t points_per_dimension = 6;
  const Mesh<3> subcell_mesh{points_per_dimension,
                             Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};
  const auto logical_coords =
      TestHelpers::Ccz4::fd::detail::set_logical_coordinates(subcell_mesh);

  using System = ::Ccz4::fd::System;

  Variables<System::variables_tag_list> volume_evolved_variables{
      subcell_mesh.number_of_grid_points()};

  DirectionalIdMap<3, evolution::dg::subcell::GhostData>
      neighbor_data_for_reconstruction{};

  set_solution<System>(&volume_evolved_variables,
                       &neighbor_data_for_reconstruction, subcell_mesh,
                       logical_coords, 6, 5);

  // Test with epsilon = 0.0 (should throw error)
  {
    auto box = db::create<db::AddSimpleTags<
        System::variables_tag, evolution::dg::subcell::Tags::Mesh<3>,
        Ccz4::fd::Tags::EvolveLapseAndShift,
        Ccz4::fd::Tags::KreissOligerEpsilon,
        evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>>(
        volume_evolved_variables, subcell_mesh, true, -0.1,
        neighbor_data_for_reconstruction);

    CHECK_THROWS_WITH(
        db::mutate_apply<Ccz4::fd::ApplyFilter>(make_not_null(&box)),
        Catch::Matchers::ContainsSubstring(
            "Kreiss-Oliger epsilon should be in the interval [0, 1]"));
  }

  // Test with epsilon = 1.0 (should throw error)
  {
    auto box = db::create<db::AddSimpleTags<
        System::variables_tag, evolution::dg::subcell::Tags::Mesh<3>,
        Ccz4::fd::Tags::EvolveLapseAndShift,
        Ccz4::fd::Tags::KreissOligerEpsilon,
        evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>>(
        volume_evolved_variables, subcell_mesh, true, 1.1,
        neighbor_data_for_reconstruction);

    CHECK_THROWS_WITH(
        db::mutate_apply<Ccz4::fd::ApplyFilter>(make_not_null(&box)),
        Catch::Matchers::ContainsSubstring(
            "Kreiss-Oliger epsilon should be in the interval [0, 1]"));
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Fd.ApplyFilter",
                  "[Unit][Evolution]") {
  test(true);
  test(false);
  test_error_when_epsilon_out_of_range();
}
}  // namespace
