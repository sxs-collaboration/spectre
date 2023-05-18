// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <iterator>
#include <unordered_set>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/FiniteDifference/Filter.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace {
// Compute polynomial on cell centers in FD cluster of points
template <size_t Dim>
void set_polynomial(
    const gsl::not_null<DataVector*> var1_ptr,
    const gsl::not_null<DataVector*> var2_ptr,
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& local_logical_coords,
    const size_t degree) {
  *var1_ptr = 0.0;
  *var2_ptr = 100.0;  // some constant offset to distinguish the var values
  for (size_t i = 0; i < Dim; ++i) {
    *var1_ptr += pow(local_logical_coords.get(i), degree);
    *var2_ptr += pow(local_logical_coords.get(i), degree);
  }
}

template <size_t Dim>
void set_solution(
    const gsl::not_null<DataVector*> volume_vars,
    const gsl::not_null<DirectionMap<Dim, DataVector>*> neighbor_data,
    const gsl::not_null<DirectionMap<Dim, gsl::span<const double>>*>
        ghost_cell_vars,
    const Mesh<Dim>& mesh, const size_t number_of_vars,
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& logical_coords,
    const size_t deriv_order, const size_t degree) {
  *volume_vars = DataVector{mesh.number_of_grid_points() * number_of_vars, 0.0};
  DataVector var1(volume_vars->data(), mesh.number_of_grid_points());
  DataVector var2(
      std::next(volume_vars->data(),
                static_cast<std::ptrdiff_t>(mesh.number_of_grid_points())),
      mesh.number_of_grid_points());
  set_polynomial(&var1, &var2, logical_coords, degree);

  for (const auto& direction : Direction<Dim>::all_directions()) {
    auto neighbor_logical_coords = logical_coords;
    neighbor_logical_coords.get(direction.dimension()) +=
        direction.sign() * 2.0;
    DataVector neighbor_vars{mesh.number_of_grid_points() * number_of_vars,
                             0.0};
    DataVector neighbor_var1(neighbor_vars.data(),
                             mesh.number_of_grid_points());
    DataVector neighbor_var2(
        neighbor_vars.data() + mesh.number_of_grid_points(),  // NOLINT
        mesh.number_of_grid_points());
    set_polynomial(&neighbor_var1, &neighbor_var2, neighbor_logical_coords,
                   degree);

    const auto sliced_data = evolution::dg::subcell::detail::slice_data_impl(
        gsl::make_span(neighbor_vars.data(), neighbor_vars.size()),
        mesh.extents(), deriv_order / 2 + 1,
        std::unordered_set{direction.opposite()}, 0);
    CAPTURE(deriv_order / 2 + 1);
    REQUIRE(sliced_data.size() == 1);
    REQUIRE(sliced_data.contains(direction.opposite()));
    (*neighbor_data)[direction] = sliced_data.at(direction.opposite());
    (*ghost_cell_vars)[direction] = gsl::make_span(
        (*neighbor_data)[direction].data(), (*neighbor_data)[direction].size());
    REQUIRE(neighbor_data->at(direction).size() ==
            number_of_vars * (deriv_order / 2 + 1) *
                mesh.slice_away(0).number_of_grid_points());
  }
}

template <size_t Dim>
void test_ko_dissipation() {
  CAPTURE(Dim);
  const size_t number_of_vars = 2;
  const double epsilon = 0.1;

  const Mesh<Dim> mesh{13, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered};
  auto logical_coords = logical_coordinates(mesh);
  // Make the logical coordinates different in each direction
  for (size_t i = 1; i < Dim; ++i) {
    logical_coords.get(i) += 4.0 * i;
  }

  for (size_t deriv_order = 2; deriv_order < 12; deriv_order += 2) {
    CAPTURE(deriv_order);
    DataVector volume_vars{};
    DirectionMap<Dim, DataVector> neighbor_data{};
    DirectionMap<Dim, gsl::span<const double>> ghost_cell_vars{};
    set_solution(make_not_null(&volume_vars), make_not_null(&neighbor_data),
                 make_not_null(&ghost_cell_vars), mesh, number_of_vars,
                 logical_coords, deriv_order, deriv_order - 1);

    DataVector filtered_vars{mesh.number_of_grid_points() * number_of_vars,
                             0.0};
    auto filtered_vars_span =
        gsl::make_span(filtered_vars.data(), filtered_vars.size());
    fd::kreiss_oliger_filter(
        make_not_null(&filtered_vars_span),
        gsl::make_span(volume_vars.data(), volume_vars.size()), ghost_cell_vars,
        mesh, number_of_vars, deriv_order, epsilon);

    // Get only the KO dissipation term by subtracting off the volume
    // variables from the filtered data.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    const DataVector view_volume_vars{const_cast<double*>(volume_vars.data()),
                                      volume_vars.size()};
    DataVector view_filtered_vars{filtered_vars.data(), filtered_vars.size()};
    view_filtered_vars -= view_volume_vars;
    if (Dim > 1 and deriv_order == 10) {
      Approx custom_approx = Approx::custom().epsilon(1.0e-10).scale(1.0);
      CHECK_ITERABLE_CUSTOM_APPROX(
          view_filtered_vars,
          DataVector(mesh.number_of_grid_points() * number_of_vars, 0.0),
          custom_approx);
    } else {
      CHECK_ITERABLE_APPROX(
          view_filtered_vars,
          DataVector(mesh.number_of_grid_points() * number_of_vars, 0.0));
    }
  }
}

template <size_t Dim>
void test_low_pass_filter() {
  CAPTURE(Dim);
  const size_t number_of_vars = 2;
  const double epsilon = 1.0;

  const Mesh<Dim> mesh{13, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered};
  auto logical_coords = logical_coordinates(mesh);
  // Make the logical coordinates different in each direction
  for (size_t i = 1; i < Dim; ++i) {
    logical_coords.get(i) += 4.0 * i;
  }

  // Large-ish relative error in 3d
  Approx custom_approx = Approx::custom().epsilon(1.0e-8).scale(1.0);
  for (size_t deriv_order = 2; deriv_order < 10; deriv_order += 2) {
    CAPTURE(deriv_order);
    for (size_t degree = 0; degree < deriv_order; ++degree) {
      CAPTURE(degree);
      DataVector volume_vars{};
      DirectionMap<Dim, DataVector> neighbor_data{};
      DirectionMap<Dim, gsl::span<const double>> ghost_cell_vars{};
      set_solution(make_not_null(&volume_vars), make_not_null(&neighbor_data),
                   make_not_null(&ghost_cell_vars), mesh, number_of_vars,
                   logical_coords, deriv_order, degree);

      DataVector filtered_vars{mesh.number_of_grid_points() * number_of_vars,
                               0.0};
      auto filtered_vars_span =
          gsl::make_span(filtered_vars.data(), filtered_vars.size());

      fd::low_pass_filter(
          make_not_null(&filtered_vars_span),
          gsl::make_span(volume_vars.data(), volume_vars.size()),
          ghost_cell_vars, mesh, number_of_vars, deriv_order, epsilon);

      // Get only the low-pass filter term by subtracting off the volume
      // variables from the filtered data.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const DataVector view_volume_vars{const_cast<double*>(volume_vars.data()),
                                        volume_vars.size()};
      DataVector view_filtered_vars{filtered_vars.data(), filtered_vars.size()};
      view_filtered_vars -= view_volume_vars;
      CHECK_ITERABLE_CUSTOM_APPROX(
          view_filtered_vars,
          DataVector(mesh.number_of_grid_points() * number_of_vars, 0.0),
          custom_approx);
    }
  }
}

SPECTRE_TEST_CASE("Unit.FiniteDifference.Filter",
                  "[Unit][NumericalAlgorithms]") {
  test_ko_dissipation<1>();
  test_ko_dissipation<2>();
  test_ko_dissipation<3>();

  test_low_pass_filter<1>();
  test_low_pass_filter<2>();
  test_low_pass_filter<3>();
}
}  // namespace
