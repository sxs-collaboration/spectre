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
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
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
        std::unordered_set{direction.opposite()}, 0, {});
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

// Test filtering with Cartoon bases (axially or spherically symmetric meshes).
//
// For cartoon meshes, `filter_impl` determines a computational dimension
// `comp_dim < Dim` and only filters along those directions.  We verify the
// filter correction vanishes when the data is a low-degree polynomial in the
// FD dimensions only.
template <bool Spherical>
void test_cartoon_filter() {
  CAPTURE(Spherical);
  constexpr size_t comp_dim = Spherical ? 1 : 2;
  const size_t number_of_vars = 2;
  const double epsilon_ko = 0.1;
  const double epsilon_lp = 1.0;

  for (size_t deriv_order = 2; deriv_order < 10; deriv_order += 2) {
    CAPTURE(deriv_order);
    const size_t n = deriv_order + 3;  // enough points for the stencil

    Mesh<3> mesh{};
    if constexpr (Spherical) {
      mesh = Mesh<3>{{n, 1, 1},
                     {Spectral::Basis::FiniteDifference,
                      Spectral::Basis::Cartoon, Spectral::Basis::Cartoon},
                     {Spectral::Quadrature::CellCentered,
                      Spectral::Quadrature::SphericalSymmetry,
                      Spectral::Quadrature::SphericalSymmetry}};
    } else {
      mesh =
          Mesh<3>{{n, n, 1},
                  {Spectral::Basis::FiniteDifference,
                   Spectral::Basis::FiniteDifference, Spectral::Basis::Cartoon},
                  {Spectral::Quadrature::CellCentered,
                   Spectral::Quadrature::CellCentered,
                   Spectral::Quadrature::AxialSymmetry}};
    }

    // FD sub-mesh and its logical coordinates (comp_dim components).
    // Because each Cartoon dim has exactly 1 grid point, the FD sub-mesh
    // grid-point ordering matches the full 3D mesh ordering.
    const Mesh<comp_dim> fd_submesh = [&]() {
      if constexpr (Spherical) {
        return mesh.slice_through(0);
      } else {
        return mesh.slice_through(0, 1);
      }
    }();
    const auto fd_coords = logical_coordinates(fd_submesh);

    const size_t num_pts = mesh.number_of_grid_points();

    // Evaluate sum_d coords.get(d)^degree at given FD coords.
    const auto poly_at =
        [&](const size_t degree,
            const tnsr::I<DataVector, comp_dim, Frame::ElementLogical>& coords)
        -> DataVector {
      DataVector val(num_pts, 0.0);
      for (size_t d = 0; d < comp_dim; ++d) {
        val += pow(coords.get(d), degree);
      }
      return val;
    };
    const auto poly = [&](const size_t degree) -> DataVector {
      return poly_at(degree, fd_coords);
    };

    // Build ghost data for directions 0..comp_dim-1 only.
    // For each neighbor, shift its logical coords by \pm 2 in the FD direction
    const auto make_ghost_data =
        [&](const size_t degree,
            const size_t ghost_pts) -> DirectionMap<3, DataVector> {
      DirectionMap<3, DataVector> nbr_store{};

      for (size_t d = 0; d < comp_dim; ++d) {
        for (const auto side : {Side::Lower, Side::Upper}) {
          auto nbr_coords = fd_coords;
          nbr_coords.get(d) += (side == Side::Lower ? -1.0 : 1.0) * 2.0;

          DataVector nbr_vars(num_pts * number_of_vars, 0.0);
          for (size_t v = 0; v < number_of_vars; ++v) {
            DataVector vview(nbr_vars.data() + v * num_pts, num_pts);
            vview =
                poly_at(degree, nbr_coords) + static_cast<double>(v) * 100.0;
          }

          const Direction<comp_dim> dir_comp{d, side};
          const Direction<3> dir3{d, side};
          const auto sliced = evolution::dg::subcell::detail::slice_data_impl(
              gsl::make_span(nbr_vars.data(), nbr_vars.size()),
              fd_submesh.extents(), ghost_pts,
              std::unordered_set{dir_comp.opposite()}, 0, {});
          REQUIRE(sliced.contains(dir_comp.opposite()));
          nbr_store[dir3] = sliced.at(dir_comp.opposite());
        }
      }
      return nbr_store;
    };

    const auto make_ghost_spans =
        [](const DirectionMap<3, DataVector>& nbr_store)
        -> DirectionMap<3, gsl::span<const double>> {
      DirectionMap<3, gsl::span<const double>> ghost{};
      for (const auto& [dir, data] : nbr_store) {
        ghost[dir] = gsl::make_span(data.data(), data.size());
      }
      return ghost;
    };

    // Build volume vars for a given degree (two vars, second offset by 100)
    const auto make_volume_vars = [&](const size_t degree) -> DataVector {
      DataVector vvars(num_pts * number_of_vars, 0.0);
      const DataVector p = poly(degree);
      for (size_t v = 0; v < number_of_vars; ++v) {
        DataVector vview(vvars.data() + v * num_pts, num_pts);
        vview = p + static_cast<double>(v) * 100.0;
      }
      return vvars;
    };

    // Helper to apply a filter and return the correction (filtered - volume)
    const auto filter_correction =
        [&](const DataVector& volume_vars,
            const DirectionMap<3, gsl::span<const double>>& ghost_cell_vars,
            const bool use_ko) -> DataVector {
      DataVector filtered(num_pts * number_of_vars, 0.0);
      auto fspan = gsl::make_span(filtered.data(), filtered.size());
      const auto vspan = gsl::make_span(volume_vars.data(), volume_vars.size());
      if (use_ko) {
        fd::kreiss_oliger_filter(make_not_null(&fspan), vspan, ghost_cell_vars,
                                 mesh, number_of_vars, deriv_order, epsilon_ko);
      } else {
        fd::low_pass_filter(make_not_null(&fspan), vspan, ghost_cell_vars, mesh,
                            number_of_vars, deriv_order, epsilon_lp);
      }
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const DataVector vview{const_cast<double*>(volume_vars.data()),
                             volume_vars.size()};
      DataVector fview{filtered.data(), filtered.size()};
      fview -= vview;
      return filtered;
    };

    const size_t ghost_pts = deriv_order / 2 + 1;

    // KO dissipation: degree = deriv_order - 1 should give zero correction
    {
      const DataVector vvars = make_volume_vars(deriv_order - 1);
      const DirectionMap<3, DataVector> nbr_store =
          make_ghost_data(deriv_order - 1, ghost_pts);
      const DirectionMap<3, gsl::span<const double>> ghost =
          make_ghost_spans(nbr_store);
      const DataVector corr = filter_correction(vvars, ghost, /*use_ko=*/true);
      const Approx custom_approx = Approx::custom().epsilon(1.0e-14).scale(1.0);
      CHECK_ITERABLE_CUSTOM_APPROX(
          corr, DataVector(num_pts * number_of_vars, 0.0), custom_approx);
    }
    // Low-pass: all degrees < fd_order should give zero correction
    {
      const Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
      for (size_t degree = 0; degree < deriv_order; ++degree) {
        CAPTURE(degree);
        const DataVector vvars = make_volume_vars(degree);
        const DirectionMap<3, DataVector> nbr_store =
            make_ghost_data(degree, ghost_pts);
        const DirectionMap<3, gsl::span<const double>> ghost =
            make_ghost_spans(nbr_store);
        const DataVector corr =
            filter_correction(vvars, ghost, /*use_ko=*/false);
        CHECK_ITERABLE_CUSTOM_APPROX(
            corr, DataVector(num_pts * number_of_vars, 0.0), custom_approx);
      }
    }
  }
}

// Build minimal but valid ghost data for a 1D mesh so that asserts that fire
// before the ghost-data checks can be reached cleanly.
DirectionMap<1, gsl::span<const double>> make_valid_ghost_spans_1d(
    const DirectionMap<1, DataVector>& store) {
  DirectionMap<1, gsl::span<const double>> ghost{};
  for (const auto& [dir, data] : store) {
    ghost[dir] = gsl::make_span(data.data(), data.size());
  }
  return ghost;
}

void test_asserts() {
  const size_t number_of_vars = 1;
  const double epsilon = 0.1;
  {
    INFO("Unsupported fd_order value");
    const Mesh<1> mesh{5, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered};
    DataVector vol(5, 1.0);
    const DataVector lower_ghost(2, 1.0);
    const DataVector upper_ghost(2, 1.0);
    DirectionMap<1, DataVector> store{};
    store[Direction<1>::lower_xi()] = lower_ghost;
    store[Direction<1>::upper_xi()] = upper_ghost;
    const auto ghost = make_valid_ghost_spans_1d(store);
    DataVector filtered(5, 0.0);
    auto fspan = gsl::make_span(filtered.data(), filtered.size());
    const auto vspan = gsl::make_span(vol.data(), vol.size());
    CHECK_THROWS_WITH(
        fd::kreiss_oliger_filter(make_not_null(&fspan), vspan, ghost, mesh,
                                 number_of_vars, /*fd_order=*/3, epsilon),
        Catch::Matchers::ContainsSubstring(
            "Cannot do finite difference filter of order 3"));
    CHECK_THROWS_WITH(
        fd::low_pass_filter(make_not_null(&fspan), vspan, ghost, mesh,
                            number_of_vars, /*fd_order=*/7, epsilon),
        Catch::Matchers::ContainsSubstring(
            "Cannot do finite difference filter of order 7"));
  }
#ifdef SPECTRE_DEBUG
  {
    INFO("Missing ghost data directions");
    const Mesh<1> mesh{5, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered};
    DataVector vol(5, 1.0);
    DataVector filtered(5, 0.0);
    auto fspan = gsl::make_span(filtered.data(), filtered.size());
    const auto vspan = gsl::make_span(vol.data(), vol.size());
    {
      INFO("Missing lower-xi ghost data");
      const DataVector upper_ghost(2, 1.0);
      DirectionMap<1, DataVector> store{};
      store[Direction<1>::upper_xi()] = upper_ghost;
      const auto ghost = make_valid_ghost_spans_1d(store);
      CHECK_THROWS_WITH(
          fd::kreiss_oliger_filter(make_not_null(&fspan), vspan, ghost, mesh,
                                   number_of_vars, 2, epsilon),
          Catch::Matchers::ContainsSubstring(
              "Couldn't find lower ghost data in lower-xi"));
    }
    {
      INFO("Missing upper-xi ghost data");
      const DataVector lower_ghost(2, 1.0);
      DirectionMap<1, DataVector> store{};
      store[Direction<1>::lower_xi()] = lower_ghost;
      const auto ghost = make_valid_ghost_spans_1d(store);
      CHECK_THROWS_WITH(
          fd::kreiss_oliger_filter(make_not_null(&fspan), vspan, ghost, mesh,
                                   number_of_vars, 2, epsilon),
          Catch::Matchers::ContainsSubstring(
              "Couldn't find upper ghost data in upper-xi"));
    }
  }
  {
    INFO("Non-isotropic mesh (1D path)");
    const Mesh<1> mesh_aniso{5, Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};
    // Override with a mesh whose extents differ from isotropic—only reachable
    // for Dim == 1 when basis(0) != FiniteDifference.
    // Easiest: wrong basis.
    const Mesh<1> mesh_bad{5, Spectral::Basis::Chebyshev,
                           Spectral::Quadrature::GaussLobatto};
    const DataVector vol(5, 1.0);
    const DataVector lower_ghost(2, 1.0);
    const DataVector upper_ghost(2, 1.0);
    DirectionMap<1, DataVector> store{};
    store[Direction<1>::lower_xi()] = lower_ghost;
    store[Direction<1>::upper_xi()] = upper_ghost;
    const auto ghost = make_valid_ghost_spans_1d(store);
    DataVector filtered(5, 0.0);
    auto fspan = gsl::make_span(filtered.data(), filtered.size());
    const auto vspan = gsl::make_span(vol.data(), vol.size());
    CHECK_THROWS_WITH(
        fd::kreiss_oliger_filter(make_not_null(&fspan), vspan, ghost, mesh_bad,
                                 number_of_vars, 2, epsilon),
        Catch::Matchers::ContainsSubstring(
            "Mesh basis must be FiniteDifference"));
  }
  {
    INFO("Wrong quadrature");
    const Mesh<1> mesh_bad{5, Spectral::Basis::FiniteDifference,
                           Spectral::Quadrature::FaceCentered};
    const DataVector vol(5, 1.0);
    const DataVector lower_ghost(2, 1.0);
    const DataVector upper_ghost(2, 1.0);
    DirectionMap<1, DataVector> store{};
    store[Direction<1>::lower_xi()] = lower_ghost;
    store[Direction<1>::upper_xi()] = upper_ghost;
    const auto ghost = make_valid_ghost_spans_1d(store);
    DataVector filtered(5, 0.0);
    auto fspan = gsl::make_span(filtered.data(), filtered.size());
    const auto vspan = gsl::make_span(vol.data(), vol.size());
    CHECK_THROWS_WITH(
        fd::kreiss_oliger_filter(make_not_null(&fspan), vspan, ghost, mesh_bad,
                                 number_of_vars, 2, epsilon),
        Catch::Matchers::ContainsSubstring(
            "Mesh quadrature must be CellCentered"));
  }
  {
    INFO("volume_vars size mismatch");
    const Mesh<1> mesh{5, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered};
    const DataVector vol(4, 1.0);  // wrong: should be 5
    const DataVector lower_ghost(2, 1.0);
    const DataVector upper_ghost(2, 1.0);
    DirectionMap<1, DataVector> store{};
    store[Direction<1>::lower_xi()] = lower_ghost;
    store[Direction<1>::upper_xi()] = upper_ghost;
    const auto ghost = make_valid_ghost_spans_1d(store);
    DataVector filtered(5, 0.0);
    auto fspan = gsl::make_span(filtered.data(), filtered.size());
    const auto vspan = gsl::make_span(vol.data(), vol.size());
    CHECK_THROWS_WITH(
        fd::kreiss_oliger_filter(make_not_null(&fspan), vspan, ghost, mesh,
                                 number_of_vars, 2, epsilon),
        Catch::Matchers::ContainsSubstring(
            "The size of the volume vars must be the number of points"));
  }
  {
    INFO("filtered_data size mismatch");
    const Mesh<1> mesh{5, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered};
    const DataVector vol(5, 1.0);
    const DataVector lower_ghost(2, 1.0);
    const DataVector upper_ghost(2, 1.0);
    DirectionMap<1, DataVector> store{};
    store[Direction<1>::lower_xi()] = lower_ghost;
    store[Direction<1>::upper_xi()] = upper_ghost;
    const auto ghost = make_valid_ghost_spans_1d(store);
    DataVector filtered(4, 0.0);  // wrong: should be 5
    auto fspan = gsl::make_span(filtered.data(), filtered.size());
    const auto vspan = gsl::make_span(vol.data(), vol.size());
    CHECK_THROWS_WITH(
        fd::kreiss_oliger_filter(make_not_null(&fspan), vspan, ghost, mesh,
                                 number_of_vars, 2, epsilon),
        Catch::Matchers::ContainsSubstring("The filtered data must have size"));
  }
  {
    INFO("Volume extent too small for stencil (filter_fastest_dim assert)");
    const Mesh<1> mesh{1, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered};
    const DataVector vol(1, 1.0);
    const DataVector lower_ghost(2, 1.0);
    const DataVector upper_ghost(2, 1.0);
    DirectionMap<1, DataVector> store{};
    store[Direction<1>::lower_xi()] = lower_ghost;
    store[Direction<1>::upper_xi()] = upper_ghost;
    const auto ghost = make_valid_ghost_spans_1d(store);
    DataVector filtered(1, 0.0);
    auto fspan = gsl::make_span(filtered.data(), filtered.size());
    const auto vspan = gsl::make_span(vol.data(), vol.size());
    CHECK_THROWS_WITH(
        fd::kreiss_oliger_filter(make_not_null(&fspan), vspan, ghost, mesh,
                                 number_of_vars, 2, epsilon),
        Catch::Matchers::ContainsSubstring("Subcell volume extent"));
  }
  {
    INFO("Lower ghost data size not a multiple of number_of_stripes");
    // Use 2 variables so number_of_stripes = 2; ghost size 3 is not divisible
    // by 2, triggering the assert before any other issue.
    const Mesh<1> mesh{5, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered};
    const DataVector vol(10, 1.0);         // 5 pts * 2 vars
    const DataVector lower_ghost(3, 1.0);  // not divisible by stripes=2
    const DataVector upper_ghost(3, 1.0);
    DirectionMap<1, DataVector> store{};
    store[Direction<1>::lower_xi()] = lower_ghost;
    store[Direction<1>::upper_xi()] = upper_ghost;
    const auto ghost = make_valid_ghost_spans_1d(store);
    DataVector filtered(10, 0.0);
    auto fspan = gsl::make_span(filtered.data(), filtered.size());
    const auto vspan = gsl::make_span(vol.data(), vol.size());
    CHECK_THROWS_WITH(
        fd::kreiss_oliger_filter(make_not_null(&fspan), vspan, ghost, mesh,
                                 /*number_of_variables=*/2, 2, epsilon),
        Catch::Matchers::ContainsSubstring(
            "The lower ghost data must be a multiple of the number of "
            "stripes"));
  }
  {
    INFO("Lower/upper ghost sizes differ");
    const Mesh<1> mesh{5, Spectral::Basis::FiniteDifference,
                       Spectral::Quadrature::CellCentered};
    const DataVector vol(5, 1.0);
    const DataVector lower_ghost(2, 1.0);
    const DataVector upper_ghost(4, 1.0);  // different size
    DirectionMap<1, DataVector> store{};
    store[Direction<1>::lower_xi()] = lower_ghost;
    store[Direction<1>::upper_xi()] = upper_ghost;
    const auto ghost = make_valid_ghost_spans_1d(store);
    DataVector filtered(5, 0.0);
    auto fspan = gsl::make_span(filtered.data(), filtered.size());
    const auto vspan = gsl::make_span(vol.data(), vol.size());
    CHECK_THROWS_WITH(
        fd::kreiss_oliger_filter(make_not_null(&fspan), vspan, ghost, mesh,
                                 number_of_vars, 2, epsilon),
        Catch::Matchers::ContainsSubstring("The lower ghost data size"));
  }
  {
    INFO("Non-isotropic 3D mesh (all FD dims)");
    const Mesh<3> mesh_aniso{{5, 7, 5},
                             Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};
    const DataVector vol(5_st * 7_st * 5_st, 1.0);
    // Provide ghost data for all directions with any size; assert fires before
    // accessing it.
    const DataVector ghost_data(1, 1.0);
    DirectionMap<3, DataVector> store3{};
    for (const auto& dir : Direction<3>::all_directions()) {
      store3[dir] = ghost_data;
    }
    DirectionMap<3, gsl::span<const double>> ghost3{};
    for (const auto& [dir, data] : store3) {
      ghost3[dir] = gsl::make_span(data.data(), data.size());
    }
    DataVector filtered(5_st * 7_st * 5_st, 0.0);
    auto fspan = gsl::make_span(filtered.data(), filtered.size());
    const auto vspan = gsl::make_span(vol.data(), vol.size());
    CHECK_THROWS_WITH(
        fd::kreiss_oliger_filter(make_not_null(&fspan), vspan, ghost3,
                                 mesh_aniso, number_of_vars, 2, epsilon),
        Catch::Matchers::ContainsSubstring("The mesh must be isotropic"));
  }
#endif  // SPECTRE_DEBUG
}

SPECTRE_TEST_CASE("Unit.FiniteDifference.Filter",
                  "[Unit][NumericalAlgorithms]") {
  test_ko_dissipation<1>();
  test_ko_dissipation<2>();
  test_ko_dissipation<3>();

  test_low_pass_filter<1>();
  test_low_pass_filter<2>();
  test_low_pass_filter<3>();

  test_cartoon_filter<false>();
  test_cartoon_filter<true>();

  test_asserts();
}
}  // namespace
