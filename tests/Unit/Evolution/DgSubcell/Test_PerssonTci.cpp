// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace Tags {
struct Scalar : db::SimpleTag {
  using type = ::Scalar<DataVector>;
};

template <size_t Dim>
struct Vector : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
};
}  // namespace Tags

template <size_t Dim>
DataVector soln(const tnsr::I<DataVector, Dim, Frame::ElementLogical>& coords,
                const size_t number_of_modes_per_dim,
                const std::array<double, Dim>& highest_coeffs,
                const std::array<double, Dim>& second_highest_coeffs) {
  DataVector result =
      Spectral::compute_basis_function_value<Spectral::Basis::Legendre>(
          1, get<0>(coords));
  for (size_t d = 1; d < Dim; ++d) {
    result += Spectral::compute_basis_function_value<Spectral::Basis::Legendre>(
        1, coords.get(d));
  }

  for (size_t d = 0; d < Dim; ++d) {
    result += gsl::at(highest_coeffs, d) *
              Spectral::compute_basis_function_value<Spectral::Basis::Legendre>(
                  number_of_modes_per_dim - 1, coords.get(d));
    result += gsl::at(second_highest_coeffs, d) *
              Spectral::compute_basis_function_value<Spectral::Basis::Legendre>(
                  number_of_modes_per_dim - 2, coords.get(d));
  }
  return result;
}

template <size_t Dim, typename TagToCheck>
void test_persson_impl(
    const size_t num_pts_1d,
    const std::array<double, Dim>& oscillatory_highest_coeffs,
    const std::array<double, Dim>& oscillatory_second_highest_coeffs,
    const size_t tensor_component_to_modify, const double persson_exponent,
    const size_t persson_number_of_highest_modes,
    const bool expected_tci_triggered) {
  CAPTURE(Dim);
  CAPTURE(db::tag_name<TagToCheck>());
  CAPTURE(num_pts_1d);
  CAPTURE(oscillatory_highest_coeffs);
  CAPTURE(oscillatory_second_highest_coeffs);
  CAPTURE(tensor_component_to_modify);
  CAPTURE(persson_exponent);
  CAPTURE(persson_number_of_highest_modes);
  CAPTURE(expected_tci_triggered);
  const Mesh<Dim> dg_mesh{num_pts_1d, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const auto logical_coords = logical_coordinates(dg_mesh);
  const std::array<double, Dim> zero_spectral_coeffs = make_array<Dim>(0.0);

  Variables<tmpl::list<Tags::Scalar, Tags::Vector<Dim>>> vars(
      dg_mesh.number_of_grid_points());
  if (tensor_component_to_modify == 0) {
    get(get<Tags::Scalar>(vars)) =
        soln(logical_coords, dg_mesh.extents(0), oscillatory_highest_coeffs,
             oscillatory_second_highest_coeffs);
  } else {
    get(get<Tags::Scalar>(vars)) =
        soln(logical_coords, dg_mesh.extents(0), zero_spectral_coeffs,
             zero_spectral_coeffs);
  }
  for (size_t d = 0; d < Dim; ++d) {
    if (tensor_component_to_modify == d + 1) {
      get<Tags::Vector<Dim>>(vars).get(d) =
          (static_cast<double>(d) + 0.3) *
          soln(logical_coords, dg_mesh.extents(0), oscillatory_highest_coeffs,
               oscillatory_second_highest_coeffs);
    } else {
      get<Tags::Vector<Dim>>(vars).get(d) =
          (static_cast<double>(d) + 0.3) *
          soln(logical_coords, dg_mesh.extents(0), zero_spectral_coeffs,
               zero_spectral_coeffs);
    }
  }

  CHECK(evolution::dg::subcell::persson_tci(
            get<TagToCheck>(vars), dg_mesh, persson_exponent,
            persson_number_of_highest_modes) == expected_tci_triggered);
}

template <size_t Dim>
void test_persson() {
  const auto zero_spectral_coeffs = make_array<Dim>(0.0);
  // We lower the maximum number of 1d points in 3d in order to reduce total
  // test runtime.
  const size_t maximum_number_of_points_1d =
      Dim == 3 ? 7
               : Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
  for (const double persson_exponent : {1.0, 10.0}) {
    for (size_t num_pts_1d = 4; num_pts_1d < maximum_number_of_points_1d;
         ++num_pts_1d) {
      // Test all coeffs set to zero.
      for (const size_t num_highest_modes : {1_st, 2_st}) {
        test_persson_impl<Dim, Tags::Scalar>(
            num_pts_1d, zero_spectral_coeffs, zero_spectral_coeffs, 0,
            persson_exponent, num_highest_modes, false);
        test_persson_impl<Dim, Tags::Vector<Dim>>(
            num_pts_1d, zero_spectral_coeffs, zero_spectral_coeffs, 0,
            persson_exponent, num_highest_modes, false);
      }

      // Test only the highest coeffs are zero but the second highest mode is
      // excited.
      for (size_t i = 0; i < Dim; ++i) {
        std::array<double, Dim> second_highest_coeffs = make_array<Dim>(0.0);
        gsl::at(second_highest_coeffs, i) = 0.05;

        for (const size_t num_highest_modes : {1_st, 2_st}) {
          // If the TCI is only looking at the highest mode, it should not
          // trigger.
          const bool should_trigger =
              (persson_exponent == 10.0) and (num_highest_modes == 2_st);

          // Test Scalar
          test_persson_impl<Dim, Tags::Scalar>(
              num_pts_1d, zero_spectral_coeffs, second_highest_coeffs, 0,
              persson_exponent, num_highest_modes, should_trigger);
          // Test no trigger if different tags are oscillatory
          for (size_t j = 0; j < Dim; ++j) {
            test_persson_impl<Dim, Tags::Scalar>(
                num_pts_1d, zero_spectral_coeffs, second_highest_coeffs, j + 1,
                persson_exponent, num_highest_modes, false);
          }
          // Test Vector
          test_persson_impl<Dim, Tags::Vector<Dim>>(
              num_pts_1d, zero_spectral_coeffs, second_highest_coeffs, 0,
              persson_exponent, num_highest_modes, false);
          for (size_t j = 0; j < Dim; ++j) {
            test_persson_impl<Dim, Tags::Vector<Dim>>(
                num_pts_1d, zero_spectral_coeffs, second_highest_coeffs, j + 1,
                persson_exponent, num_highest_modes, should_trigger);
          }
        }
      }

      // Test highest coeffs are not zero.
      for (size_t i = 0; i < Dim; ++i) {
        std::array<double, Dim> highest_coeffs = make_array<Dim>(0.0);
        gsl::at(highest_coeffs, i) = 0.05;

        for (const size_t num_highest_modes : {1_st, 2_st}) {
          const bool should_trigger = persson_exponent == 10.0;

          // Test Scalar
          test_persson_impl<Dim, Tags::Scalar>(
              num_pts_1d, highest_coeffs, zero_spectral_coeffs, 0,
              persson_exponent, num_highest_modes, should_trigger);
          // Test no trigger if different tags are oscillatory
          for (size_t j = 0; j < Dim; ++j) {
            test_persson_impl<Dim, Tags::Scalar>(
                num_pts_1d, highest_coeffs, zero_spectral_coeffs, j + 1,
                persson_exponent, num_highest_modes, false);
          }
          // Test Vector
          test_persson_impl<Dim, Tags::Vector<Dim>>(
              num_pts_1d, highest_coeffs, zero_spectral_coeffs, 0,
              persson_exponent, num_highest_modes, false);
          for (size_t j = 0; j < Dim; ++j) {
            test_persson_impl<Dim, Tags::Vector<Dim>>(
                num_pts_1d, highest_coeffs, zero_spectral_coeffs, j + 1,
                persson_exponent, num_highest_modes, should_trigger);
          }
        }
      }
    }
  }
}

// [[TimeOut, 20]]
SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Tci.Persson", "[Evolution][Unit]") {
  test_persson<1>();
  test_persson<2>();
  test_persson<3>();
}
}  // namespace
