// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <tuple>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/Spectral/DiskTestFunctions.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/FilteringB2.hpp"
#include "NumericalAlgorithms/Spectral/FilteringB2.tpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace {
void test_disk_filter() {
  using TagsList = tmpl::list<::Tags::TempScalar<0>, ::Tags::Tempi<0, 2>>;

  // Test over several mesh sizes satisfying the constraints:
  //   n_ph odd,  M = n_ph/2 <= n_r_max = 2*n_r - 2
  const std::vector<std::pair<size_t, size_t>> mesh_sizes{
      {2, 5}, {3, 3}, {3, 5}, {3, 9}, {4, 5}, {5, 5}, {5, 7}, {6, 7},
  };

  for (const auto& [n_r, n_ph] : mesh_sizes) {
    CAPTURE(n_r);
    CAPTURE(n_ph);
    const Mesh<2> mesh{{n_r, n_ph},
                       {Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2},
                       {Spectral::Quadrature::GaussRadauUpper,
                        Spectral::Quadrature::Equiangular}};
    const auto x = logical_coordinates(mesh);
    const DataVector r = 0.5 * (x[0] + 1.0);
    const DataVector& phi = x[1];
    const size_t num_grid_points = mesh.number_of_grid_points();

    // Use the lowest-degree modes that are non-trivial and exactly
    // representable on every test mesh:
    //   r*cos(phi) — Zernike (n=1, m=1) cosine mode
    //   r*sin(phi) — Zernike (n=1, m=1) sine mode
    // Both have Fourier order m=1 <= M for all meshes (M >= 1).
    const size_t M = n_ph / 2;
    const DiskTestFunctions::ProductOfPolynomials f{1, 0};
    const DataVector f_vals = f(r, phi);
    const DataVector f_sin_vals = r * sin(phi);

    Variables<TagsList> u{num_grid_points};
    Variables<TagsList> expected_result{num_grid_points};

    // Test alpha=0 is the identity
    {
      get(get<::Tags::TempScalar<0>>(u)) = f_vals;
      get<0>(get<::Tags::Tempi<0, 2>>(u)) = f_vals;
      get<1>(get<::Tags::Tempi<0, 2>>(u)) = f_sin_vals;
      get(get<::Tags::TempScalar<0>>(expected_result)) = f_vals;
      get<0>(get<::Tags::Tempi<0, 2>>(expected_result)) = f_vals;
      get<1>(get<::Tags::Tempi<0, 2>>(expected_result)) = f_sin_vals;

      Spectral::filtering::zernike_b2_disk_exponential_filter(make_not_null(&u),
                                                              mesh, 0.0, 2);

      CHECK_VARIABLES_APPROX(u, expected_result);
    }

    // Test constant function is unaffected by any filter.
    // f=1 is purely the (n=0, m=0) mode; filter factor = exp(0)=1 always.
    {
      get(get<::Tags::TempScalar<0>>(u)) = 1.0;
      get<0>(get<::Tags::Tempi<0, 2>>(u)) = 1.0;
      get<1>(get<::Tags::Tempi<0, 2>>(u)) = 1.0;
      get(get<::Tags::TempScalar<0>>(expected_result)) = 1.0;
      get<0>(get<::Tags::Tempi<0, 2>>(expected_result)) = 1.0;
      get<1>(get<::Tags::Tempi<0, 2>>(expected_result)) = 1.0;

      Spectral::filtering::zernike_b2_disk_exponential_filter(make_not_null(&u),
                                                              mesh, 36.0, 32);

      CHECK_VARIABLES_APPROX(u, expected_result);
    }

    // Test the top Fourier mode r^M*cos/sin(M*phi) is killed by heavy filtering
    {
      const DataVector f_top_cos =
          pow(r, static_cast<double>(M)) * cos(static_cast<double>(M) * phi);
      const DataVector f_top_sin =
          pow(r, static_cast<double>(M)) * sin(static_cast<double>(M) * phi);
      get(get<::Tags::TempScalar<0>>(u)) = f_top_cos;
      get<0>(get<::Tags::Tempi<0, 2>>(u)) = f_top_cos;
      get<1>(get<::Tags::Tempi<0, 2>>(u)) = f_top_sin;
      get(get<::Tags::TempScalar<0>>(expected_result)) = 0.;
      get<0>(get<::Tags::Tempi<0, 2>>(expected_result)) = 0.;
      get<1>(get<::Tags::Tempi<0, 2>>(expected_result)) = 0.;

      Spectral::filtering::zernike_b2_disk_exponential_filter(make_not_null(&u),
                                                              mesh, 36.0, 32);

      CHECK_VARIABLES_APPROX(u, expected_result);
    }

    // Test constant + top Fourier mode: constant survives, top disappears.
    {
      const DataVector f_mixed = 1.0 + pow(r, static_cast<double>(M)) *
                                           cos(static_cast<double>(M) * phi);
      get(get<::Tags::TempScalar<0>>(u)) = f_mixed;
      get<0>(get<::Tags::Tempi<0, 2>>(u)) = f_mixed;
      get<1>(get<::Tags::Tempi<0, 2>>(u)) = f_mixed;
      get(get<::Tags::TempScalar<0>>(expected_result)) = 1.;
      get<0>(get<::Tags::Tempi<0, 2>>(expected_result)) = 1.;
      get<1>(get<::Tags::Tempi<0, 2>>(expected_result)) = 1.;

      Spectral::filtering::zernike_b2_disk_exponential_filter(make_not_null(&u),
                                                              mesh, 36.0, 32);

      CHECK_VARIABLES_APPROX(u, expected_result);
    }
  }
}

// The nodal function r^m*{cos,sin}(m*phi) is a pure single mode, so applying
// the filter must return the same function scaled by exactly the expected
// weight.
// The filter multiplies the (n, m) ZernikeB2 by Fourier spectral coefficient by
//   exp(-alpha*(ns/n_order)^(2p)) * exp(-alpha*(m/M)^(2p)),
// where ns = (n+1)/2 (integer division).
void test_disk_filter_weights() {
  using TagsList = tmpl::list<::Tags::TempScalar<0>>;

  const std::vector<std::pair<double, unsigned>> params{
      {10.0, 2}, {20.0, 4}, {36.0, 8}};
  const std::vector<std::pair<size_t, size_t>> mesh_sizes{
      {2, 3}, {2, 5}, {3, 5}, {3, 9}, {4, 5}, {5, 7}, {6, 7}};

  for (const auto& [n_r, n_ph] : mesh_sizes) {
    CAPTURE(n_r);
    CAPTURE(n_ph);
    const Mesh<2> mesh{{n_r, n_ph},
                       {Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2},
                       {Spectral::Quadrature::GaussRadauUpper,
                        Spectral::Quadrature::Equiangular}};
    const auto x = logical_coordinates(mesh);
    const DataVector r = 0.5 * (x[0] + 1.0);
    const DataVector& phi = x[1];
    const size_t num_grid_points = mesh.number_of_grid_points();
    const size_t M = n_ph / 2;
    const size_t n_order = n_r - 1;

    for (const auto& [alpha, half_power] : params) {
      CAPTURE(alpha);
      CAPTURE(half_power);

      for (size_t m = 1; m <= M; ++m) {
        CAPTURE(m);
        const size_t ns = (m + 1) / 2;
        const double expected_factor =
            exp(-alpha *
                pow(static_cast<double>(ns) / static_cast<double>(n_order),
                    2 * half_power)) *
            exp(-alpha * pow(static_cast<double>(m) / static_cast<double>(M),
                             2 * half_power));

        const DataVector r_m = pow(r, static_cast<double>(m));
        Variables<TagsList> u{num_grid_points};
        Variables<TagsList> expected_result{num_grid_points};

        // Cosine mode: r^m * cos(m*phi)
        {
          const DataVector f_cos = r_m * cos(static_cast<double>(m) * phi);
          get(get<::Tags::TempScalar<0>>(u)) = f_cos;
          Spectral::filtering::zernike_b2_disk_exponential_filter(
              make_not_null(&u), mesh, alpha, half_power);
          get(get<::Tags::TempScalar<0>>(expected_result)) =
              expected_factor * f_cos;
          CHECK_VARIABLES_APPROX(u, expected_result);
        }

        // Sine mode: r^m * sin(m*phi)
        {
          const DataVector f_sin = r_m * sin(static_cast<double>(m) * phi);
          get(get<::Tags::TempScalar<0>>(u)) = f_sin;
          Spectral::filtering::zernike_b2_disk_exponential_filter(
              make_not_null(&u), mesh, alpha, half_power);
          get(get<::Tags::TempScalar<0>>(expected_result)) =
              expected_factor * f_sin;
          CHECK_VARIABLES_APPROX(u, expected_result);
        }
      }
    }
  }
}

void test_cylinder_filter() {
  using TagsList = tmpl::list<::Tags::TempScalar<0>, ::Tags::Tempi<0, 2>>;

  // Mesh sizes: {n_r, n_ph, n_z}. Same (n_r, n_ph) constraints as disk;
  // n_z >= 2 so the z filter has something to do.
  const std::vector<std::tuple<size_t, size_t, size_t>> mesh_sizes{
      {2, 5, 2}, {3, 3, 3}, {3, 5, 2}, {3, 5, 4}, {4, 5, 3}, {5, 7, 3},
  };

  for (const auto& [n_r, n_ph, n_z] : mesh_sizes) {
    CAPTURE(n_r);
    CAPTURE(n_ph);
    CAPTURE(n_z);
    const Mesh<3> mesh{{n_r, n_ph, n_z},
                       {Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2,
                        Spectral::Basis::Legendre},
                       {Spectral::Quadrature::GaussRadauUpper,
                        Spectral::Quadrature::Equiangular,
                        Spectral::Quadrature::GaussLobatto}};
    const auto x = logical_coordinates(mesh);
    const DataVector r = 0.5 * (x[0] + 1.0);
    const DataVector& phi = x[1];
    const DataVector& z_logical = x[2];
    const size_t num_grid_points = mesh.number_of_grid_points();
    const size_t M = n_ph / 2;

    Variables<TagsList> u{num_grid_points};
    Variables<TagsList> expected_result{num_grid_points};

    // Test alpha=0 is the identity.
    {
      const DiskTestFunctions::ProductOfPolynomials f{1, 0};
      const DataVector f_vals = f(r, phi) * (1.0 + z_logical);
      get(get<::Tags::TempScalar<0>>(u)) = f_vals;
      get<0>(get<::Tags::Tempi<0, 2>>(u)) = f_vals;
      get<1>(get<::Tags::Tempi<0, 2>>(u)) = f_vals;
      get(get<::Tags::TempScalar<0>>(expected_result)) = f_vals;
      get<0>(get<::Tags::Tempi<0, 2>>(expected_result)) = f_vals;
      get<1>(get<::Tags::Tempi<0, 2>>(expected_result)) = f_vals;

      Spectral::filtering::zernike_b2_cylinder_exponential_filter(
          make_not_null(&u), mesh, 0.0, 2);

      CHECK_VARIABLES_APPROX(u, expected_result);
    }

    // Test constant function is unaffected by any filter.
    // f=1 is the (n=0, m=0, k_z=0) mode; all filter factors = exp(0) = 1.
    {
      get(get<::Tags::TempScalar<0>>(u)) = 1.0;
      get<0>(get<::Tags::Tempi<0, 2>>(u)) = 1.0;
      get<1>(get<::Tags::Tempi<0, 2>>(u)) = 1.0;
      get(get<::Tags::TempScalar<0>>(expected_result)) = 1.0;
      get<0>(get<::Tags::Tempi<0, 2>>(expected_result)) = 1.0;
      get<1>(get<::Tags::Tempi<0, 2>>(expected_result)) = 1.0;

      Spectral::filtering::zernike_b2_cylinder_exponential_filter(
          make_not_null(&u), mesh, 36.0, 32);

      CHECK_VARIABLES_APPROX(u, expected_result);
    }

    // Test the top disk mode r^M*cos/sin(M*phi) is killed by heavy
    // filtering regardless of z variation.
    {
      const DataVector f_top_cos =
          pow(r, static_cast<double>(M)) * cos(static_cast<double>(M) * phi);
      const DataVector f_top_sin =
          pow(r, static_cast<double>(M)) * sin(static_cast<double>(M) * phi);
      get(get<::Tags::TempScalar<0>>(u)) = f_top_cos;
      get<0>(get<::Tags::Tempi<0, 2>>(u)) = f_top_cos;
      get<1>(get<::Tags::Tempi<0, 2>>(u)) = f_top_sin;
      get(get<::Tags::TempScalar<0>>(expected_result)) = 0.;
      get<0>(get<::Tags::Tempi<0, 2>>(expected_result)) = 0.;
      get<1>(get<::Tags::Tempi<0, 2>>(expected_result)) = 0.;

      Spectral::filtering::zernike_b2_cylinder_exponential_filter(
          make_not_null(&u), mesh, 36.0, 32);

      CHECK_VARIABLES_APPROX(u, expected_result);
    }

    // Test constant + top disk mode: constant survives, top disappears.
    {
      const DataVector f_mixed = 1.0 + pow(r, static_cast<double>(M)) *
                                           cos(static_cast<double>(M) * phi);
      get(get<::Tags::TempScalar<0>>(u)) = f_mixed;
      get<0>(get<::Tags::Tempi<0, 2>>(u)) = f_mixed;
      get<1>(get<::Tags::Tempi<0, 2>>(u)) = f_mixed;
      get(get<::Tags::TempScalar<0>>(expected_result)) = 1.;
      get<0>(get<::Tags::Tempi<0, 2>>(expected_result)) = 1.;
      get<1>(get<::Tags::Tempi<0, 2>>(expected_result)) = 1.;

      Spectral::filtering::zernike_b2_cylinder_exponential_filter(
          make_not_null(&u), mesh, 36.0, 32);

      CHECK_VARIABLES_APPROX(u, expected_result);
    }

    // Test 4: the top z mode is killed by heavy filtering. We use a function
    // that is purely that top Legendre mode in z and constant in (r, phi): the
    // nodal function with a non-trivial z-profile that maps entirely onto the
    // highest-order Legendre coefficient.
    // Rather than constructing analytically, we verify that a constant- in-z
    // function (which has no high-z-mode content) is preserved, while a
    // function with only high-z-mode content vanishes.
    const Matrix& modal_to_nodal_z =
        Spectral::modal_to_nodal_matrix(mesh.slice_through(2));
    DataVector f_top_z_vals(num_grid_points, 0.0);
    const size_t n_rph = n_r * n_ph;
    for (size_t k = 0; k < n_z; ++k) {
      const double z_nodal_val = modal_to_nodal_z(k, n_z - 1);
      for (size_t ij = 0; ij < n_rph; ++ij) {
        f_top_z_vals[ij + n_rph * k] = z_nodal_val;
      }
    }
    get(get<::Tags::TempScalar<0>>(u)) = f_top_z_vals;
    get<0>(get<::Tags::Tempi<0, 2>>(u)) = f_top_z_vals;
    get<1>(get<::Tags::Tempi<0, 2>>(u)) = f_top_z_vals;
    get(get<::Tags::TempScalar<0>>(expected_result)) = 0.;
    get<0>(get<::Tags::Tempi<0, 2>>(expected_result)) = 0.;
    get<1>(get<::Tags::Tempi<0, 2>>(expected_result)) = 0.;

    Spectral::filtering::zernike_b2_cylinder_exponential_filter(
        make_not_null(&u), mesh, 36.0, 32);

    CHECK_VARIABLES_APPROX(u, expected_result);
  }
}

// The (r, phi) part is tested the same way as for the disk: apply the filter
// to a pure single-mode function r^m*{cos,sin}(m*phi) (constant in z) and
// verify the result equals the input scaled by the expected (r,phi) weight
// times the z weight for the k=0 mode (which is 1.0 since k_z=0 means
// exp(0)=1 for the z filter).
//
// The z part is tested by applying the filter to a function that is the k-th
// Legendre modal basis function in z (constant in r, phi) and verifying the
// output is scaled by the expected z filter weight
// exp(-alpha*(k/(n_z-1))^(2p)).
void test_cylinder_filter_weights() {
  using TagsList = tmpl::list<::Tags::TempScalar<0>>;

  const std::vector<std::pair<double, unsigned>> params{
      {10.0, 2}, {20.0, 4}, {36.0, 8}};
  const std::vector<std::tuple<size_t, size_t, size_t>> mesh_sizes{
      {2, 3, 2}, {2, 5, 3}, {3, 5, 3}, {3, 9, 3}, {4, 5, 4}, {5, 7, 3}};

  // The radial-angular plane is fixed to ZernikeB2 x Fourier, but the axial z
  // filter uses exponential_filter() in whatever 1D basis the mesh has in z, so
  // both Legendre and Chebyshev must work there.
  const std::array<std::pair<Spectral::Basis, Spectral::Quadrature>, 2> z_bases{
      {{Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto},
       {Spectral::Basis::Chebyshev, Spectral::Quadrature::GaussLobatto}}};

  for (const auto& [z_basis, z_quadrature] : z_bases) {
    CAPTURE(z_basis);
    CAPTURE(z_quadrature);
    for (const auto& [n_r, n_ph, n_z] : mesh_sizes) {
      CAPTURE(n_r);
      CAPTURE(n_ph);
      CAPTURE(n_z);
      const Mesh<3> mesh{
          {n_r, n_ph, n_z},
          {Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2, z_basis},
          {Spectral::Quadrature::GaussRadauUpper,
           Spectral::Quadrature::Equiangular, z_quadrature}};
      const auto x = logical_coordinates(mesh);
      const DataVector r = 0.5 * (x[0] + 1.0);
      const DataVector& phi = x[1];
      const size_t num_grid_points = mesh.number_of_grid_points();
      const size_t M = n_ph / 2;
      const size_t n_order = n_r - 1;
      const size_t n_rph = n_r * n_ph;

      for (const auto& [alpha, half_power] : params) {
        CAPTURE(alpha);
        CAPTURE(half_power);

        // (r, phi) weight test: r^m*{cos,sin}(m*phi) constant in z.
        // Since the function is constant in z it has only the k_z=0 mode, whose
        // z filter factor is exp(0)=1. So the full weight is purely the disk
        // weight.
        for (size_t m = 1; m <= M; ++m) {
          CAPTURE(m);
          const size_t ns = (m + 1) / 2;
          const double disk_factor =
              exp(-alpha *
                  pow(static_cast<double>(ns) / static_cast<double>(n_order),
                      2 * half_power)) *
              exp(-alpha * pow(static_cast<double>(m) / static_cast<double>(M),
                               2 * half_power));

          const DataVector r_m = pow(r, static_cast<double>(m));
          Variables<TagsList> u{num_grid_points};
          Variables<TagsList> expected_result{num_grid_points};

          // Cosine mode: r^m * cos(m*phi), constant in z
          {
            const DataVector f_cos = r_m * cos(static_cast<double>(m) * phi);
            get(get<::Tags::TempScalar<0>>(u)) = f_cos;
            Spectral::filtering::zernike_b2_cylinder_exponential_filter(
                make_not_null(&u), mesh, alpha, half_power);
            get(get<::Tags::TempScalar<0>>(expected_result)) =
                disk_factor * f_cos;
            CHECK_VARIABLES_APPROX(u, expected_result);
          }

          // Sine mode: r^m * sin(m*phi), constant in z
          {
            const DataVector f_sin = r_m * sin(static_cast<double>(m) * phi);
            get(get<::Tags::TempScalar<0>>(u)) = f_sin;
            Spectral::filtering::zernike_b2_cylinder_exponential_filter(
                make_not_null(&u), mesh, alpha, half_power);
            get(get<::Tags::TempScalar<0>>(expected_result)) =
                disk_factor * f_sin;
            CHECK_VARIABLES_APPROX(u, expected_result);
          }
        }

        // z weight test: the k-th modal basis function in z (Legendre or
        // Chebyshev, see z_bases above), constant in (r, phi). The nodal
        // representation is column k of the modal-to-nodal matrix for the z
        // mesh, which is built generically from the mesh's z basis, so the same
        // check exercises both bases. Since the function is constant
        // in (r, phi) it sits entirely in the (n=0, m=0) disk mode whose disk
        // filter weight is exp(0)=1. So the full weight is the z weight alone:
        //   exp(-alpha * (k / (n_z-1))^(2p)).
        const Matrix& modal_to_nodal_z =
            Spectral::modal_to_nodal_matrix(mesh.slice_through(2));
        const auto n_z_order = static_cast<double>(n_z - 1);

        for (size_t k = 0; k < n_z; ++k) {
          CAPTURE(k);
          const double z_factor = exp(
              -alpha * pow(static_cast<double>(k) / n_z_order, 2 * half_power));

          DataVector f_z_vals(num_grid_points, 0.0);
          for (size_t kk = 0; kk < n_z; ++kk) {
            const double z_nodal_val = modal_to_nodal_z(kk, k);
            for (size_t ij = 0; ij < n_rph; ++ij) {
              f_z_vals[ij + n_rph * kk] = z_nodal_val;
            }
          }

          Variables<TagsList> u{num_grid_points};
          Variables<TagsList> expected_result{num_grid_points};
          get(get<::Tags::TempScalar<0>>(u)) = f_z_vals;
          Spectral::filtering::zernike_b2_cylinder_exponential_filter(
              make_not_null(&u), mesh, alpha, half_power);
          get(get<::Tags::TempScalar<0>>(expected_result)) =
              z_factor * f_z_vals;
          CHECK_VARIABLES_APPROX(u, expected_result);
        }
      }
    }
  }
}

// Exercises the generalized `zernike_b2_{disk,cylinder}_filter` entry points:
// independent disk/z half-powers, the top angular mode cutoff, identity when
// nothing is requested, and a regression check that the `*_exponential_filter`
// wrappers reproduce the generalized call.
void test_generalized_filters() {
  using TagsList = tmpl::list<::Tags::TempScalar<0>>;

  const Mesh<3> mesh{
      {5, 7, 4},
      {Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2,
       Spectral::Basis::Legendre},
      {Spectral::Quadrature::GaussRadauUpper, Spectral::Quadrature::Equiangular,
       Spectral::Quadrature::GaussLobatto}};
  const Mesh<2> disk = mesh.slice_away(2);
  const auto x = logical_coordinates(mesh);
  const DataVector r = 0.5 * (x[0] + 1.0);
  const DataVector& phi = x[1];
  const size_t num_grid_points = mesh.number_of_grid_points();
  const size_t M = 7 / 2;
  const size_t n_order = 5 - 1;
  const size_t n_rph = 5_st * 7_st;
  const size_t n_z = 4;

  DataVector smooth_field(num_grid_points);
  for (size_t i = 0; i < num_grid_points; ++i) {
    smooth_field[i] = sin(0.3 * static_cast<double>(i)) + 1.1;
  }

  {
    INFO("Regression: cylinder exponential wrapper == generalized call");
    const double alpha = 36.0;
    const unsigned half_power = 8;
    Variables<TagsList> wrapper{num_grid_points};
    Variables<TagsList> general{num_grid_points};
    get(get<::Tags::TempScalar<0>>(wrapper)) = smooth_field;
    get(get<::Tags::TempScalar<0>>(general)) = smooth_field;
    Spectral::filtering::zernike_b2_cylinder_exponential_filter(
        make_not_null(&wrapper), mesh, alpha, half_power);
    Spectral::filtering::zernike_b2_cylinder_filter(
        make_not_null(&general), mesh, alpha,
        std::optional<unsigned>{half_power},
        std::optional<unsigned>{half_power}, 0);
    CHECK_VARIABLES_APPROX(wrapper, general);
  }
  {
    INFO("Regression: disk exponential wrapper == generalized call");
    const DataVector disk_data(smooth_field.data(),
                               disk.number_of_grid_points());
    const double alpha = 36.0;
    const unsigned half_power = 8;
    Variables<TagsList> wrapper{disk.number_of_grid_points()};
    Variables<TagsList> general{disk.number_of_grid_points()};
    get(get<::Tags::TempScalar<0>>(wrapper)) = disk_data;
    get(get<::Tags::TempScalar<0>>(general)) = disk_data;
    Spectral::filtering::zernike_b2_disk_exponential_filter(
        make_not_null(&wrapper), disk, alpha, half_power);
    Spectral::filtering::zernike_b2_disk_filter(
        make_not_null(&general), disk, alpha,
        std::optional<unsigned>{half_power}, 0);
    CHECK_VARIABLES_APPROX(wrapper, general);
  }
  {
    INFO("Identity: no half-powers and no cutoff leaves the data unchanged");
    Variables<TagsList> u{num_grid_points};
    Variables<TagsList> expected_result{num_grid_points};
    get(get<::Tags::TempScalar<0>>(u)) = smooth_field;
    get(get<::Tags::TempScalar<0>>(expected_result)) = smooth_field;
    Spectral::filtering::zernike_b2_cylinder_filter(
        make_not_null(&u), mesh, 36.0, std::nullopt, std::nullopt, 0);
    CHECK_VARIABLES_APPROX(u, expected_result);
  }
  {
    INFO("Disk half-power is independent of the (constant-in-z) z direction");
    // A pure (r, phi) mode that is constant in z must be scaled only by the
    // disk weight, regardless of the z half-power.
    const double alpha = 20.0;
    const unsigned disk_half = 4;
    for (size_t m = 1; m <= M; ++m) {
      CAPTURE(m);
      const size_t ns = (m + 1) / 2;
      const double disk_factor =
          exp(-alpha *
              pow(static_cast<double>(ns) / static_cast<double>(n_order),
                  2 * disk_half)) *
          exp(-alpha * pow(static_cast<double>(m) / static_cast<double>(M),
                           2 * disk_half));
      const DataVector f =
          pow(r, static_cast<double>(m)) * cos(static_cast<double>(m) * phi);
      Variables<TagsList> expected_result{num_grid_points};
      get(get<::Tags::TempScalar<0>>(expected_result)) = disk_factor * f;
      // z_half_power = None and z_half_power = 6 must give the same result.
      for (const std::optional<unsigned> z_half :
           {std::optional<unsigned>{std::nullopt},
            std::optional<unsigned>{6}}) {
        Variables<TagsList> u{num_grid_points};
        get(get<::Tags::TempScalar<0>>(u)) = f;
        Spectral::filtering::zernike_b2_cylinder_filter(
            make_not_null(&u), mesh, alpha, std::optional<unsigned>{disk_half},
            z_half, 0);
        CHECK_VARIABLES_APPROX(u, expected_result);
      }
    }
  }
  {
    INFO("Z half-power acts on z modes with the disk left unfiltered");
    // A pure top-z Legendre mode, constant in (r, phi), is scaled only by the
    // z weight when the disk half-power is None.
    const double alpha = 36.0;
    const unsigned z_half = 8;
    const Matrix& modal_to_nodal_z =
        Spectral::modal_to_nodal_matrix(mesh.slice_through(2));
    const auto n_z_order = static_cast<double>(n_z - 1);
    for (size_t k = 0; k < n_z; ++k) {
      CAPTURE(k);
      const double z_factor =
          exp(-alpha * pow(static_cast<double>(k) / n_z_order, 2 * z_half));
      DataVector f_z_vals(num_grid_points, 0.0);
      for (size_t kk = 0; kk < n_z; ++kk) {
        const double z_nodal_val = modal_to_nodal_z(kk, k);
        for (size_t ij = 0; ij < n_rph; ++ij) {
          f_z_vals[ij + n_rph * kk] = z_nodal_val;
        }
      }
      Variables<TagsList> u{num_grid_points};
      Variables<TagsList> expected_result{num_grid_points};
      get(get<::Tags::TempScalar<0>>(u)) = f_z_vals;
      Spectral::filtering::zernike_b2_cylinder_filter(
          make_not_null(&u), mesh, alpha, std::nullopt,
          std::optional<unsigned>{z_half}, 0);
      get(get<::Tags::TempScalar<0>>(expected_result)) = z_factor * f_z_vals;
      CHECK_VARIABLES_APPROX(u, expected_result);
    }
  }
  {
    INFO("NumModesToKill zeroes the top angular mode and keeps m = 0");
    // f = 1 + r^M cos(M phi): killing one angular mode removes the top mode
    // exactly while leaving the constant untouched, with no exponential
    // roll-off applied.
    const DataVector f_mixed = 1.0 + pow(r, static_cast<double>(M)) *
                                         cos(static_cast<double>(M) * phi);
    Variables<TagsList> u{num_grid_points};
    Variables<TagsList> expected_result{num_grid_points};
    get(get<::Tags::TempScalar<0>>(u)) = f_mixed;
    get(get<::Tags::TempScalar<0>>(expected_result)) = 1.0;
    Spectral::filtering::zernike_b2_cylinder_filter(
        make_not_null(&u), mesh, 36.0, std::nullopt, std::nullopt, 1);
    CHECK_VARIABLES_APPROX(u, expected_result);

    // A lower angular mode (m = 1 <= M - num_modes_to_kill) is retained
    // exactly when no exponential roll-off is requested.
    const DataVector f_low = r * cos(phi);
    Variables<TagsList> u_low{num_grid_points};
    Variables<TagsList> expected_low{num_grid_points};
    get(get<::Tags::TempScalar<0>>(u_low)) = f_low;
    get(get<::Tags::TempScalar<0>>(expected_low)) = f_low;
    Spectral::filtering::zernike_b2_cylinder_filter(
        make_not_null(&u_low), mesh, 36.0, std::nullopt, std::nullopt, 1);
    CHECK_VARIABLES_APPROX(u_low, expected_low);
  }
  {
    INFO("Disk generalized filter: analytic checks on the 2D entry point");
    // The 2D `zernike_b2_disk_filter` is otherwise only reached through the
    // wrapper regression check above (and indirectly via the cylinder, which
    // reimplements the disk transform), so verify its new branches directly.
    // The disk slice keeps n_r = 5 and n_ph = 7, so M and n_order match the
    // values computed above for the 3D mesh.
    const auto disk_x = logical_coordinates(disk);
    const DataVector disk_r = 0.5 * (disk_x[0] + 1.0);
    const DataVector& disk_phi = disk_x[1];
    const size_t disk_num_grid_points = disk.number_of_grid_points();

    {
      INFO("no half-power and no cutoff is the identity");
      DataVector disk_field(disk_num_grid_points);
      for (size_t i = 0; i < disk_num_grid_points; ++i) {
        disk_field[i] = sin(0.3 * static_cast<double>(i)) + 1.1;
      }
      Variables<TagsList> u{disk_num_grid_points};
      Variables<TagsList> expected_result{disk_num_grid_points};
      get(get<::Tags::TempScalar<0>>(u)) = disk_field;
      get(get<::Tags::TempScalar<0>>(expected_result)) = disk_field;
      Spectral::filtering::zernike_b2_disk_filter(make_not_null(&u), disk, 36.0,
                                                  std::nullopt, 0);
      CHECK_VARIABLES_APPROX(u, expected_result);
    }
    {
      INFO("num_modes_to_kill zeroes the top angular mode and keeps m = 0");
      // f = 1 + r^M cos(M phi): killing one angular mode removes the top mode
      // exactly while leaving the constant untouched, with no roll-off applied.
      const DataVector f_mixed =
          1.0 + pow(disk_r, static_cast<double>(M)) *
                    cos(static_cast<double>(M) * disk_phi);
      Variables<TagsList> u{disk_num_grid_points};
      Variables<TagsList> expected_result{disk_num_grid_points};
      get(get<::Tags::TempScalar<0>>(u)) = f_mixed;
      get(get<::Tags::TempScalar<0>>(expected_result)) = 1.0;
      Spectral::filtering::zernike_b2_disk_filter(make_not_null(&u), disk, 36.0,
                                                  std::nullopt, 1);
      CHECK_VARIABLES_APPROX(u, expected_result);

      // A lower angular mode (m = 1 <= M - num_modes_to_kill) survives exactly.
      const DataVector f_low = disk_r * cos(disk_phi);
      Variables<TagsList> u_low{disk_num_grid_points};
      Variables<TagsList> expected_low{disk_num_grid_points};
      get(get<::Tags::TempScalar<0>>(u_low)) = f_low;
      get(get<::Tags::TempScalar<0>>(expected_low)) = f_low;
      Spectral::filtering::zernike_b2_disk_filter(make_not_null(&u_low), disk,
                                                  36.0, std::nullopt, 1);
      CHECK_VARIABLES_APPROX(u_low, expected_low);
    }
    {
      INFO("exponential roll-off and top-mode cutoff compose");
      // f = 1 + r cos(phi) + r^M cos(M phi): the killed top mode (m = M)
      // vanishes, the constant (m = 0) is untouched, and the surviving m = 1
      // mode is scaled by its disk weight.
      const double alpha = 20.0;
      const unsigned half_power = 4;
      const size_t m_keep = 1;
      const size_t ns = (m_keep + 1) / 2;
      const double disk_factor_m1 =
          exp(-alpha *
              pow(static_cast<double>(ns) / static_cast<double>(n_order),
                  2 * half_power)) *
          exp(-alpha * pow(static_cast<double>(m_keep) / static_cast<double>(M),
                           2 * half_power));
      const DataVector f_low = disk_r * cos(disk_phi);
      const DataVector f_top = pow(disk_r, static_cast<double>(M)) *
                               cos(static_cast<double>(M) * disk_phi);
      Variables<TagsList> u{disk_num_grid_points};
      Variables<TagsList> expected_result{disk_num_grid_points};
      get(get<::Tags::TempScalar<0>>(u)) = 1.0 + f_low + f_top;
      get(get<::Tags::TempScalar<0>>(expected_result)) =
          1.0 + disk_factor_m1 * f_low;
      Spectral::filtering::zernike_b2_disk_filter(
          make_not_null(&u), disk, alpha, std::optional<unsigned>{half_power},
          1);
      CHECK_VARIABLES_APPROX(u, expected_result);
    }
  }
  {
    INFO("Precomputed z_filter gives same result as z_half_power");
    // Passing the z filter matrix explicitly (with z_half_power = nullopt)
    // must produce the same result as passing z_half_power and letting the
    // function compute the matrix internally.
    const unsigned z_half = 6;
    const Matrix z_mat = Spectral::filtering::exponential_filter(
        mesh.slice_through(2), 36.0, z_half);
    Variables<TagsList> u_precomputed{num_grid_points};
    Variables<TagsList> u_computed{num_grid_points};
    get(get<::Tags::TempScalar<0>>(u_precomputed)) = smooth_field;
    get(get<::Tags::TempScalar<0>>(u_computed)) = smooth_field;
    DataVector buf{};
    Spectral::filtering::zernike_b2_cylinder_filter(
        make_not_null(&u_precomputed), make_not_null(&buf), mesh, 36.0,
        std::optional<unsigned>{8}, std::nullopt, 0,
        std::optional<Matrix>{z_mat});
    Spectral::filtering::zernike_b2_cylinder_filter(
        make_not_null(&u_computed), mesh, 36.0, std::optional<unsigned>{8},
        std::optional<unsigned>{z_half}, 0);
    CHECK_VARIABLES_APPROX(u_precomputed, u_computed);
  }
}

#ifdef SPECTRE_DEBUG
void test_asserts() {
  using TagsList = tmpl::list<::Tags::TempScalar<0>>;

  {
    INFO("Disk: n_r <= 1 triggers min n_r assert");
    const Mesh<2> mesh{{1, 10},
                       {Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2},
                       {Spectral::Quadrature::GaussRadauUpper,
                        Spectral::Quadrature::Equiangular}};
    Variables<TagsList> u{mesh.number_of_grid_points()};
    CHECK_THROWS_WITH(
        Spectral::filtering::zernike_b2_disk_exponential_filter(
            make_not_null(&u), mesh, 1.0, 2),
        Catch::Matchers::ContainsSubstring(
            "At least 2 radial grid points are required to filter ZernikeB2,"));
  }
  {
    INFO("Disk: even n_phi triggers the odd-check assert");
    const Mesh<2> mesh{{3, 4},
                       {Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2},
                       {Spectral::Quadrature::GaussRadauUpper,
                        Spectral::Quadrature::Equiangular}};
    Variables<TagsList> u{mesh.number_of_grid_points()};
    CHECK_THROWS_WITH(
        Spectral::filtering::zernike_b2_disk_exponential_filter(
            make_not_null(&u), mesh, 1.0, 2),
        Catch::Matchers::ContainsSubstring(
            "Fourier with an even number of grid points can be unstable"));
  }
  {
    INFO("Disk: M > n_r_max triggers the Fourier-vs-Zernike size assert");
    const Mesh<2> mesh{{2, 7},
                       {Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2},
                       {Spectral::Quadrature::GaussRadauUpper,
                        Spectral::Quadrature::Equiangular}};
    Variables<TagsList> u{mesh.number_of_grid_points()};
    CHECK_THROWS_WITH(
        Spectral::filtering::zernike_b2_disk_exponential_filter(
            make_not_null(&u), mesh, 1.0, 2),
        Catch::Matchers::ContainsSubstring(
            "We choose to enforce the restriction that the Fourier modal space "
            "is not larger than the Zernike angular capabilities"));
  }
  {
    INFO("Disk: killing more angular modes than resolved triggers assert");
    const Mesh<2> mesh{{5, 7},
                       {Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2},
                       {Spectral::Quadrature::GaussRadauUpper,
                        Spectral::Quadrature::Equiangular}};
    Variables<TagsList> u{mesh.number_of_grid_points()};
    CHECK_THROWS_WITH(Spectral::filtering::zernike_b2_disk_filter(
                          make_not_null(&u), mesh, 1.0, std::nullopt, 4),
                      Catch::Matchers::ContainsSubstring(
                          "Cannot zero 4 angular modes when only 3"));
  }
  {
    INFO("Cylinder: even n_phi triggers the odd-check assert");
    const Mesh<3> mesh{{3, 4, 3},
                       {Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2,
                        Spectral::Basis::Legendre},
                       {Spectral::Quadrature::GaussRadauUpper,
                        Spectral::Quadrature::Equiangular,
                        Spectral::Quadrature::GaussLobatto}};
    Variables<TagsList> u{mesh.number_of_grid_points()};
    CHECK_THROWS_WITH(
        Spectral::filtering::zernike_b2_cylinder_exponential_filter(
            make_not_null(&u), mesh, 1.0, 2),
        Catch::Matchers::ContainsSubstring(
            "Fourier with an even number of grid points can be unstable"));
  }
  {
    INFO("Cylinder: M > n_r_max triggers the Fourier-vs-Zernike size assert");
    const Mesh<3> mesh{{2, 7, 3},
                       {Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2,
                        Spectral::Basis::Legendre},
                       {Spectral::Quadrature::GaussRadauUpper,
                        Spectral::Quadrature::Equiangular,
                        Spectral::Quadrature::GaussLobatto}};
    Variables<TagsList> u{mesh.number_of_grid_points()};
    CHECK_THROWS_WITH(
        Spectral::filtering::zernike_b2_cylinder_exponential_filter(
            make_not_null(&u), mesh, 1.0, 2),
        Catch::Matchers::ContainsSubstring(
            "We choose to enforce the restriction that the Fourier modal space "
            "is not larger than the Zernike angular capabilities"));
  }
  {
    INFO("Cylinder: killing more angular modes than resolved triggers assert");
    const Mesh<3> mesh{{5, 7, 3},
                       {Spectral::Basis::ZernikeB2, Spectral::Basis::ZernikeB2,
                        Spectral::Basis::Legendre},
                       {Spectral::Quadrature::GaussRadauUpper,
                        Spectral::Quadrature::Equiangular,
                        Spectral::Quadrature::GaussLobatto}};
    Variables<TagsList> u{mesh.number_of_grid_points()};
    CHECK_THROWS_WITH(
        Spectral::filtering::zernike_b2_cylinder_filter(
            make_not_null(&u), mesh, 1.0, std::nullopt, std::nullopt, 4),
        Catch::Matchers::ContainsSubstring(
            "Cannot zero 4 angular modes when only 3"));
  }
}
#endif

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.B2Filter",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  // We test the filtering by verifying known outcomes (i.e. constants
  // unaffected, highest modes completely zeroed) as well as testing
  // individual modes are scaled as expected
  test_disk_filter();
  test_disk_filter_weights();
  test_cylinder_filter();
  test_cylinder_filter_weights();
  test_generalized_filters();
#ifdef SPECTRE_DEBUG
  test_asserts();
#endif  // SPECTRE_DEBUG
}
}  // namespace
