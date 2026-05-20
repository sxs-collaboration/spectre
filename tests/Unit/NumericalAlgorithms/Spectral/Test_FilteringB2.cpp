// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
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
    const Matrix& mtn_z =
        Spectral::modal_to_nodal_matrix(mesh.slice_through(2));
    DataVector f_top_z_vals(num_grid_points, 0.0);
    const size_t n_rph = n_r * n_ph;
    for (size_t k = 0; k < n_z; ++k) {
      const double z_nodal_val = mtn_z(k, n_z - 1);
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

      // z weight test: the k-th Legendre modal basis function in z, constant
      // in (r, phi). The nodal representation is column k of the
      // modal-to-nodal matrix for the z mesh. Since the function is constant
      // in (r, phi) it sits entirely in the (n=0, m=0) disk mode whose disk
      // filter weight is exp(0)=1. So the full weight is the z weight alone:
      //   exp(-alpha * (k / (n_z-1))^(2p)).
      const Matrix& mtn_z =
          Spectral::modal_to_nodal_matrix(mesh.slice_through(2));
      const auto n_z_order = static_cast<double>(n_z - 1);

      for (size_t k = 0; k < n_z; ++k) {
        CAPTURE(k);
        const double z_factor = exp(
            -alpha * pow(static_cast<double>(k) / n_z_order, 2 * half_power));

        DataVector f_z_vals(num_grid_points, 0.0);
        for (size_t kk = 0; kk < n_z; ++kk) {
          const double z_nodal_val = mtn_z(kk, k);
          for (size_t ij = 0; ij < n_rph; ++ij) {
            f_z_vals[ij + n_rph * kk] = z_nodal_val;
          }
        }

        Variables<TagsList> u{num_grid_points};
        Variables<TagsList> expected_result{num_grid_points};
        get(get<::Tags::TempScalar<0>>(u)) = f_z_vals;
        Spectral::filtering::zernike_b2_cylinder_exponential_filter(
            make_not_null(&u), mesh, alpha, half_power);
        get(get<::Tags::TempScalar<0>>(expected_result)) = z_factor * f_z_vals;
        CHECK_VARIABLES_APPROX(u, expected_result);
      }
    }
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
#ifdef SPECTRE_DEBUG
  test_asserts();
#endif  // SPECTRE_DEBUG
}
}  // namespace
