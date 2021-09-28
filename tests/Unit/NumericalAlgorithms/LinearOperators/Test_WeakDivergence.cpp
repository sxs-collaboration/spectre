// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MetricIdentityJacobian.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/WeakDivergence.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct Var2 : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim>;
};

template <size_t Dim>
void test_weak_divergence_random_jacobian(const Mesh<Dim>& mesh) {
  CAPTURE(Dim);
  CAPTURE(mesh);
  CAPTURE(mesh.quadrature(0));
  std::uniform_real_distribution<double> dist(-1.0, 2.3);
  MAKE_GENERATOR(gen);
  using flux_tags =
      tmpl::list<Tags::Flux<Var1, tmpl::size_t<Dim>, Frame::Inertial>,
                 Tags::Flux<Var2<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>;
  using div_tags = tmpl::list<
      Tags::div<Tags::Flux<Var1, tmpl::size_t<Dim>, Frame::Inertial>>,
      Tags::div<Tags::Flux<Var2<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>>;

  tnsr::I<DataVector, Dim, Frame::Inertial> inertial_coords{
      mesh.number_of_grid_points()};
  Jacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial> jacobian{
      mesh.number_of_grid_points()};

  fill_with_random_values(make_not_null(&inertial_coords), make_not_null(&gen),
                          make_not_null(&dist));
  fill_with_random_values(make_not_null(&jacobian), make_not_null(&gen),
                          make_not_null(&dist));

  InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
      det_jac_times_inverse_jacobian{};

  dg::metric_identity_det_jac_times_inv_jac(
      make_not_null(&det_jac_times_inverse_jacobian), mesh, inertial_coords,
      jacobian);
  // Generate constant fluxes that aren't all the same.
  Variables<flux_tags> fluxes{mesh.number_of_grid_points(), 2.0};
  tmpl::for_each<flux_tags>([&fluxes](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    auto& flux = get<tag>(fluxes);
    for (size_t storage_index = 0; storage_index < flux.size();
         ++storage_index) {
      flux[storage_index] = storage_index + 3.0;
    }
  });

  Variables<div_tags> divergence_result{mesh.number_of_grid_points()};

  weak_divergence(make_not_null(&divergence_result), fluxes, mesh,
                  det_jac_times_inverse_jacobian);

  Approx local_approx = Approx::custom().epsilon(5.0e-12).scale(1.);
  const Variables<div_tags> expected_divergence_result{
      mesh.number_of_grid_points(), 0.0};
  tmpl::for_each<div_tags>([&divergence_result, &expected_divergence_result,
                            &local_approx, &mesh](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    if (mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto) {
      // Only the interior points are zero. Points on the boundary are non-zero
      // because they need the flux lifted to be zero.
      for (size_t i = 1; i < mesh.extents(0) - 1; ++i) {
        CAPTURE(i);
        for (size_t j = 1; j < ((Dim > 1) ? mesh.extents(1) - 1 : 2); ++j) {
          CAPTURE(j);
          for (size_t k = 1; k < ((Dim > 2) ? mesh.extents(2) - 1 : 2); ++k) {
            CAPTURE(k);
            for (size_t storage_index = 0;
                 storage_index < get<tag>(divergence_result).size();
                 ++storage_index) {
              size_t collapsed_index = 0;
              if constexpr (Dim == 3) {
                collapsed_index =
                    ::collapsed_index(Index<3>{i, j, k}, mesh.extents());
              } else if constexpr (Dim == 2) {
                collapsed_index =
                    ::collapsed_index(Index<2>{i, j}, mesh.extents());
              } else {
                collapsed_index =
                    ::collapsed_index(Index<1>{i}, mesh.extents());
              }
              CHECK_ITERABLE_CUSTOM_APPROX(
                  get<tag>(divergence_result)[storage_index][collapsed_index],
                  get<tag>(expected_divergence_result)[storage_index]
                                                      [collapsed_index],
                  local_approx);
            }
          }
        }
      }
    } else {
      for (size_t storage_index = 0;
           storage_index < get<tag>(divergence_result).size() and
           (Dim == 1 or mesh.extents(0) == 3);
           ++storage_index) {
        size_t collapsed_index = std::numeric_limits<size_t>::max();
        if constexpr (Dim == 3) {
          collapsed_index = ::collapsed_index(
              Index<3>{mesh.extents(0) / 2, mesh.extents(1) / 2,
                       mesh.extents(2) / 2},
              mesh.extents());
        } else if constexpr (Dim == 2) {
          collapsed_index = ::collapsed_index(
              Index<2>{mesh.extents(0) / 2, mesh.extents(1) / 2},
              mesh.extents());
        } else {
          collapsed_index =
              ::collapsed_index(Index<1>{mesh.extents(0) / 2}, mesh.extents());
        }
        CHECK_ITERABLE_CUSTOM_APPROX(
            get<tag>(divergence_result)[storage_index][collapsed_index],
            get<tag>(
                expected_divergence_result)[storage_index][collapsed_index],
            local_approx);
      }
    }
  });
}

template <size_t Dim>
void test_weak_divergence_constant_jacobian(const Mesh<Dim>& mesh) {
  CAPTURE(Dim);
  CAPTURE(mesh);
  CAPTURE(mesh.quadrature(0));
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  MAKE_GENERATOR(gen);
  using flux_tags =
      tmpl::list<Tags::Flux<Var1, tmpl::size_t<Dim>, Frame::Inertial>,
                 Tags::Flux<Var2<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>;
  using div_tags = tmpl::list<
      Tags::div<Tags::Flux<Var1, tmpl::size_t<Dim>, Frame::Inertial>>,
      Tags::div<Tags::Flux<Var2<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>>;

  const auto logical_coords = logical_coordinates(mesh);
  tnsr::I<DataVector, Dim, Frame::Inertial> inertial_coords{
      mesh.number_of_grid_points()};
  Jacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial> jacobian{
      mesh.number_of_grid_points(), 0.0};
  InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
      inverse_jacobian{mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < Dim; ++i) {
    jacobian.get(i, i) = 2.0;
    inertial_coords.get(i) = 2.0 * logical_coords.get(i);
  }

  InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
      det_jac_times_inverse_jacobian{};

  dg::metric_identity_det_jac_times_inv_jac(
      make_not_null(&det_jac_times_inverse_jacobian), mesh, inertial_coords,
      jacobian);
  const auto det_jacobian = determinant(jacobian);
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      inverse_jacobian.get(i, j) =
          det_jac_times_inverse_jacobian.get(i, j) / get(det_jacobian);
    }
  }

  // Generate smooth fluxes
  Variables<flux_tags> fluxes{mesh.number_of_grid_points()};
  const auto compute_smooth_fluxes = [](const auto fluxes_ptr,
                                        const auto& coords) {
    auto& local_fluxes = *fluxes_ptr;
    tmpl::for_each<flux_tags>([&local_fluxes, &coords](auto tag_v) {
      using tag = tmpl::type_from<decltype(tag_v)>;
      for (size_t tensor_index = 0;
           tensor_index < get<tag>(local_fluxes).size(); ++tensor_index) {
        get<tag>(local_fluxes)[tensor_index] =
            std::sqrt(tensor_index + 1) * square(get<0>(coords));
        for (size_t d = 1; d < Dim; ++d) {
          get<tag>(local_fluxes)[tensor_index] +=
              std::sqrt(tensor_index + 1) * coords.get(d);
        }
      }
    });
  };
  if (mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto) {
    fill_with_random_values(make_not_null(&fluxes), make_not_null(&gen),
                            make_not_null(&dist));
  } else {
    compute_smooth_fluxes(make_not_null(&fluxes), inertial_coords);
  }

  Variables<div_tags> divergence_result{mesh.number_of_grid_points()};
  weak_divergence(make_not_null(&divergence_result), fluxes, mesh,
                  det_jac_times_inverse_jacobian);

  Variables<div_tags> expected_divergence = [&compute_smooth_fluxes,
                                             &det_jacobian, &fluxes,
                                             &inverse_jacobian, &mesh]() {
    if (mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto) {
      auto local_expected_divergence =
          divergence(fluxes, mesh, inverse_jacobian);
      local_expected_divergence *= -get(det_jacobian);
      return local_expected_divergence;
    } else {
      // We are on a Gauss grid. Compute the weak divergence on a Gauss-Lobatto
      // grid and interpolate it to the Gauss grid. In this case we explicitly
      // didn't use random data so that the result all fits inside the basis
      // function space.
      const Mesh<Dim> gl_mesh{mesh.extents(0), mesh.basis(0),
                              Spectral::Quadrature::GaussLobatto};
      const auto gl_logical_coords = logical_coordinates(gl_mesh);
      tnsr::I<DataVector, Dim, Frame::Inertial> gl_inertial_coords{
          gl_mesh.number_of_grid_points()};
      Jacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
          gl_jacobian{gl_mesh.number_of_grid_points(), 0.0};
      for (size_t i = 0; i < Dim; ++i) {
        gl_jacobian.get(i, i) = 2.0;
        gl_inertial_coords.get(i) = 2.0 * gl_logical_coords.get(i);
      }

      InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
          gl_det_jac_times_inverse_jacobian{};

      dg::metric_identity_det_jac_times_inv_jac(
          make_not_null(&gl_det_jac_times_inverse_jacobian), gl_mesh,
          gl_inertial_coords, gl_jacobian);

      Variables<flux_tags> gl_fluxes{gl_mesh.number_of_grid_points()};
      compute_smooth_fluxes(make_not_null(&gl_fluxes), gl_inertial_coords);

      Variables<div_tags> gl_divergence_result{gl_mesh.number_of_grid_points()};
      weak_divergence(make_not_null(&gl_divergence_result), gl_fluxes, gl_mesh,
                      gl_det_jac_times_inverse_jacobian);

      const Matrix interp_1d = Spectral::interpolation_matrix(
          Mesh<1>{mesh.extents(0), mesh.basis(0),
                  Spectral::Quadrature::GaussLobatto},
          get<0>(logical_coordinates(Mesh<1>{mesh.extents(0), mesh.basis(0),
                                             Spectral::Quadrature::Gauss})));
      return apply_matrices(make_array<Dim>(interp_1d), gl_divergence_result,
                            gl_mesh.extents());
    }
  }();

  Approx local_approx = Approx::custom().epsilon(5.0e-12).scale(1.);
  tmpl::for_each<div_tags>([&divergence_result, &expected_divergence,
                            &local_approx, &mesh](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    if (mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto) {
      // Only the interior points are zero. Points on the boundary are non-zero
      // because they need the flux lifted to be zero.
      for (size_t i = 1; i < mesh.extents(0) - 1; ++i) {
        CAPTURE(i);
        for (size_t j = 1; j < ((Dim > 1) ? mesh.extents(1) - 1 : 2); ++j) {
          CAPTURE(j);
          for (size_t k = 1; k < ((Dim > 2) ? mesh.extents(2) - 1 : 2); ++k) {
            CAPTURE(k);
            for (size_t storage_index = 0;
                 storage_index < get<tag>(divergence_result).size();
                 ++storage_index) {
              size_t collapsed_index = 0;
              if constexpr (Dim == 3) {
                collapsed_index =
                    ::collapsed_index(Index<3>{i, j, k}, mesh.extents());
              } else if constexpr (Dim == 2) {
                collapsed_index =
                    ::collapsed_index(Index<2>{i, j}, mesh.extents());
              } else {
                collapsed_index =
                    ::collapsed_index(Index<1>{i}, mesh.extents());
              }
              CHECK_ITERABLE_CUSTOM_APPROX(
                  get<tag>(divergence_result)[storage_index][collapsed_index],
                  get<tag>(expected_divergence)[storage_index][collapsed_index],
                  local_approx);
            }
          }
        }
      }
    } else {
      CHECK_ITERABLE_CUSTOM_APPROX(get<tag>(divergence_result),
                                   get<tag>(expected_divergence), local_approx);
    }
  });
}

template <size_t Dim>
void test() {
  for (size_t num_pts = 3; num_pts < 9; num_pts += 2) {
    for (const auto& quadrature :
         {Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto}) {
      test_weak_divergence_random_jacobian(
          Mesh<Dim>{num_pts, Spectral::Basis::Legendre, quadrature});
      if constexpr (Dim == 1) {
        // Haven't figured out a good test in 2d and 3d
        test_weak_divergence_constant_jacobian(
            Mesh<Dim>{num_pts, Spectral::Basis::Legendre, quadrature});
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.WeakDivergence",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  // We already have tests that verify that the matrix used in the weak
  // divergence is correct. What we need to test is that the function
  // weak_divergence behaves correctly. This is pretty tricky, unfortunately.
  // The best way of doing so is to check that the strong divergence is
  // identical to weak divergence plus the lifted boundary terms. However, we do
  // not (yet) have the lifting terms coded up, so we need to do something else.
  // We have two functions that do the hard work:
  //
  // - test_weak_divergence_random_jacobian
  // - test_weak_divergence_constant_jacobian
  //
  // test_weak_divergence_random_jacobian:
  //
  // This generates a completely random Jacobian and verifies that the
  // divergence of this Jacobian is zero. Again, this is tricky because the
  // surface terms are not accounted for. With Gauss-Lobatto points this means
  // the interior points are zero. For Gauss points the problem is much harder.
  // Only the central grid point is zero and only when there is an odd number of
  // grid points. Verifying that the metric identities are satisfied is a
  // necessary but not sufficient condition.
  //
  // test_weak_divergence_constant_jacobian:
  //
  // This uses a constant but non-trivial Jacobian and verifies that the strong
  // and weak divergence are equal. With Gauss-Lobatto points this is everywhere
  // in the interior. For Gauss points we only check that interpolating the
  // Gauss-Lobatto result to Gauss points gives the same weak divergence. Since
  // we verify Gauss-Lobatto points separately this is a good test. We only do
  // this check in 1d because 2d and 3d are annoying to get right, so much so
  // that it's better to check everything exactly once the lifting terms are in.
  test<1>();
  test<2>();
  test<3>();
}
