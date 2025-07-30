// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <random>
#include <string>
#include <type_traits>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/SphericalToCartesianPfaffian.hpp"
#include "Domain/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/SphericalHarmonics/YlmTestFunctions.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/RealSphericalHarmonics.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <typename DataType, size_t Dim, class Frame = ::Frame::Grid>
struct Var1 : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame>;
  static auto f(const std::array<size_t, Dim>& coeffs,
                const tnsr::I<DataVector, Dim, Frame>& x) {
    tnsr::i<DataType, Dim, Frame> result(x.begin()->size(), 0.);
    for (size_t i = 0; i < Dim; ++i) {
      result.get(i) = (i + 2);
      for (size_t d = 0; d < Dim; ++d) {
        result.get(i) *= pow(x.get(d), gsl::at(coeffs, d));
      }
    }
    // Set the imaginary part
    if constexpr (std::is_same_v<DataType, ComplexDataVector>) {
      for (size_t i = 0; i < Dim; ++i) {
        ComplexDataVector imaginary_part(x.begin()->size(), 0.);
        imaginary_part = std::complex<double>(0., static_cast<double>(i) + 3);
        for (size_t d = 0; d < Dim; ++d) {
          imaginary_part *=
              (static_cast<double>(d) + 2) * pow(x.get(d), gsl::at(coeffs, d));
        }
        result.get(i) += imaginary_part;
      }
    }
    return result;
  }
  static auto df(const std::array<size_t, Dim>& coeffs,
                 const tnsr::I<DataVector, Dim, Frame>& x) {
    tnsr::ij<DataType, Dim, Frame> result(x.begin()->size(), 0.);
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = 0; j < Dim; ++j) {
        result.get(i, j) = (j + 2);
        for (size_t d = 0; d < Dim; ++d) {
          if (d == i) {
            if (0 == gsl::at(coeffs, d)) {
              result.get(i, j) = 0.;
            } else {
              result.get(i, j) *=
                  gsl::at(coeffs, d) * pow(x.get(d), gsl::at(coeffs, d) - 1);
            }
          } else {
            result.get(i, j) *= pow(x.get(d), gsl::at(coeffs, d));
          }
        }
      }
    }
    // Set the imaginary part
    if constexpr (std::is_same_v<DataType, ComplexDataVector>) {
      for (size_t i = 0; i < Dim; ++i) {
        for (size_t j = 0; j < Dim; ++j) {
          ComplexDataVector imaginary_part(x.begin()->size(), 0.);
          imaginary_part = std::complex<double>(0., static_cast<double>(j) + 3);
          for (size_t d = 0; d < Dim; ++d) {
            if (d == i) {
              if (0 == gsl::at(coeffs, d)) {
                imaginary_part = 0.;
              } else {
                imaginary_part *= (static_cast<double>(d) + 2) *
                                  gsl::at(coeffs, d) *
                                  pow(x.get(d), gsl::at(coeffs, d) - 1);
              }
            } else {
              imaginary_part *= (static_cast<double>(d) + 2) *
                                pow(x.get(d), gsl::at(coeffs, d));
            }
          }
          result.get(i, j) += imaginary_part;
        }
      }
    }
    return result;
  }
};

template <typename DataType>
struct Var2 : db::SimpleTag {
  using type = Scalar<DataType>;
  template <size_t Dim, class Frame>
  static auto f(const std::array<size_t, Dim>& coeffs,
                const tnsr::I<DataVector, Dim, Frame>& x) {
    Scalar<DataType> result(x.begin()->size(), 1.);
    for (size_t d = 0; d < Dim; ++d) {
      result.get() *= pow(x.get(d), gsl::at(coeffs, d));
    }
    return result;
  }
  template <size_t Dim, class Frame>
  static auto df(const std::array<size_t, Dim>& coeffs,
                 const tnsr::I<DataVector, Dim, Frame>& x) {
    tnsr::i<DataType, Dim, Frame> result(x.begin()->size(), 1.);
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t d = 0; d < Dim; ++d) {
        if (d == i) {
          if (0 == gsl::at(coeffs, d)) {
            result.get(i) = 0.0;
          } else {
            result.get(i) *=
                gsl::at(coeffs, d) * pow(x.get(d), gsl::at(coeffs, d) - 1);
          }
        } else {
          result.get(i) *= pow(x.get(d), gsl::at(coeffs, d));
        }
      }
    }
    return result;
  }
};

template <typename DataType, size_t Dim>
using two_vars = tmpl::list<Var1<DataType, Dim>, Var2<DataType>>;

template <typename DataType, size_t Dim>
using one_var = tmpl::list<Var1<DataType, Dim>>;

template <typename DataType>
using scalar_var = tmpl::list<Var2<DataType>>;

template <typename GradientTags, typename VariableTags, size_t Dim>
void test_logical_partial_derivative_per_tensor(
    const std::array<Variables<GradientTags>, Dim>& du,
    const Variables<VariableTags>& u, const Mesh<Dim>& mesh) {
  using VectorType = typename Variables<VariableTags>::vector_type;
  using ValueType = typename Variables<VariableTags>::value_type;
  tmpl::for_each<GradientTags>([&du, &mesh, &u](auto gradient_tag_v) {
    using gradient_tag = tmpl::type_from<decltype(gradient_tag_v)>;
    const auto single_du =
        logical_partial_derivative(get<gradient_tag>(u), mesh);
    for (size_t storage_index = 0; storage_index < get<gradient_tag>(u).size();
         ++storage_index) {
      for (size_t d = 0; d < Dim; ++d) {
        const auto deriv_tensor_index =
            prepend(get<gradient_tag>(u).get_tensor_index(storage_index), d);
        CHECK_ITERABLE_APPROX(single_du.get(deriv_tensor_index),
                              get<gradient_tag>(gsl::at(du, d))[storage_index]);
      }
    }
    std::decay_t<decltype(single_du)> single_du_not_null{};
    VectorType buffer{mesh.number_of_grid_points()};
    gsl::span<ValueType> buffer_view{buffer.data(), buffer.size()};
    logical_partial_derivative(make_not_null(&single_du_not_null),
                               make_not_null(&buffer_view),
                               get<gradient_tag>(u), mesh);
    CHECK_ITERABLE_APPROX(single_du_not_null, single_du);

    // Check we can do derivatives when the components of `u` aren't contiguous
    // in memory.
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    std::decay_t<decltype(get<gradient_tag>(u))> non_contiguous_u =
        get<gradient_tag>(u);
    const auto non_contiguous_single_du =
        logical_partial_derivative(get<gradient_tag>(u), mesh);
    CHECK_ITERABLE_APPROX(non_contiguous_single_du, single_du);
  });
}

template <typename GradientTags, typename VariableTags, size_t Dim,
          typename DerivativeFrame>
void test_partial_derivative_per_tensor(
    const Variables<GradientTags>& du, const Variables<VariableTags>& u,
    const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) {
  tmpl::for_each<GradientTags>(
      [&du, &mesh, &u, &inverse_jacobian](auto gradient_tag_v) {
        using gradient_tag = tmpl::type_from<decltype(gradient_tag_v)>;
        using var_tag = typename gradient_tag::tag;

        const auto single_du =
            partial_derivative(get<var_tag>(u), mesh, inverse_jacobian);

        Approx local_approx = Approx::custom().epsilon(1e-10).scale(1.0);
        CHECK_ITERABLE_CUSTOM_APPROX(single_du, get<gradient_tag>(du),
                                     local_approx);

        // Check we can do derivatives when the components of `u` aren't
        // contiguous in memory.
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        const auto non_contiguous_u = get<var_tag>(u);
        const auto non_contiguous_single_du =
            partial_derivative(non_contiguous_u, mesh, inverse_jacobian);
        CHECK_ITERABLE_APPROX(non_contiguous_single_du, single_du);
      });
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_logical_partial_derivatives_1d(const Mesh<1>& mesh) {
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const DataVector& xi = Spectral::collocation_points(mesh.slice_through(0));
  Variables<VariableTags> u(number_of_grid_points);
  Variables<GradientTags> du_expected(number_of_grid_points);
  for (size_t a = 0; a < mesh.extents(0); ++a) {
    for (size_t n = 0; n < u.number_of_independent_components; ++n) {
      for (size_t s = 0; s < number_of_grid_points; ++s) {
        u.data()[s + n * number_of_grid_points]  // NOLINT
            = static_cast<double>(n + 1) * pow(xi[s], static_cast<double>(a));
      }
    }
    // Generate expected data
    for (size_t n = 0;
         n < Variables<GradientTags>::number_of_independent_components; ++n) {
      for (size_t s = 0; s < number_of_grid_points; ++s) {
        du_expected.data()[s + n * number_of_grid_points] =
            (0 == a ? 0.0
                    : static_cast<double>(a * (n + 1)) *
                          pow(xi[s], static_cast<double>(a - 1)));
      }
    }
    CHECK_VARIABLES_APPROX(
        (logical_partial_derivatives<GradientTags>(u, mesh)[0]), du_expected);
    std::array<Variables<GradientTags>, 1> du{};
    logical_partial_derivatives(make_not_null(&du), u, mesh);
    CHECK_VARIABLES_APPROX(du[0], du_expected);
    // We've checked that du is correct, now test that taking derivatives of
    // individual tensors gets the matching result.
    test_logical_partial_derivative_per_tensor(du, u, mesh);
  }
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_logical_partial_derivatives_2d(const Mesh<2>& mesh) {
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const DataVector& xi = Spectral::collocation_points(mesh.slice_through(0));
  const DataVector& eta = Spectral::collocation_points(mesh.slice_through(1));
  Variables<VariableTags> u(mesh.number_of_grid_points());
  std::array<Variables<GradientTags>, 2> du_expected{};
  du_expected[0].initialize(mesh.number_of_grid_points());
  du_expected[1].initialize(mesh.number_of_grid_points());
  const size_t a = mesh.extents(0) - 1;
  const size_t b = mesh.extents(1) - 1;
  for (size_t n = 0; n < u.number_of_independent_components; ++n) {
    for (IndexIterator<2> ii(mesh.extents()); ii; ++ii) {
      u.data()[ii.collapsed_index() + n * number_of_grid_points] =  // NOLINT
          static_cast<double>(n + 1) *
          pow(xi[ii()[0]], static_cast<double>(a)) *
          pow(eta[ii()[1]], static_cast<double>(b));
    }
  }
  // Generate expected data
  for (size_t n = 0;
       n < Variables<GradientTags>::number_of_independent_components; ++n) {
    for (IndexIterator<2> ii(mesh.extents()); ii; ++ii) {
      du_expected[0].data()[ii.collapsed_index() + n * number_of_grid_points] =
          (0 == a ? 0.0
                  : static_cast<double>(a * (n + 1)) *
                        pow(xi[ii()[0]], static_cast<double>(a - 1)) *
                        pow(eta[ii()[1]], static_cast<double>(b)));
      du_expected[1].data()[ii.collapsed_index() + n * number_of_grid_points] =
          (0 == b ? 0.0
                  : static_cast<double>(b * (n + 1)) *
                        pow(xi[ii()[0]], static_cast<double>(a)) *
                        pow(eta[ii()[1]],
                            static_cast<double>(static_cast<double>(b - 1))));
    }
  }
  const auto du = logical_partial_derivatives<GradientTags>(u, mesh);
  CHECK_VARIABLES_APPROX(du[0], du_expected[0]);
  CHECK_VARIABLES_APPROX(du[1], du_expected[1]);
  // We've checked that du is correct, now test that taking derivatives of
  // individual tensors gets the matching result.
  test_logical_partial_derivative_per_tensor(du, u, mesh);
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_logical_partial_derivatives_3d(const Mesh<3>& mesh) {
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const DataVector& xi = Spectral::collocation_points(mesh.slice_through(0));
  const DataVector& eta = Spectral::collocation_points(mesh.slice_through(1));
  const DataVector& zeta = Spectral::collocation_points(mesh.slice_through(2));
  Variables<VariableTags> u(number_of_grid_points);
  std::array<Variables<GradientTags>, 3> du_expected{};
  du_expected[0].initialize(number_of_grid_points);
  du_expected[1].initialize(number_of_grid_points);
  du_expected[2].initialize(number_of_grid_points);
  const size_t a = mesh.extents(0) - 1;
  const size_t b = mesh.extents(1) - 1;
  const size_t c = mesh.extents(2) - 1;
  for (size_t n = 0; n < u.number_of_independent_components; ++n) {
    for (IndexIterator<3> ii(mesh.extents()); ii; ++ii) {
      u.data()[ii.collapsed_index() + n * number_of_grid_points] =  // NOLINT
          static_cast<double>(n + 1) *
          pow(xi[ii()[0]], static_cast<double>(a)) *
          pow(eta[ii()[1]], static_cast<double>(b)) *
          pow(zeta[ii()[2]], static_cast<double>(c));
    }
  }
  // Generate expected data
  for (size_t n = 0;
       n < Variables<GradientTags>::number_of_independent_components; ++n) {
    for (IndexIterator<3> ii(mesh.extents()); ii; ++ii) {
      du_expected[0].data()[ii.collapsed_index() + n * number_of_grid_points] =
          (0 == a ? 0.0
                  : static_cast<double>(a * (n + 1)) *
                        pow(xi[ii()[0]], static_cast<double>(a - 1)) *
                        pow(eta[ii()[1]], static_cast<double>(b)) *
                        pow(zeta[ii()[2]], static_cast<double>(c)));
      du_expected[1].data()[ii.collapsed_index() + n * number_of_grid_points] =
          (0 == b ? 0.0
                  : static_cast<double>(b * (n + 1)) *
                        pow(xi[ii()[0]], static_cast<double>(a)) *
                        pow(eta[ii()[1]], static_cast<double>(b - 1)) *
                        pow(zeta[ii()[2]], static_cast<double>(c)));
      du_expected[2].data()[ii.collapsed_index() + n * number_of_grid_points] =
          (0 == c ? 0.0
                  : static_cast<double>(c * (n + 1)) *
                        pow(xi[ii()[0]], static_cast<double>(a)) *
                        pow(eta[ii()[1]], static_cast<double>(b)) *
                        pow(zeta[ii()[2]], static_cast<double>(c - 1)));
    }
  }
  const auto du = logical_partial_derivatives<GradientTags>(u, mesh);
  CHECK_VARIABLES_APPROX(du[0], du_expected[0]);
  CHECK_VARIABLES_APPROX(du[1], du_expected[1]);
  CHECK_VARIABLES_APPROX(du[2], du_expected[2]);
  // We've checked that du is correct, now test that taking derivatives of
  // individual tensors gets the matching result.
  test_logical_partial_derivative_per_tensor(du, u, mesh);
}

template <size_t l, int m, typename VariableTags,
          typename GradientTags = VariableTags>
void test_logical_partial_derivatives_spherical_shell(const size_t n_r,
                                                      const size_t L) {
  const Mesh<3> mesh{
      {n_r, L + 1, 2 * L + 1},
      {Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
       Spectral::Basis::SphericalHarmonic},
      {Spectral::Quadrature::Gauss, Spectral::Quadrature::Gauss,
       Spectral::Quadrature::Equiangular}};
  const auto x = logical_coordinates(mesh);
  const DataVector r = x[0] + 2.0;
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const YlmTestFunctions::Ylm<l, m> y_lm{n_r, L, L};
  Variables<VariableTags> u{number_of_grid_points};
  for (size_t n = 0; n < u.number_of_independent_components; ++n) {
    DataVector component_ref(u.data() + n * number_of_grid_points,
                             number_of_grid_points);
    component_ref = (1.0 + static_cast<double>(n)) *
                    pow(r, -1.0 + static_cast<double>(n_r)) * y_lm.f();
  }
  std::array<Variables<GradientTags>, 3> du_expected{};
  du_expected[0].initialize(number_of_grid_points);
  du_expected[1].initialize(number_of_grid_points);
  du_expected[2].initialize(number_of_grid_points);
  for (size_t n = 0;
       n < Variables<GradientTags>::number_of_independent_components; ++n) {
    DataVector du_dxi_ref(du_expected[0].data() + n * number_of_grid_points,
                          number_of_grid_points);
    if (n_r == 1) {
      du_dxi_ref = 0.0;
    } else {
      du_dxi_ref = (-1.0 + static_cast<double>(n_r)) *
                   (1.0 + static_cast<double>(n)) *
                   pow(r, -2.0 + static_cast<double>(n_r)) * y_lm.f();
    }
    DataVector du_dth_ref(du_expected[1].data() + n * number_of_grid_points,
                          number_of_grid_points);
    du_dth_ref = (1.0 + static_cast<double>(n)) *
                 pow(r, -1.0 + static_cast<double>(n_r)) * y_lm.df_dth();
    DataVector du_dph_ref(du_expected[2].data() + n * number_of_grid_points,
                          number_of_grid_points);
    du_dph_ref = (1.0 + static_cast<double>(n)) *
                 pow(r, -1.0 + static_cast<double>(n_r)) * y_lm.df_dph();
  }
  const auto du = logical_partial_derivatives<GradientTags>(u, mesh);
  Approx local_approx = Approx::custom().epsilon(1.0e-13).scale(1.0);
  CHECK_VARIABLES_CUSTOM_APPROX(du[0], du_expected[0], local_approx);
  CHECK_VARIABLES_CUSTOM_APPROX(du[1], du_expected[1], local_approx);
  CHECK_VARIABLES_CUSTOM_APPROX(du[2], du_expected[2], local_approx);
  // We've checked that du is correct, now test that taking derivatives of
  // individual tensors gets the matching result.
  test_logical_partial_derivative_per_tensor(du, u, mesh);
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_partial_derivatives_1d(const Mesh<1>& mesh) {
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const Affine x_map{-1.0, 1.0, -0.3, 0.7};
  const auto map_1d =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
          Affine{x_map});
  const auto x = map_1d(logical_coordinates(mesh));
  const InverseJacobian<DataVector, 1, Frame::ElementLogical, Frame::Grid>
      inverse_jacobian(number_of_grid_points, 2.0);

  Variables<VariableTags> u(number_of_grid_points);
  Variables<
      db::wrap_tags_in<Tags::deriv, GradientTags, tmpl::size_t<1>, Frame::Grid>>
      expected_du(number_of_grid_points);
  Approx local_approx = Approx::custom().epsilon(1e-10).scale(1.0);
  for (size_t a = 0; a < mesh.extents(0); ++a) {
    CAPTURE(a);
    tmpl::for_each<VariableTags>([&a, &x, &u](auto tag) {
      using Tag = tmpl::type_from<decltype(tag)>;
      get<Tag>(u) = Tag::f({{a}}, x);
    });
    tmpl::for_each<GradientTags>([&a, &x, &expected_du](auto tag) {
      using Tag = typename decltype(tag)::type;
      using DerivativeTag = Tags::deriv<Tag, tmpl::size_t<1>, Frame::Grid>;
      get<DerivativeTag>(expected_du) = Tag::df({{a}}, x);
    });

    CHECK_VARIABLES_CUSTOM_APPROX(
        (partial_derivatives<GradientTags>(u, mesh, inverse_jacobian)),
        expected_du, local_approx);
    using vars_type =
        decltype(partial_derivatives<GradientTags>(u, mesh, inverse_jacobian));
    vars_type du{};
    partial_derivatives(make_not_null(&du), u, mesh, inverse_jacobian);
    CHECK_VARIABLES_CUSTOM_APPROX(du, expected_du, local_approx);
    partial_derivatives(make_not_null(&du),
                        logical_partial_derivatives<GradientTags>(u, mesh),
                        inverse_jacobian);
    CHECK_VARIABLES_CUSTOM_APPROX(du, expected_du, local_approx);
    // We've checked that du is correct, now test that taking derivatives of
    // individual tensors gets the matching result.
    test_partial_derivative_per_tensor(expected_du, u, mesh, inverse_jacobian);
  }
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_partial_derivatives_2d(const Mesh<2>& mesh) {
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const auto prod_map2d =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
          Affine2D{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55}});
  const auto x = prod_map2d(logical_coordinates(mesh));
  InverseJacobian<DataVector, 2, Frame::ElementLogical, Frame::Grid>
      inverse_jacobian(number_of_grid_points, 0.0);
  inverse_jacobian.get(0, 0) = 2.0;
  inverse_jacobian.get(1, 1) = 8.0;

  Variables<VariableTags> u(number_of_grid_points);
  Variables<
      db::wrap_tags_in<Tags::deriv, GradientTags, tmpl::size_t<2>, Frame::Grid>>
      expected_du(number_of_grid_points);
  Approx local_approx = Approx::custom().epsilon(1e-10).scale(1.0);
  for (size_t a = 0; a < mesh.extents(0); ++a) {
    for (size_t b = 0; b < mesh.extents(1); ++b) {
      tmpl::for_each<VariableTags>([&a, &b, &x, &u](auto tag) {
        using Tag = typename decltype(tag)::type;
        get<Tag>(u) = Tag::f({{a, b}}, x);
      });
      tmpl::for_each<GradientTags>([&a, &b, &x, &expected_du](auto tag) {
        using Tag = typename decltype(tag)::type;
        using DerivativeTag = Tags::deriv<Tag, tmpl::size_t<2>, Frame::Grid>;
        get<DerivativeTag>(expected_du) = Tag::df({{a, b}}, x);
      });

      CHECK_VARIABLES_CUSTOM_APPROX(
          (partial_derivatives<GradientTags>(u, mesh, inverse_jacobian)),
          expected_du, local_approx);
      using vars_type = decltype(partial_derivatives<GradientTags>(
          u, mesh, inverse_jacobian));
      vars_type du{};
      partial_derivatives(make_not_null(&du), u, mesh, inverse_jacobian);
      CHECK_VARIABLES_CUSTOM_APPROX(du, expected_du, local_approx);

      vars_type du_with_logical{};
      partial_derivatives(make_not_null(&du_with_logical),
                          logical_partial_derivatives<GradientTags>(u, mesh),
                          inverse_jacobian);
      CHECK_VARIABLES_CUSTOM_APPROX(du_with_logical, expected_du, local_approx);

      // We've checked that du is correct, now test that taking derivatives of
      // individual tensors gets the matching result.
      test_partial_derivative_per_tensor(du, u, mesh, inverse_jacobian);
    }
  }
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_partial_derivatives_3d(const Mesh<3>& mesh) {
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const auto prod_map3d =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
          Affine3D{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55},
                   Affine{-1.0, 1.0, 2.3, 2.8}});
  const auto x = prod_map3d(logical_coordinates(mesh));
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Grid>
      inverse_jacobian(number_of_grid_points, 0.0);
  inverse_jacobian.get(0, 0) = 2.0;
  inverse_jacobian.get(1, 1) = 8.0;
  inverse_jacobian.get(2, 2) = 4.0;

  Variables<VariableTags> u(number_of_grid_points);
  Variables<
      db::wrap_tags_in<Tags::deriv, GradientTags, tmpl::size_t<3>, Frame::Grid>>
      expected_du(number_of_grid_points);
  Approx local_approx = Approx::custom().epsilon(1e-9).scale(1.0);
  for (size_t a = 0; a < mesh.extents(0) / 2; ++a) {
    for (size_t b = 0; b < mesh.extents(1) / 2; ++b) {
      for (size_t c = 0; c < mesh.extents(2) / 2; ++c) {
        tmpl::for_each<VariableTags>([&a, &b, &c, &x, &u](auto tag) {
          using Tag = typename decltype(tag)::type;
          get<Tag>(u) = Tag::f({{a, b, c}}, x);
        });
        tmpl::for_each<GradientTags>([&a, &b, &c, &x, &expected_du](auto tag) {
          using Tag = typename decltype(tag)::type;
          using DerivativeTag = Tags::deriv<Tag, tmpl::size_t<3>, Frame::Grid>;
          get<DerivativeTag>(expected_du) = Tag::df({{a, b, c}}, x);
        });

        CHECK_VARIABLES_CUSTOM_APPROX(
            (partial_derivatives<GradientTags>(u, mesh, inverse_jacobian)),
            expected_du, local_approx);
        using vars_type = decltype(partial_derivatives<GradientTags>(
            u, mesh, inverse_jacobian));
        vars_type du{};
        partial_derivatives(make_not_null(&du), u, mesh, inverse_jacobian);
        CHECK_VARIABLES_CUSTOM_APPROX(du, expected_du, local_approx);

        vars_type du_with_logical{};
        partial_derivatives(make_not_null(&du_with_logical),
                            logical_partial_derivatives<GradientTags>(u, mesh),
                            inverse_jacobian);
        CHECK_VARIABLES_CUSTOM_APPROX(du_with_logical, expected_du,
                                      local_approx);

        // We've checked that du is correct, now test that taking derivatives of
        // individual tensors gets the matching result.
        test_partial_derivative_per_tensor(du, u, mesh, inverse_jacobian);
      }
    }
  }
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_partial_derivatives_spherical_shell() {
  const size_t L = 4;
  const size_t n_r = 4;
  const Mesh<3> mesh{
      {n_r, L + 1, 2 * L + 1},
      {Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
       Spectral::Basis::SphericalHarmonic},
      {Spectral::Quadrature::Gauss, Spectral::Quadrature::Gauss,
       Spectral::Quadrature::Equiangular}};
  const size_t number_of_grid_points = mesh.number_of_grid_points();
  const auto prod_map3d =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
          domain::CoordinateMaps::ProductOf2Maps<
              Affine, domain::CoordinateMaps::Identity<2>>{
              Affine{-1.0, 1.0, 1.0, 1.5},
              domain::CoordinateMaps::Identity<2>{}},
          domain::CoordinateMaps::SphericalToCartesianPfaffian{});
  const auto xi = logical_coordinates(mesh);
  const auto x = prod_map3d(xi);
  const InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Grid>
      inverse_jacobian = prod_map3d.inv_jacobian(xi);

  Variables<VariableTags> u(number_of_grid_points);
  Variables<
      db::wrap_tags_in<Tags::deriv, GradientTags, tmpl::size_t<3>, Frame::Grid>>
      expected_du(number_of_grid_points);
  Approx local_approx = Approx::custom().epsilon(1e-13).scale(1.0);
  for (size_t a = 0; a < L; ++a) {
    CAPTURE(a);
    for (size_t b = 0; b < L - a; ++b) {
      CAPTURE(b);
      for (size_t c = 0; c < L - a - b; ++c) {
        CAPTURE(c);
        tmpl::for_each<VariableTags>([&a, &b, &c, &x, &u](auto tag) {
          using Tag = typename decltype(tag)::type;
          get<Tag>(u) = Tag::f({{a, b, c}}, x);
        });
        tmpl::for_each<GradientTags>([&a, &b, &c, &x, &expected_du](auto tag) {
          using Tag = typename decltype(tag)::type;
          using DerivativeTag = Tags::deriv<Tag, tmpl::size_t<3>, Frame::Grid>;
          get<DerivativeTag>(expected_du) = Tag::df({{a, b, c}}, x);
        });

        CHECK_VARIABLES_CUSTOM_APPROX(
            (partial_derivatives<GradientTags>(u, mesh, inverse_jacobian)),
            expected_du, local_approx);
        using vars_type = decltype(partial_derivatives<GradientTags>(
            u, mesh, inverse_jacobian));
        vars_type du{};
        partial_derivatives(make_not_null(&du), u, mesh, inverse_jacobian);
        CHECK_VARIABLES_CUSTOM_APPROX(du, expected_du, local_approx);

        vars_type du_with_logical{};
        partial_derivatives(make_not_null(&du_with_logical),
                            logical_partial_derivatives<GradientTags>(u, mesh),
                            inverse_jacobian);
        CHECK_VARIABLES_CUSTOM_APPROX(du_with_logical, expected_du,
                                      local_approx);

        // We've checked that du is correct, now test that taking derivatives of
        // individual tensors gets the matching result.
        test_partial_derivative_per_tensor(du, u, mesh, inverse_jacobian);
      }
    }
  }
}

template <bool Spherical>
DataVector cartoon_func(const tnsr::I<DataVector, 3, Frame::Grid>& coords) {
  if constexpr (Spherical) {
    // a radial function, f(r) = f(x) because the computational domain is the x
    // axis
    return 0.01 * pow(get<0>(coords), 4) + 0.3 * pow(get<0>(coords), 3) -
           0.1 * pow(get<0>(coords), 2) - 2.0 * get<0>(coords) - 1.5;
  } else {
    // an axially symmetric function about the y axis,
    // f(\sqrt{x^2 + z^2}, y) = f(x, y) because the computational domain is the
    // x-y plane
    return square(get<1>(coords)) + square(get<0>(coords)) * get<1>(coords);
  }
}

template <bool Spherical>
DataVector cartoon_dfunc(size_t deriv_index,
                         const tnsr::I<DataVector, 3, Frame::Grid>& coords) {
  if constexpr (Spherical) {
    if (deriv_index == 0) {
      return 0.04 * pow(get<0>(coords), 3) + 0.9 * pow(get<0>(coords), 2) -
             0.2 * get<0>(coords) - 2.0;
    } else {
      return {get<0>(coords).size(), 0.0};
    }
  } else {
    if (deriv_index == 0) {
      return 2.0 * get<0>(coords) * get<1>(coords);
    } else if (deriv_index == 1) {
      return 2.0 * get<1>(coords) + square(get<0>(coords));
    } else {
      return {get<0>(coords).size(), 0.0};
    }
  }
}

template <bool Spherical>
void test_cartoon(const double x_start) {
  Mesh<3> mesh;
  tnsr::I<DataVector, 3, Frame::Grid> coords;
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Grid>
      inv_jacobian;

  using Identity1D = domain::CoordinateMaps::Identity<1>;
  const Identity1D identity_cartoon_map;

  if constexpr (Spherical) {
    // spherical  symmetry
    const size_t num_grid_pts = 8;
    const double x_end = 4.0;

    mesh = Mesh<3>{{{num_grid_pts, 1, 1}},
                   {{Spectral::Basis::Legendre, Spectral::Basis::Cartoon,
                     Spectral::Basis::Cartoon}},
                   {{Spectral::Quadrature::GaussLobatto,
                     Spectral::Quadrature::SphericalSymmetry,
                     Spectral::Quadrature::SphericalSymmetry}}};

    const Affine affine_x_map(-1.0, 1.0, x_start, x_end);

    using Cartoon_map_combination =
        domain::CoordinateMaps::ProductOf3Maps<Affine, Identity1D, Identity1D>;
    const domain::CoordinateMap<Frame::ElementLogical, Frame::Grid,
                                Cartoon_map_combination>
        map{{affine_x_map, identity_cartoon_map, identity_cartoon_map}};
    inv_jacobian = map.inv_jacobian(logical_coordinates(mesh));
    coords = map(logical_coordinates(mesh));
  } else {
    // axial symmetry
    const size_t num_x_grid_pts = 6;
    const double x_end = 3.25;
    const size_t num_y_grid_pts = 7;
    const double y_start = -2.5;
    const double y_end = 4.0;

    mesh = Mesh<3>{{{num_x_grid_pts, num_y_grid_pts, 1}},
                   {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
                     Spectral::Basis::Cartoon}},
                   {{Spectral::Quadrature::GaussLobatto,
                     Spectral::Quadrature::GaussLobatto,
                     Spectral::Quadrature::AxialSymmetry}}};

    const Affine affine_x_map(-1.0, 1.0, x_start, x_end);
    const Affine affine_y_map(-1.0, 1.0, y_start, y_end);

    using Cartoon_map_combination =
        domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Identity1D>;
    const domain::CoordinateMap<Frame::ElementLogical, Frame::Grid,
                                Cartoon_map_combination>
        map{{affine_x_map, affine_y_map, identity_cartoon_map}};
    inv_jacobian = map.inv_jacobian(logical_coordinates(mesh));
    coords = map(logical_coordinates(mesh));
  }

  using Tempabc =
      ::Tags::TempTensor<0, tnsr::abc<DataVector, 3, Frame::Inertial>>;

  using VarTags =
      tmpl::list<::Tags::TempScalar<0>, ::Tags::Tempa<0, 3, Frame::Inertial>,
                 ::Tags::TempA<0, 3, Frame::Inertial>,
                 ::Tags::TempAb<0, 3, Frame::Inertial>,
                 ::Tags::Tempab<0, 3, Frame::Inertial>, Tempabc>;

  Variables<VarTags> vars{mesh.number_of_grid_points()};

  using d_VarTags =
      tmpl::transform<VarTags, tmpl::bind<::Tags::deriv, tmpl::_1,
                                          tmpl::size_t<3>, Frame::Grid>>;

  Variables<d_VarTags> expected_d_vars{mesh.number_of_grid_points()};

  // Here we create "prefactors", which serve to fill out our tensors by being
  // multiplied by our cartoon_func(), which themselves make our tensors have
  // some nontrivial spatial derivative
  // The point of these prefactors is to ensure the tensors respect the
  // symmetry of the spacetime: it is not sufficient that each component
  // follows the symmetry, rather the entire tensor must satisfy
  // \mathcal{L}_\xi (tensor) = 0. Each rank has it's own form of prefactor
  Variables<VarTags> prefactor_vars{mesh.number_of_grid_points()};

  using TempiA =
      ::Tags::TempTensor<0, tnsr::iA<DataVector, 3, Frame::Inertial>>;
  using TempiAb =
      ::Tags::TempTensor<0, tnsr::iAb<DataVector, 3, Frame::Inertial>>;
  using Tempiab =
      ::Tags::TempTensor<0, tnsr::iab<DataVector, 3, Frame::Inertial>>;
  using Tempiabc =
      ::Tags::TempTensor<0, TensorMetafunctions::prepend_spatial_index<
                                tnsr::abc<DataVector, 3, Frame::Inertial>, 3,
                                UpLo::Lo, Frame::Inertial>>;

  // prepending a spatial index to VarTags (= PrefactorVarTags if it existed)
  using d_PrefactorVarTags = tmpl::list<::Tags::Tempi<0, 3, Frame::Inertial>,
                                        ::Tags::Tempia<0, 3, Frame::Inertial>,
                                        TempiA, TempiAb, Tempiab, Tempiabc>;

  Variables<d_PrefactorVarTags> d_prefactor_vars{mesh.number_of_grid_points()};

  const size_t dv_size = get<0>(coords).size();

  // in the following, the time component of the indices is arbirarily taken
  // to be 1
  auto& scalar = get<::Tags::TempScalar<0>>(prefactor_vars);
  auto& d_scalar = get<::Tags::Tempi<0, 3>>(d_prefactor_vars);
  // scalar, doing nothing
  scalar.get() = DataVector(dv_size, 1.0);
  // \partial_i of 1 is zero
  for (size_t i = 0; i < index_dim<0>(d_scalar); ++i) {
    d_scalar.get(i) = DataVector(dv_size, 0.0);
  }

  auto& one_form = get<::Tags::Tempa<0, 3>>(prefactor_vars);
  auto& d_one_form = get<::Tags::Tempia<0, 3>>(d_prefactor_vars);
  auto& vector = get<::Tags::TempA<0, 3>>(prefactor_vars);
  auto& d_vector = get<TempiA>(d_prefactor_vars);
  if constexpr (Spherical) {
    // spherical case, vector/one form, using x^a
    get<0>(vector) = get<0>(one_form) = DataVector(dv_size, 1.0);
    for (size_t i = 1; i < index_dim<0>(vector); ++i) {
      vector.get(i) = one_form.get(i) = coords.get(i - 1);
    }
    // partial_i of x_a is \delta_ia
    for (size_t i = 0; i < index_dim<0>(d_vector); ++i) {
      for (size_t a = 0; a < index_dim<1>(d_vector); ++a) {
        if ((i + 1) == a and a != 0) {
          d_vector.get(i, a) = d_one_form.get(i, a) = DataVector(dv_size, 1.0);
        } else {
          d_vector.get(i, a) = d_one_form.get(i, a) = DataVector(dv_size, 0.0);
        }
      }
    }
  } else {
    // axial case, vector/one form, using x^a = (0, -z, 0, x) (pure rotation)
    get<0>(vector) = get<0>(one_form) = DataVector(dv_size, 0.0);
    get<1>(vector) = get<1>(one_form) = -1.0 * coords.get(2);
    get<2>(vector) = get<2>(one_form) = DataVector(dv_size, 0.0);
    get<3>(vector) = get<3>(one_form) = get<0>(coords);

    // \partial_i of x_a is (0, \delta_i2, 0, \delta_i0)
    for (size_t i = 0; i < index_dim<0>(d_vector); ++i) {
      for (size_t a = 0; a < index_dim<1>(d_vector); ++a) {
        if (i == 2 and a == 1) {
          d_vector.get(i, a) = d_one_form.get(i, a) = DataVector(dv_size, -1.0);
        } else if (i == 0 and a == 3) {
          d_vector.get(i, a) = d_one_form.get(i, a) = DataVector(dv_size, 1.0);
        } else {
          d_vector.get(i, a) = d_one_form.get(i, a) = DataVector(dv_size, 0.0);
        }
      }
    }
  }

  auto& mixed = get<::Tags::TempAb<0, 3>>(prefactor_vars);
  auto& d_mixed = get<TempiAb>(d_prefactor_vars);
  auto& rank2 = get<::Tags::Tempab<0, 3>>(prefactor_vars);
  auto& d_rank2 = get<Tempiab>(d_prefactor_vars);
  // filling space with (essenially) projector to tangent space of sphere
  // P_ab = \delta_ab + x_a x_b
  // can have arbitrary function in from of each term, not doing here
  // (the real projector is \delta_ij - x_i x_j / r^2)
  for (size_t a = 0; a < index_dim<0>(rank2); ++a) {
    for (size_t b = 0; b < index_dim<1>(rank2); ++b) {
      if (a == 0 and b == 0) {
        mixed.get(a, b) = rank2.get(a, b) = DataVector(dv_size, 1.0);
      } else if (a == 0) {
        mixed.get(a, b) = rank2.get(a, b) = coords.get(b - 1);
      } else if (b == 0) {
        mixed.get(a, b) = rank2.get(a, b) = coords.get(a - 1);
      } else {
        if (a == b) {
          mixed.get(a, b) = rank2.get(a, b) =
              DataVector(dv_size, 1.0) + coords.get(a - 1) * coords.get(b - 1);
        } else {
          mixed.get(a, b) = rank2.get(a, b) =
              coords.get(a - 1) * coords.get(b - 1);
        }
      }
    }
  }
  // \partial_i of (\delta_ab + x_a x_b) is (x_a \delta_ib + x_b \delta_ia)
  for (size_t i = 0; i < index_dim<0>(d_rank2); ++i) {
    for (size_t a = 0; a < index_dim<1>(d_rank2); ++a) {
      for (size_t b = 0; b < index_dim<2>(d_rank2); ++b) {
        d_mixed.get(i, a, b) = d_rank2.get(i, a, b) = DataVector(dv_size, 0.0);
        if ((i + 1) == b) {
          if (a == 0) {
            d_mixed.get(i, a, b) += 1.0;
            d_rank2.get(i, a, b) += 1.0;
          } else {
            d_mixed.get(i, a, b) += coords.get(a - 1);
            d_rank2.get(i, a, b) += coords.get(a - 1);
          }
        }
        if ((i + 1) == a) {
          if (b == 0) {
            d_mixed.get(i, a, b) += 1.0;
            d_rank2.get(i, a, b) += 1.0;
          } else {
            d_mixed.get(i, a, b) += coords.get(b - 1);
            d_rank2.get(i, a, b) += coords.get(b - 1);
          }
        }
      }
    }
  }

  auto& rank3 = get<Tempabc>(prefactor_vars);
  auto& d_rank3 = get<Tempiabc>(d_prefactor_vars);
  // Rank 3 is important because of \Phi_iab, the spatial derivative of the
  // metric, which though not a tensor mathematically still must obey
  // \mathcal{L}_\xi \Phi_{iab} = 0
  // Using the form
  // x_a x_b x_c + \delta_ab x_c \delta_ac x_b \delta_bc x_a
  for (size_t a = 0; a < index_dim<0>(rank3); ++a) {
    for (size_t b = 0; b < index_dim<1>(rank3); ++b) {
      for (size_t c = 0; c < index_dim<2>(rank3); ++c) {
        rank3.get(a, b, c) =
            (a == 0 ? DataVector(dv_size, 1.0) : coords.get(a - 1)) *
            (b == 0 ? DataVector(dv_size, 1.0) : coords.get(b - 1)) *
            (c == 0 ? DataVector(dv_size, 1.0) : coords.get(c - 1));
        if (a == b) {
          rank3.get(a, b, c) +=
              (c == 0 ? DataVector(dv_size, 1.0) : coords.get(c - 1));
        }
        if (a == c) {
          rank3.get(a, b, c) +=
              (b == 0 ? DataVector(dv_size, 1.0) : coords.get(b - 1));
        }
        if (b == c) {
          rank3.get(a, b, c) +=
              (a == 0 ? DataVector(dv_size, 1.0) : coords.get(a - 1));
        }
      }
    }
  }
  // \partial_i of (x_a x_b x_c + \delta_ab x_c \delta_ac x_b \delta_bc x_a) is
  // \delta_ia x_b x_c + \delta_ib x_a x_c + \delta_ic x_a x_b +
  //   \delta_ab \delta_ic + \delta_ac \delta_ib + \delta_bc \delta_ia
  for (size_t i = 0; i < index_dim<0>(d_rank3); ++i) {
    for (size_t a = 0; a < index_dim<1>(d_rank3); ++a) {
      for (size_t b = 0; b < index_dim<2>(d_rank3); ++b) {
        for (size_t c = 0; c < index_dim<3>(d_rank3); ++c) {
          d_rank3.get(i, a, b, c) = DataVector(dv_size, 0.0);
          if ((i + 1) == a) {
            d_rank3.get(i, a, b, c) +=
                (b == 0 ? DataVector(dv_size, 1.0) : coords.get(b - 1)) *
                (c == 0 ? DataVector(dv_size, 1.0) : coords.get(c - 1));
          }
          if ((i + 1) == b) {
            d_rank3.get(i, a, b, c) +=
                (a == 0 ? DataVector(dv_size, 1.0) : coords.get(a - 1)) *
                (c == 0 ? DataVector(dv_size, 1.0) : coords.get(c - 1));
          }
          if ((i + 1) == c) {
            d_rank3.get(i, a, b, c) +=
                (a == 0 ? DataVector(dv_size, 1.0) : coords.get(a - 1)) *
                (b == 0 ? DataVector(dv_size, 1.0) : coords.get(b - 1));
          }
          if (a == b and (i + 1) == c) {
            d_rank3.get(i, a, b, c) += 1.0;
          }
          if (a == c and (i + 1) == b) {
            d_rank3.get(i, a, b, c) += 1.0;
          }
          if (b == c and (i + 1) == a) {
            d_rank3.get(i, a, b, c) += 1.0;
          }
        }
      }
    }
  }

  tmpl::for_each<VarTags>([&vars, &prefactor_vars, &expected_d_vars,
                           &d_prefactor_vars, &coords]<typename tensor_tag>(
                              tmpl::type_<tensor_tag> /*meta*/) {
    auto& tensor = get<tensor_tag>(vars);
    const auto& prefactor_tensor = get<tensor_tag>(prefactor_vars);
    using deriv_tag = tmpl::apply<
        tmpl::bind<::Tags::deriv, tmpl::_1, tmpl::size_t<3>, Frame::Grid>,
        tensor_tag>;
    auto& d_tensor = get<deriv_tag>(expected_d_vars);
    const auto& d_prefactor_tensor =
        get<tmpl::at<d_PrefactorVarTags, tmpl::index_of<VarTags, tensor_tag>>>(
            d_prefactor_vars);

    for (size_t storage_index = 0; storage_index < tensor.size();
         ++storage_index) {
      tensor[storage_index] =
          prefactor_tensor[storage_index] * cartoon_func<Spherical>(coords);

      const auto input_index = tensor.get_tensor_index(storage_index);
      for (size_t deriv_index = 0; deriv_index < 3; ++deriv_index) {
        const auto output_index = prepend(input_index, size_t{deriv_index});

        d_tensor.get(output_index) =
            prefactor_tensor.get(input_index) *
                cartoon_dfunc<Spherical>(deriv_index, coords) +
            cartoon_func<Spherical>(coords) *
                d_prefactor_tensor.get(output_index);
      }
    }
  });

  const auto to_inertial =
      domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                            domain::CoordinateMaps::Identity<3>>{};
  Variables<d_VarTags> d_vars{mesh.number_of_grid_points()};
  cartoon_partial_derivatives(make_not_null(&d_vars), vars, mesh, inv_jacobian,
                              to_inertial(coords));
  // `d_vars` holds our partial derivatives we want to test against the truth
  const Approx local_approx = Approx::custom().epsilon(1e-10).scale(1.0);
  CHECK_VARIABLES_CUSTOM_APPROX(d_vars, expected_d_vars, local_approx);
}
}  // namespace

// [[Timeout, 60]]
SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.LogicalDerivs",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  constexpr size_t min_points =
      Spectral::minimum_number_of_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>;
  constexpr size_t max_points =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre> / 2;
  for (size_t n0 = min_points; n0 <= max_points; ++n0) {
    // To keep test time reasonable we don't check all possible values.
    if (n0 > 6 and n0 != max_points) {
      continue;
    }
    const Mesh<1> mesh_1d{n0, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
    test_logical_partial_derivatives_1d<two_vars<DataVector, 1>>(mesh_1d);
    test_logical_partial_derivatives_1d<two_vars<DataVector, 1>,
                                        one_var<DataVector, 1>>(mesh_1d);
    test_logical_partial_derivatives_1d<two_vars<ComplexDataVector, 1>>(
        mesh_1d);
    test_logical_partial_derivatives_1d<two_vars<ComplexDataVector, 1>,
                                        one_var<ComplexDataVector, 1>>(mesh_1d);
    for (size_t n1 = min_points; n1 <= max_points; ++n1) {
      // To keep test time reasonable we don't check all possible values.
      if (n1 > 6 and n1 != max_points) {
        continue;
      }
      const Mesh<2> mesh_2d{{{n0, n1}},
                            Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto};
      test_logical_partial_derivatives_2d<two_vars<DataVector, 2>>(mesh_2d);
      test_logical_partial_derivatives_2d<two_vars<DataVector, 2>,
                                          one_var<DataVector, 2>>(mesh_2d);
      test_logical_partial_derivatives_2d<two_vars<ComplexDataVector, 2>>(
          mesh_2d);
      test_logical_partial_derivatives_2d<two_vars<ComplexDataVector, 2>,
                                          one_var<ComplexDataVector, 2>>(
          mesh_2d);
      for (size_t n2 = min_points; n2 <= max_points; ++n2) {
        // To keep test time reasonable we don't check all possible values.
        if (n2 > 6 and n2 != max_points) {
          continue;
        }
        const Mesh<3> mesh_3d{{{n0, n1, n2}},
                              Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto};
        test_logical_partial_derivatives_3d<two_vars<DataVector, 3>>(mesh_3d);
        test_logical_partial_derivatives_3d<two_vars<DataVector, 3>,
                                            one_var<DataVector, 3>>(mesh_3d);
        test_logical_partial_derivatives_3d<two_vars<ComplexDataVector, 3>>(
            mesh_3d);
        test_logical_partial_derivatives_3d<two_vars<ComplexDataVector, 3>,
                                            one_var<ComplexDataVector, 3>>(
            mesh_3d);
      }
    }
  }
  for (size_t n_r = 1; n_r < 5; ++n_r) {
    for (size_t L = 2; L < 5; ++L) {
      test_logical_partial_derivatives_spherical_shell<0, 0,
                                                       two_vars<DataVector, 3>>(
          n_r, L);
      test_logical_partial_derivatives_spherical_shell<1, 0,
                                                       two_vars<DataVector, 3>>(
          n_r, L);
      test_logical_partial_derivatives_spherical_shell<1, 1,
                                                       two_vars<DataVector, 3>>(
          n_r, L);
      test_logical_partial_derivatives_spherical_shell<1, -1,
                                                       two_vars<DataVector, 3>>(
          n_r, L);
      if (L > 2) {
        test_logical_partial_derivatives_spherical_shell<
            2, 0, two_vars<DataVector, 3>>(n_r, L);
        test_logical_partial_derivatives_spherical_shell<
            2, 1, two_vars<DataVector, 3>>(n_r, L);
        test_logical_partial_derivatives_spherical_shell<
            2, -1, two_vars<DataVector, 3>>(n_r, L);
        test_logical_partial_derivatives_spherical_shell<
            2, 2, two_vars<DataVector, 3>>(n_r, L);
        test_logical_partial_derivatives_spherical_shell<
            2, -2, two_vars<DataVector, 3>>(n_r, L);
      }
    }
  }
}

// [[Timeout, 90]]
SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.PartialDerivs",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  test_partial_derivatives_spherical_shell<two_vars<DataVector, 3>>();
  test_partial_derivatives_spherical_shell<two_vars<DataVector, 3>,
                                           one_var<DataVector, 3>>();
  const size_t n0 =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre> / 2;
  const size_t n1 =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre> / 2 + 1;
  const size_t n2 =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre> / 2 - 1;
  const Mesh<1> mesh_1d{n0, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  test_partial_derivatives_1d<two_vars<DataVector, 1>>(mesh_1d);
  test_partial_derivatives_1d<two_vars<DataVector, 1>, one_var<DataVector, 1>>(
      mesh_1d);
  test_partial_derivatives_1d<two_vars<ComplexDataVector, 1>>(mesh_1d);
  const Mesh<2> mesh_2d{{{n0, n1}},
                        Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  test_partial_derivatives_2d<two_vars<DataVector, 2>>(mesh_2d);
  test_partial_derivatives_2d<two_vars<DataVector, 2>, one_var<DataVector, 2>>(
      mesh_2d);
  test_partial_derivatives_2d<two_vars<ComplexDataVector, 2>,
                              one_var<ComplexDataVector, 2>>(mesh_2d);
  const Mesh<3> mesh_3d{{{n0, n1, n2}},
                        Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  test_partial_derivatives_3d<two_vars<DataVector, 3>>(mesh_3d);
  test_partial_derivatives_3d<two_vars<DataVector, 3>, one_var<DataVector, 3>>(
      mesh_3d);
  test_partial_derivatives_3d<two_vars<ComplexDataVector, 3>,
                              one_var<ComplexDataVector, 3>>(mesh_3d);

  // testing with L'Hopital
  test_cartoon<true>(0.0);
  test_cartoon<false>(0.0);
  // testing without L'Hopital
  test_cartoon<true>(1.0);
  test_cartoon<false>(1.0);

  TestHelpers::db::test_prefix_tag<
      Tags::deriv<Var1<DataVector, 3>, tmpl::size_t<3>, Frame::Grid>>(
      "deriv(Var1)");
  TestHelpers::db::test_prefix_tag<
      Tags::spacetime_deriv<Var1<DataVector, 3>, tmpl::size_t<3>, Frame::Grid>>(
      "spacetime_deriv(Var1)");

  BENCHMARK_ADVANCED("Partial derivatives")
  (Catch::Benchmark::Chronometer meter) {
    const Mesh<3> mesh{4, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const Affine map1d(-1.0, 1.0, -1.0, 1.0);
    const domain::CoordinateMap<Frame::ElementLogical, Frame::Grid, Affine3D>
        map(Affine3D{map1d, map1d, map1d});
    const auto inv_jacobian = map.inv_jacobian(logical_coordinates(mesh));
    const Variables<tmpl::list<Var1<DataVector, 3>, Var2<DataVector>>> u{
        mesh.number_of_grid_points(), 0.0};
    Variables<tmpl::list<
        ::Tags::deriv<Var1<DataVector, 3>, tmpl::size_t<3>, Frame::Grid>,
        ::Tags::deriv<Var2<DataVector>, tmpl::size_t<3>, Frame::Grid>>>
        du{mesh.number_of_grid_points()};
    meter.measure([&du, &u, &mesh, &inv_jacobian]() {
      partial_derivatives(make_not_null(&du), u, mesh, inv_jacobian);
    });
  };

  BENCHMARK_ADVANCED("Partial derivatives complex")
  (Catch::Benchmark::Chronometer meter) {
    const Mesh<3> mesh{4, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const Affine map1d(-1.0, 1.0, -1.0, 1.0);
    const domain::CoordinateMap<Frame::ElementLogical, Frame::Grid, Affine3D>
        map(Affine3D{map1d, map1d, map1d});
    const auto inv_jacobian = map.inv_jacobian(logical_coordinates(mesh));
    const Variables<
        tmpl::list<Var1<ComplexDataVector, 3>, Var2<ComplexDataVector>>>
        u{mesh.number_of_grid_points(), 0.0};
    Variables<tmpl::list<
        ::Tags::deriv<Var1<ComplexDataVector, 3>, tmpl::size_t<3>, Frame::Grid>,
        ::Tags::deriv<Var2<ComplexDataVector>, tmpl::size_t<3>, Frame::Grid>>>
        du{mesh.number_of_grid_points()};
    meter.measure([&du, &u, &mesh, &inv_jacobian]() {
      partial_derivatives(make_not_null(&du), u, mesh, inv_jacobian);
    });
  };
}

namespace {
template <class MapType>
struct MapTag : db::SimpleTag {
  static constexpr size_t dim = MapType::dim;
  using target_frame = typename MapType::target_frame;
  using source_frame = typename MapType::source_frame;

  using type = MapType;
};

template <typename Tag>
struct SomePrefix : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
  static std::string name() {
    return "SomePrefix(" + db::tag_name<Tag>() + ")";
  }
};

template <size_t Dim, typename T>
void test_partial_derivatives_compute_item(
    const std::array<size_t, Dim> extents_array, const T& map) {
  using vars_tags = tmpl::list<Var1<DataVector, Dim>, Var2<DataVector>>;
  using map_tag = MapTag<std::decay_t<decltype(map)>>;
  using inv_jac_tag = domain::Tags::InverseJacobianCompute<
      map_tag, typename domain::Tags::LogicalCoordinates<Dim>::base>;
  using deriv_tag =
      Tags::DerivCompute<Tags::Variables<vars_tags>, domain::Tags::Mesh<Dim>,
                         typename inv_jac_tag::base>;
  using prefixed_variables_tag =
      db::add_tag_prefix<SomePrefix, Tags::Variables<vars_tags>>;
  using deriv_prefixed_tag =
      Tags::DerivCompute<prefixed_variables_tag, domain::Tags::Mesh<Dim>,
                         typename inv_jac_tag::base,
                         tmpl::list<SomePrefix<Var1<DataVector, Dim>>>>;

  TestHelpers::db::test_compute_tag<deriv_tag>(
      "Variables(deriv(Var1),deriv(Var2))");

  const std::array<size_t, Dim> array_to_functions{extents_array -
                                                   make_array<Dim>(size_t{1})};
  const Mesh<Dim> mesh{extents_array, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const size_t num_grid_points = mesh.number_of_grid_points();
  Variables<vars_tags> u(num_grid_points);
  Variables<
      db::wrap_tags_in<Tags::deriv, vars_tags, tmpl::size_t<Dim>, Frame::Grid>>
      expected_du(num_grid_points);
  const auto x_logical = logical_coordinates(mesh);
  const auto x = map(logical_coordinates(mesh));

  tmpl::for_each<vars_tags>([&array_to_functions, &x, &u](auto tag) {
    using Tag = tmpl::type_from<decltype(tag)>;
    get<Tag>(u) = Tag::f(array_to_functions, x);
  });
  typename prefixed_variables_tag::type prefixed_vars(u);

  tmpl::for_each<vars_tags>([&array_to_functions, &x, &expected_du](auto tag) {
    using Tag = typename decltype(tag)::type;
    using DerivativeTag = Tags::deriv<Tag, tmpl::size_t<Dim>, Frame::Grid>;
    get<DerivativeTag>(expected_du) = Tag::df(array_to_functions, x);
  });

  auto box = db::create<
      db::AddSimpleTags<domain::Tags::Mesh<Dim>, Tags::Variables<vars_tags>,
                        prefixed_variables_tag, map_tag>,
      db::AddComputeTags<domain::Tags::LogicalCoordinates<Dim>, inv_jac_tag,
                         deriv_tag, deriv_prefixed_tag>>(mesh, u, prefixed_vars,
                                                         map);

  const auto& du = db::get<typename deriv_tag::base>(box);

  for (size_t n = 0; n < du.size(); ++n) {
    // clang-tidy: pointer arithmetic
    CHECK(du.data()[n] == approx(expected_du.data()[n]));  // NOLINT
  }

  // Test prefixes are handled correctly
  const auto& du_prefixed_vars = get<db::add_tag_prefix<
      Tags::deriv,
      db::add_tag_prefix<SomePrefix,
                         Tags::Variables<tmpl::list<Var1<DataVector, Dim>>>>,
      tmpl::size_t<Dim>, Frame::Grid>>(box);
  const auto& du_prefixed =
      get<Tags::deriv<SomePrefix<Var1<DataVector, Dim>>, tmpl::size_t<Dim>,
                      Frame::Grid>>(du_prefixed_vars);
  const auto& expected_du_prefixed =
      get<Tags::deriv<Var1<DataVector, Dim>, tmpl::size_t<Dim>, Frame::Grid>>(
          expected_du);
  CHECK_ITERABLE_APPROX(du_prefixed, expected_du_prefixed);
}

template <size_t Dim, typename T>
void test_partial_derivatives_tensor_compute_item(
    const std::array<size_t, Dim> extents_array, const T& map) {
  using tensor_tag = Var1<DataVector, Dim>;
  using map_tag = MapTag<std::decay_t<decltype(map)>>;
  using inv_jac_tag = domain::Tags::InverseJacobianCompute<
      map_tag, typename domain::Tags::LogicalCoordinates<Dim>::base>;
  using deriv_tensor_tag =
      Tags::DerivTensorCompute<tensor_tag, typename inv_jac_tag::base,
                               domain::Tags::Mesh<Dim>>;

  const std::array<size_t, Dim> array_to_functions{extents_array -
                                                   make_array<Dim>(size_t{1})};
  const Mesh<Dim> mesh{extents_array, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const auto x_logical = logical_coordinates(mesh);
  const auto x = map(logical_coordinates(mesh));

  const auto u = tensor_tag::f(array_to_functions, x);
  const auto expected_du = tensor_tag::df(array_to_functions, x);

  auto box = db::create<
      db::AddSimpleTags<domain::Tags::Mesh<Dim>, tensor_tag, map_tag>,
      db::AddComputeTags<domain::Tags::LogicalCoordinates<Dim>, inv_jac_tag,
                         deriv_tensor_tag>>(mesh, u, map);

  const auto& du = db::get<typename deriv_tensor_tag::base>(box);

  // CHECK_ITERABLE_APPROX(du, expected_du.data());
  for (size_t n = 0; n < du.size(); ++n) {
    CHECK_ITERABLE_APPROX(du[n], expected_du[n]);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.PartialDerivs.ComputeItems",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  Index<3> max_extents{10, 10, 5};

  for (size_t a = 1; a < max_extents[0]; ++a) {
    test_partial_derivatives_compute_item(
        std::array<size_t, 1>{{a + 1}},
        domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
            Affine{-1.0, 1.0, -0.3, 0.7}));
    for (size_t b = 1; b < max_extents[1]; ++b) {
      test_partial_derivatives_compute_item(
          std::array<size_t, 2>{{a + 1, b + 1}},
          domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
              Affine2D{Affine{-1.0, 1.0, -0.3, 0.7},
                       Affine{-1.0, 1.0, 0.3, 0.55}}));
      for (size_t c = 1; a < max_extents[0] / 2 and b < max_extents[1] / 2 and
                         c < max_extents[2];
           ++c) {
        test_partial_derivatives_compute_item(
            std::array<size_t, 3>{{a + 1, b + 1, c + 1}},
            domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
                Affine3D{Affine{-1.0, 1.0, -0.3, 0.7},
                         Affine{-1.0, 1.0, 0.3, 0.55},
                         Affine{-1.0, 1.0, 2.3, 2.8}}));
      }
    }
  }
  for (size_t a = 1; a < max_extents[0]; ++a) {
    test_partial_derivatives_tensor_compute_item(
        std::array<size_t, 1>{{a + 1}},
        domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
            Affine{-1.0, 1.0, -0.3, 0.7}));
    for (size_t b = 1; b < max_extents[1]; ++b) {
      test_partial_derivatives_tensor_compute_item(
          std::array<size_t, 2>{{a + 1, b + 1}},
          domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
              Affine2D{Affine{-1.0, 1.0, -0.3, 0.7},
                       Affine{-1.0, 1.0, 0.3, 0.55}}));
      for (size_t c = 1; a < max_extents[0] / 2 and b < max_extents[1] / 2 and
                         c < max_extents[2];
           ++c) {
        test_partial_derivatives_tensor_compute_item(
            std::array<size_t, 3>{{a + 1, b + 1, c + 1}},
            domain::make_coordinate_map<Frame::ElementLogical, Frame::Grid>(
                Affine3D{Affine{-1.0, 1.0, -0.3, 0.7},
                         Affine{-1.0, 1.0, 0.3, 0.55},
                         Affine{-1.0, 1.0, 2.3, 2.8}}));
      }
    }
  }
}
