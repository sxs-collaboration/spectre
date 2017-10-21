// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/AffineMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/GridCoordinates.hpp"
#include "Numerical/LinearOperators/PartialDerivatives.hpp"
#include "Numerical/Spectral/LegendreGaussLobatto.hpp"
#include "Utilities/MakeArray.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

template <size_t Dim>
struct Var1 : db::DataBoxTag {
  using type = tnsr::i<DataVector, Dim, Frame::Grid>;
  static constexpr db::DataBoxString_t label = "Vector_t";
  static auto f(const std::array<size_t, Dim>& coeffs,
                const tnsr::I<DataVector, Dim, Frame::Grid>& x) {
    tnsr::i<DataVector, Dim, Frame::Grid> result(x.begin()->size(), 0.);
    for (size_t i = 0; i < Dim; ++i) {
      result.get(i) = (i + 2);
      for (size_t d = 0; d < Dim; ++d) {
        result.get(i) *= pow(x.get(d), gsl::at(coeffs, d));
      }
    }
    return result;
  }
  static auto df(const std::array<size_t, Dim>& coeffs,
                 const tnsr::I<DataVector, Dim, Frame::Grid>& x) {
    tnsr::ij<DataVector, Dim, Frame::Grid> result(x.begin()->size(), 0.);
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
    return result;
  }
};

struct Var2 : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString_t label = "Scalar_t";
  template <size_t Dim>
  static auto f(const std::array<size_t, Dim>& coeffs,
                const tnsr::I<DataVector, Dim, Frame::Grid>& x) {
    Scalar<DataVector> result(x.begin()->size(), 1.);
    for (size_t d = 0; d < Dim; ++d) {
      result.get() *= pow(x.get(d), gsl::at(coeffs, d));
    }
    return result;
  }
  template <size_t Dim>
  static auto df(const std::array<size_t, Dim>& coeffs,
                 const tnsr::I<DataVector, Dim, Frame::Grid>& x) {
    tnsr::i<DataVector, Dim, Frame::Grid> result(x.begin()->size(), 1.);
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

template <size_t Dim>
using two_vars = typelist<Var1<Dim>, Var2>;

template <size_t Dim>
using one_var = typelist<Var1<Dim>>;

template <typename VariableTags, typename GradientTags = VariableTags>
void test_logical_partial_derivatives_1d(const Index<1>& extents) {
  const size_t number_of_grid_points = extents.product();
  const DataVector& xi = Basis::lgl::collocation_points(extents[0]);
  Variables<VariableTags> u(number_of_grid_points);
  for (size_t a = 0; a < extents[0]; ++a) {
    for (size_t n = 0; n < u.number_of_independent_components; ++n) {
      for (size_t s = 0; s < number_of_grid_points; ++s) {
        u.data()[s + n * number_of_grid_points]  // NOLINT
            = (n + 1) * pow(xi[s], a);
      }
    }

    const auto du = logical_partial_derivatives<GradientTags>(u, extents);

    for (size_t n = 0;
         n < Variables<GradientTags>::number_of_independent_components; ++n) {
      for (size_t s = 0; s < number_of_grid_points; ++s) {
        const double expected =
            (0 == a ? 0.0 : a * (n + 1) * pow(xi[s], a - 1));
        CHECK(du[0].data()[s + n * number_of_grid_points]  // NOLINT
              == approx(expected));
      }
    }
  }
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_logical_partial_derivatives_2d(const Index<2>& extents) {
  const size_t number_of_grid_points = extents.product();
  const DataVector& xi = Basis::lgl::collocation_points(extents[0]);
  const DataVector& eta = Basis::lgl::collocation_points(extents[1]);
  Variables<VariableTags> u(number_of_grid_points);
  const size_t a = extents[0] - 1;
  const size_t b = extents[1] - 1;
  for (size_t n = 0; n < u.number_of_independent_components; ++n) {
    for (IndexIterator<2> ii(extents); ii; ++ii) {
      u.data()[ii.offset() + n * number_of_grid_points] =  // NOLINT
          (n + 1) * pow(xi[ii()[0]], a) * pow(eta[ii()[1]], b);
    }
  }

  const auto du = logical_partial_derivatives<GradientTags>(u, extents);

  for (size_t n = 0;
       n < Variables<GradientTags>::number_of_independent_components; ++n) {
    for (IndexIterator<2> ii(extents); ii; ++ii) {
      const double expected_dxi =
          (0 == a
               ? 0.0
               : a * (n + 1) * pow(xi[ii()[0]], a - 1) * pow(eta[ii()[1]], b));
      const double expected_deta = (0 == b ? 0.0
                                           : b * (n + 1) * pow(xi[ii()[0]], a) *
                                                 pow(eta[ii()[1]], b - 1));
      CHECK(du[0].data()[ii.offset() + n * number_of_grid_points] ==  // NOLINT
            approx(expected_dxi));
      CHECK(du[1].data()[ii.offset() + n * number_of_grid_points] ==  // NOLINT
            approx(expected_deta));
    }
  }
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_logical_partial_derivatives_3d(const Index<3>& extents) {
  const size_t number_of_grid_points = extents.product();
  const DataVector& xi = Basis::lgl::collocation_points(extents[0]);
  const DataVector& eta = Basis::lgl::collocation_points(extents[1]);
  const DataVector& zeta = Basis::lgl::collocation_points(extents[2]);
  Variables<VariableTags> u(number_of_grid_points);
  const size_t a = extents[0] - 1;
  const size_t b = extents[1] - 1;
  const size_t c = extents[2] - 1;
  for (size_t n = 0; n < u.number_of_independent_components; ++n) {
    for (IndexIterator<3> ii(extents); ii; ++ii) {
      u.data()[ii.offset() + n * number_of_grid_points] =  // NOLINT
          (n + 1) * pow(xi[ii()[0]], a) * pow(eta[ii()[1]], b) *
          pow(zeta[ii()[2]], c);
    }
  }

  const auto du = logical_partial_derivatives<GradientTags>(u, extents);

  for (size_t n = 0;
       n < Variables<GradientTags>::number_of_independent_components; ++n) {
    for (IndexIterator<3> ii(extents); ii; ++ii) {
      const double expected_dxi =
          (0 == a ? 0.0
                  : a * (n + 1) * pow(xi[ii()[0]], a - 1) *
                        pow(eta[ii()[1]], b) * pow(zeta[ii()[2]], c));
      const double expected_deta =
          (0 == b ? 0.0
                  : b * (n + 1) * pow(xi[ii()[0]], a) *
                        pow(eta[ii()[1]], b - 1) * pow(zeta[ii()[2]], c));
      const double expected_dzeta =
          (0 == c ? 0.0
                  : c * (n + 1) * pow(xi[ii()[0]], a) * pow(eta[ii()[1]], b) *
                        pow(zeta[ii()[2]], c - 1));
      CHECK(du[0].data()[ii.offset() + n * number_of_grid_points] ==  // NOLINT
            approx(expected_dxi));
      CHECK(du[1].data()[ii.offset() + n * number_of_grid_points] ==  // NOLINT
            approx(expected_deta));
      CHECK(du[2].data()[ii.offset() + n * number_of_grid_points] ==  // NOLINT
            approx(expected_dzeta));
    }
  }
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_partial_derivatives_1d(const Index<1>& extents) {
  const size_t number_of_grid_points = extents.product();
  const CoordinateMaps::AffineMap x_map{-1.0, 1.0, -0.3, 0.7};
  const auto map_1d = make_coordinate_map<Frame::Logical, Frame::Grid>(
      CoordinateMaps::AffineMap{x_map});
  const auto x = map_1d(logical_coordinates(extents));
  const InverseJacobian<1, Frame::Logical, Frame::Grid> inverse_jacobian(
      extents.product(), 2.0);

  Variables<VariableTags> u(number_of_grid_points);
  Variables<
      db::wrap_tags_in<Tags::d, GradientTags, tmpl::size_t<1>, Frame::Grid>>
      expected_du(number_of_grid_points);
  for (size_t a = 0; a < extents[0]; ++a) {
    tmpl::for_each<VariableTags>([&a, &x, &u ](auto tag) noexcept {
      using Tag = tmpl::type_from<decltype(tag)>;
      get<Tag>(u) = Tag::f({{a}}, x);
    });
    tmpl::for_each<GradientTags>([&a, &x, &expected_du ](auto tag) noexcept {
      using Tag = typename decltype(tag)::type;
      using DerivativeTag = Tags::d<Tag, tmpl::size_t<1>, Frame::Grid>;
      get<DerivativeTag>(expected_du) = Tag::df({{a}}, x);
    });

    const auto du =
        partial_derivatives<GradientTags>(u, extents, inverse_jacobian);

    for (size_t n = 0; n < du.size(); ++n) {
      CAPTURE_PRECISE(du.data()[n] - expected_du.data()[n]);  // NOLINT
      CHECK(du.data()[n] == approx(expected_du.data()[n]));   // NOLINT
    }
  }
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_partial_derivatives_2d(const Index<2>& extents) {
  using affine_map = CoordinateMaps::AffineMap;
  using affine_map_2d = CoordinateMaps::ProductOf2Maps<affine_map, affine_map>;
  const size_t number_of_grid_points = extents.product();
  const auto prod_map2d =
      make_coordinate_map<Frame::Logical, Frame::Grid>(affine_map_2d{
          affine_map{-1.0, 1.0, -0.3, 0.7}, affine_map{-1.0, 1.0, 0.3, 0.55}});
  const auto x = prod_map2d(logical_coordinates(extents));
  InverseJacobian<2, Frame::Logical, Frame::Grid> inverse_jacobian(
      number_of_grid_points, 0.0);
  inverse_jacobian.get(0, 0) = 2.0;
  inverse_jacobian.get(1, 1) = 8.0;

  Variables<VariableTags> u(number_of_grid_points);
  Variables<
      db::wrap_tags_in<Tags::d, GradientTags, tmpl::size_t<2>, Frame::Grid>>
      expected_du(number_of_grid_points);
  for (size_t a = 0; a < extents[0]; ++a) {
    for (size_t b = 0; b < extents[1]; ++b) {
      tmpl::for_each<VariableTags>([&a, &b, &x, &u ](auto tag) noexcept {
        using Tag = typename decltype(tag)::type;
        get<Tag>(u) = Tag::f({{a, b}}, x);
      });
      tmpl::for_each<GradientTags>(
          [&a, &b, &x, &expected_du ](auto tag) noexcept {
            using Tag = typename decltype(tag)::type;
            using DerivativeTag = Tags::d<Tag, tmpl::size_t<2>, Frame::Grid>;
            get<DerivativeTag>(expected_du) = Tag::df({{a, b}}, x);
          });

      const auto du =
          partial_derivatives<GradientTags>(u, extents, inverse_jacobian);

      for (size_t n = 0; n < du.size(); ++n) {
        CAPTURE_PRECISE(du.data()[n] - expected_du.data()[n]);  // NOLINT
        CHECK(du.data()[n] ==                                   // NOLINT
              approx(expected_du.data()[n]).epsilon(1.e-13));   // NOLINT
      }
    }
  }
}

template <typename VariableTags, typename GradientTags = VariableTags>
void test_partial_derivatives_3d(const Index<3>& extents) {
  using affine_map = CoordinateMaps::AffineMap;
  using affine_map_3d =
      CoordinateMaps::ProductOf3Maps<affine_map, affine_map, affine_map>;
  const size_t number_of_grid_points = extents.product();
  const auto prod_map3d =
      make_coordinate_map<Frame::Logical, Frame::Grid>(affine_map_3d{
          affine_map{-1.0, 1.0, -0.3, 0.7}, affine_map{-1.0, 1.0, 0.3, 0.55},
          affine_map{-1.0, 1.0, 2.3, 2.8}});
  const auto x = prod_map3d(logical_coordinates(extents));
  InverseJacobian<3, Frame::Logical, Frame::Grid> inverse_jacobian(
      extents.product(), 0.0);
  inverse_jacobian.get(0, 0) = 2.0;
  inverse_jacobian.get(1, 1) = 8.0;
  inverse_jacobian.get(2, 2) = 4.0;

  Variables<VariableTags> u(number_of_grid_points);
  Variables<
      db::wrap_tags_in<Tags::d, GradientTags, tmpl::size_t<3>, Frame::Grid>>
      expected_du(number_of_grid_points);
  for (size_t a = 0; a < extents[0] / 2; ++a) {
    for (size_t b = 0; b < extents[1] / 2; ++b) {
      for (size_t c = 0; c < extents[2] / 2; ++c) {
        tmpl::for_each<VariableTags>([&a, &b, &c, &x, &u ](auto tag) noexcept {
          using Tag = typename decltype(tag)::type;
          get<Tag>(u) = Tag::f({{a, b, c}}, x);
        });
        tmpl::for_each<GradientTags>(
            [&a, &b, &c, &x, &expected_du ](auto tag) noexcept {
              using Tag = typename decltype(tag)::type;
              using DerivativeTag = Tags::d<Tag, tmpl::size_t<3>, Frame::Grid>;
              get<DerivativeTag>(expected_du) = Tag::df({{a, b, c}}, x);
            });

        const auto du =
            partial_derivatives<GradientTags>(u, extents, inverse_jacobian);

        for (size_t n = 0; n < du.size(); ++n) {
          CAPTURE_PRECISE(du.data()[n] - expected_du.data()[n]);  // NOLINT
          CHECK(du.data()[n] ==                                   // NOLINT
                approx(expected_du.data()[n]).epsilon(1.e-11));
        }
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.LogicalDerivs",
                  "[Numerical][LinearOperators][Unit]") {
  for (size_t n0 = 2; n0 <= Basis::lgl::maximum_number_of_pts / 2; ++n0) {
    const Index<1> extents_1d(n0);
    test_logical_partial_derivatives_1d<two_vars<1>>(extents_1d);
    test_logical_partial_derivatives_1d<two_vars<1>, one_var<1>>(extents_1d);
    for (size_t n1 = 2; n1 <= Basis::lgl::maximum_number_of_pts / 2; ++n1) {
      const Index<2> extents_2d(n0, n1);
      test_logical_partial_derivatives_2d<two_vars<2>>(extents_2d);
      test_logical_partial_derivatives_2d<two_vars<2>, one_var<2>>(extents_2d);
      for (size_t n2 = 2; n2 <= Basis::lgl::maximum_number_of_pts / 2; ++n2) {
        const Index<3> extents_3d(n0, n1, n2);
        test_logical_partial_derivatives_3d<two_vars<3>>(extents_3d);
        test_logical_partial_derivatives_3d<two_vars<3>, one_var<3>>(
            extents_3d);
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.PartialDerivs",
                  "[Numerical][LinearOperators][Unit]") {
  const size_t n0 = Basis::lgl::maximum_number_of_pts / 2;
  const size_t n1 = Basis::lgl::maximum_number_of_pts / 2 + 1;
  const size_t n2 = Basis::lgl::maximum_number_of_pts / 2 - 1;
  const Index<1> extents_1d(n0);
  test_partial_derivatives_1d<two_vars<1>>(extents_1d);
  test_partial_derivatives_1d<two_vars<1>, one_var<1>>(extents_1d);
  const Index<2> extents_2d(n0, n1);
  test_partial_derivatives_2d<two_vars<2>>(extents_2d);
  test_partial_derivatives_2d<two_vars<2>, one_var<2>>(extents_2d);
  const Index<3> extents_3d(n0, n1, n2);
  test_partial_derivatives_3d<two_vars<3>>(extents_3d);
  test_partial_derivatives_3d<two_vars<3>, one_var<3>>(extents_3d);
}
