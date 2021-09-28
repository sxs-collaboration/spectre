// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <random>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Tov.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/MathFunctions/PowX.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/MathFunctions/TensorProduct.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
// IWYU pragma: no_forward_declare MathFunction
// IWYU pragma: no_forward_declare PowX
// IWYU pragma: no_forward_declare Tensor

namespace {

using Affine = domain::CoordinateMaps::Affine;
using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

const double inertial_coord_min = -0.3;
const double inertial_coord_max = 0.7;

template <size_t VolumeDim>
auto make_affine_map();

template <>
auto make_affine_map<1>() {
  return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
      Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max});
}

template <>
auto make_affine_map<2>() {
  return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
      Affine2D{Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max},
               Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max}});
}

template <>
auto make_affine_map<3>() {
  return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
      Affine3D{Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max},
               Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max},
               Affine{-1.0, 1.0, inertial_coord_min, inertial_coord_max}});
}

namespace TestTags {

template <size_t Dim>
struct Vector : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
  static auto fill_values(const MathFunctions::TensorProduct<Dim>& f,
                          const tnsr::I<DataVector, Dim>& x) {
    auto result = make_with_value<tnsr::I<DataVector, Dim>>(x, 0.);
    const auto f_of_x = f(x);
    for (size_t d = 0; d < Dim; ++d) {
      result.get(d) = (d + 0.5) * get(f_of_x);
    }
    return result;
  }
};

template <size_t Dim>
struct SymmetricTensor : db::SimpleTag {
  using type = tnsr::ii<DataVector, Dim>;
  static auto fill_values(const MathFunctions::TensorProduct<Dim>& f,
                          const tnsr::I<DataVector, Dim>& x) {
    auto result = make_with_value<tnsr::ii<DataVector, Dim>>(x, 0.);
    const auto f_of_x = f(x);
    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = i; j < Dim; ++j) {  // Symmetry
        result.get(i, j) = (static_cast<double>(i + j) + 0.33) * get(f_of_x);
      }
    }
    return result;
  }
};

}  // namespace TestTags

template <size_t Dim>
void test_interpolate_to_points(const Mesh<Dim>& mesh) {
  // Fill target interpolation coordinates with random values
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(inertial_coord_min, inertial_coord_max);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_dist = make_not_null(&dist);

  const size_t number_of_points = 6;
  const auto target_x_inertial =
      make_with_random_values<tnsr::I<DataVector, Dim>>(
          nn_generator, nn_dist, DataVector(number_of_points));

  const auto coordinate_map = make_affine_map<Dim>();
  const auto target_x = [&target_x_inertial, &coordinate_map,
                         &number_of_points]() {
    tnsr::I<DataVector, Dim, Frame::ElementLogical> result(number_of_points);
    for (size_t s = 0; s < number_of_points; ++s) {
      tnsr::I<double, Dim> x_inertial_local{};
      for (size_t d = 0; d < Dim; ++d) {
        x_inertial_local.get(d) = target_x_inertial.get(d)[s];
      }
      const auto x_local = coordinate_map.inverse(x_inertial_local).value();
      for (size_t d = 0; d < Dim; ++d) {
        result.get(d)[s] = x_local.get(d);
      }
    }
    return result;
  }();

  // Set up interpolator. Need do this only once.
  const intrp::Irregular<Dim> irregular_interpolant(mesh, target_x);
  test_serialization(irregular_interpolant);

  // ... but we construct another interpolator to test operator!=
  {
    auto target_x_new = target_x;
    target_x_new.get(0)[0] *= 0.98;  // Change one point slightly.
    const intrp::Irregular<Dim> irregular_interpolant_new(mesh, target_x_new);
    CHECK(irregular_interpolant_new != irregular_interpolant);
  }

  // Coordinates on the grid
  const auto src_x = coordinate_map(logical_coordinates(mesh));

  // Set up variables
  using tags =
      tmpl::list<TestTags::Vector<Dim>, TestTags::SymmetricTensor<Dim>>;
  Variables<tags> src_vars(mesh.number_of_grid_points());
  Variables<tags> expected_dest_vars(number_of_points);

  // We will make polynomials of the form x^a y^b z^c ...
  // for all a,b,c, that result in exact interpolation.
  // IndexIterator loops over "a,b,c"
  for (IndexIterator<Dim> iter(mesh.extents()); iter; ++iter) {
    // Set up analytic solution.  We fill a Variables with this solution,
    // interpolate to arbitrary points, and then check that the
    // values at arbitrary points match this solution.
    // We choose polynomials so that interpolation is exact on an LGL grid.
    std::array<std::unique_ptr<MathFunction<1, Frame::Inertial>>, Dim>
        functions;
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(functions, d) =
          std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(iter()[d]);
    }
    MathFunctions::TensorProduct<Dim> f(1.0, std::move(functions));

    // Fill source and expected destination Variables with analytic solution.
    tmpl::for_each<tags>([&f, &src_x, &target_x_inertial, &src_vars,
                          &expected_dest_vars](auto tag) {
      using Tag = tmpl::type_from<decltype(tag)>;
      get<Tag>(src_vars) = Tag::fill_values(f, src_x);
      get<Tag>(expected_dest_vars) = Tag::fill_values(f, target_x_inertial);
    });

    // Interpolate
    // (g++ 7.2.0 does not allow `const auto dest_vars` here)
    const Variables<tags> dest_vars =
        irregular_interpolant.interpolate(src_vars);

    tmpl::for_each<tags>([&dest_vars, &expected_dest_vars](auto tag) {
      using Tag = tmpl::type_from<decltype(tag)>;
      CHECK_ITERABLE_APPROX(get<Tag>(dest_vars), get<Tag>(expected_dest_vars));
    });
  }
}

template <Spectral::Basis Basis, Spectral::Quadrature Quadrature>
void test_irregular_interpolant() {
  const size_t start_points = 4;
  const size_t end_points = 6;
  for (size_t n0 = start_points; n0 < end_points; ++n0) {
    test_interpolate_to_points<1>(Mesh<1>{n0, Basis, Quadrature});
    for (size_t n1 = start_points; n1 < end_points; ++n1) {
      test_interpolate_to_points<2>(Mesh<2>{{{n0, n1}}, Basis, Quadrature});
      for (size_t n2 = start_points; n2 < end_points; ++n2) {
        test_interpolate_to_points<3>(
            Mesh<3>{{{n0, n1, n2}}, Basis, Quadrature});
      }
    }
  }
}

void test_irregular_interpolant_mixed_quadrature() {
  const size_t start_points = 4;
  const size_t end_points = 6;
  for (size_t n0 = start_points; n0 < end_points; ++n0) {
    for (size_t n1 = start_points; n1 < end_points; ++n1) {
      test_interpolate_to_points<2>(Mesh<2>{
          {{n0, n1}},
          {{Spectral::Basis::Legendre, Spectral::Basis::Legendre}},
          {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}}});
      for (size_t n2 = start_points; n2 < end_points; ++n2) {
        test_interpolate_to_points<3>(Mesh<3>{
            {{n0, n1, n2}},
            {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
              Spectral::Basis::Legendre}},
            {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
              Spectral::Quadrature::GaussLobatto}}});
      }
    }
  }
}

template <size_t Dim>
Domain<Dim> create_domain(double length,
                          const std::array<size_t, Dim>& extents);

template <>
Domain<3> create_domain<3>(const double length,
                           const std::array<size_t, 3>& extents) {
  const domain::creators::Brick creator{{{0.0, 0.0, 0.0}},
                                        {{length, length, length}},
                                        {{0, 0, 0}},
                                        extents,
                                        {{false, false, false}}};
  return creator.create_domain();
}

template <>
Domain<2> create_domain<2>(const double length,
                           const std::array<size_t, 2>& extents) {
  const domain::creators::Rectangle creator{
      {{0.0, 0.0}}, {{length, length}}, {{0, 0}}, extents, {{false, false}}};
  return creator.create_domain();
}

template <>
Domain<1> create_domain<1>(const double length,
                           const std::array<size_t, 1>& extents) {
  const domain::creators::Interval creator{{{0.0}}, {{length}}, {{0}},
                                           extents, {{false}},  nullptr};
  return creator.create_domain();
}

template <size_t Dim>
tnsr::I<DataVector, Dim, Frame::ElementLogical> create_target_points(
    size_t n_random_target_points) {
  tnsr::I<DataVector, Dim, Frame::ElementLogical> xi_target{
      n_random_target_points + 2, -1.0};
  for (size_t d = 0; d < Dim; ++d) {
    xi_target.get(d)[1] = 1.0;
  }
  return xi_target;
}

namespace Tags {
struct Scalar : ::db::SimpleTag {
  using type = ::Scalar<DataVector>;
};
}  // namespace Tags

using var_tags = tmpl::list<Tags::Scalar>;

template <size_t Dim>
Variables<var_tags> polynomial(
    const tnsr::I<DataVector, Dim, Frame::Inertial>& x, const size_t degree) {
  tnsr::I<DataVector, Dim, Frame::Inertial> v = x;
  for (size_t d = 0; d < Dim; ++d) {
    for (size_t n = degree; n > 1; --n) {
      v.get(d) *= (x.get(d) + 1.0);
    }
  }
  Variables<var_tags> result{get<0>(x).size()};
  get(get<Tags::Scalar>(result)) = get<0>(v);
  for (size_t d = 1; d < Dim; ++d) {
    get(get<Tags::Scalar>(result)) *= v.get(d);
  }
  return result;
}

template <size_t Dim, size_t MaxDegree>
void test_polynomial_interpolant(const std::array<size_t, Dim>& extents) {
  const size_t n_random_target_points = 10;

  const auto domain = create_domain<Dim>(20.0 / 3.0, extents);
  const auto& block = domain.blocks()[0];
  const ElementMap<Dim, Frame::Inertial> element_map{
      ElementId<Dim>{0}, block.stationary_map().get_clone()};
  Mesh<Dim> mesh(extents, Spectral::Basis::FiniteDifference,
                 Spectral::Quadrature::CellCentered);

  const auto source_xi = logical_coordinates(mesh);
  const auto source_x = element_map(source_xi);
  const auto target_xi = create_target_points<Dim>(n_random_target_points);
  const auto target_x = element_map(target_xi);
  intrp::Irregular irregular_interp{mesh, target_xi};

  for (size_t degree = 0; degree <= MaxDegree; ++degree) {
    const auto source_vars = polynomial<Dim>(source_x, degree);
    const auto target_vars = irregular_interp.interpolate(source_vars);
    const auto expected_vars = polynomial<Dim>(target_x, degree);
    CHECK_VARIABLES_APPROX(target_vars, expected_vars);
  }
}

void test_tov() {
  const std::array<size_t, 3> isotropic_extents{{11, 11, 11}};
  constexpr size_t n_resolutions = 4;
  auto errors =
      make_array<n_resolutions>(std::numeric_limits<double>::signaling_NaN());
  const double central_density = 1.28e-3;
  for (size_t i = 0; i < n_resolutions; ++i) {
    const Domain<3> domain = create_domain<3>(
        6.6666666666666666666 / two_to_the(i), isotropic_extents);
    const Block<3>& cube = domain.blocks()[0];
    Mesh<3> mesh(isotropic_extents, Spectral::Basis::FiniteDifference,
                 Spectral::Quadrature::CellCentered);
    const auto xi = logical_coordinates(mesh);
    const ElementMap<3, Frame::Inertial> element_map{
        ElementId<3>{0}, cube.stationary_map().get_clone()};
    const auto x = element_map(xi);

    RelativisticEuler::Solutions::TovStar<gr::Solutions::TovSolution> tov_star(
        central_density, 100.0, 2.0);

    using rho_tag = hydro::Tags::RestMassDensity<DataVector>;
    auto vars = variables_from_tagged_tuple(
        tov_star.variables(x, 0.0, tmpl::list<rho_tag>{}));

    const tnsr::I<DataVector, 3, Frame::ElementLogical> xi_target{1_st, -1.0};

    intrp::Irregular irregular_interp{mesh, xi_target};

    const auto target_vars = irregular_interp.interpolate(vars);
    gsl::at(errors, i) =
        fabs(central_density - get(get<rho_tag>(target_vars))[0]);
  }

  std::reverse(std::begin(errors), std::end(errors));
  auto ratio_of_errors =
      make_array<n_resolutions>(std::numeric_limits<double>::signaling_NaN());
  std::adjacent_difference(std::begin(errors), std::end(errors),
                           std::begin(ratio_of_errors), std::divides<>{});

  Approx custom_approx = Approx::custom().epsilon(1.e-2).scale(1.);
  for (size_t i = 1; i < n_resolutions; ++i) {
    CHECK(4.0 == custom_approx(gsl::at(ratio_of_errors, i)));
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Interpolation.IrregularInterpolant",
                  "[Unit][NumericalAlgorithms]") {
  test_irregular_interpolant<Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto>();
  test_irregular_interpolant<Spectral::Basis::Legendre,
                             Spectral::Quadrature::Gauss>();
  test_irregular_interpolant_mixed_quadrature();
  test_polynomial_interpolant<1, 1>({{11}});
  test_polynomial_interpolant<2, 1>({{11, 11}});
  test_polynomial_interpolant<2, 1>({{11, 9}});
  test_polynomial_interpolant<3, 1>({{11, 11, 11}});
  test_polynomial_interpolant<3, 1>({{11, 9, 11}});
  test_polynomial_interpolant<3, 1>({{11, 11, 9}});
  test_polynomial_interpolant<3, 1>({{11, 9, 9}});
  test_polynomial_interpolant<3, 1>({{11, 9, 13}});
  test_tov();
}
