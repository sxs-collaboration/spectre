// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <type_traits>
#include <utility>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/Variables/FrameTransform.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "PointwiseFunctions/MathFunctions/TensorProduct.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Identity1D = domain::CoordinateMaps::Identity<1>;
using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <size_t VolumeDim>
auto make_affine_map();

template <>
auto make_affine_map<1>() {
  return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
      Affine{-1.0, 1.0, -0.3, 0.7});
}

template <>
auto make_affine_map<2>() {
  return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
      Affine2D{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55}});
}

template <>
auto make_affine_map<3>() {
  return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
      Affine3D{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55},
               Affine{-1.0, 1.0, 2.3, 2.8}});
}

template <typename DataType, size_t Dim, typename Frame>
struct Flux1 : db::SimpleTag {
  using type = tnsr::I<DataType, Dim, Frame>;
  static auto flux(const MathFunctions::TensorProduct<Dim>& f,
                   const tnsr::I<DataVector, Dim, Frame>& x) {
    auto result = make_with_value<tnsr::I<DataType, Dim, Frame>>(x, 0.);
    const auto f_of_x = f(x);
    for (size_t d = 0; d < Dim; ++d) {
      result.get(d) = (d + 0.5) * get(f_of_x);
    }
    if constexpr (std::is_same_v<DataType, ComplexDataVector>) {
      for (size_t d = 0; d < Dim; ++d) {
        result.get(d) +=
            std::complex<double>(0., static_cast<double>(d) + 1.5) *
            get(f_of_x);
      }
    }
    return result;
  }
  static auto divergence_of_flux(const MathFunctions::TensorProduct<Dim>& f,
                                 const tnsr::I<DataVector, Dim, Frame>& x) {
    auto result = make_with_value<Scalar<DataType>>(x, 0.);
    const auto df = f.first_derivatives(x);
    for (size_t d = 0; d < Dim; ++d) {
      get(result) += (d + 0.5) * df.get(d);
    }
    if constexpr (std::is_same_v<DataType, ComplexDataVector>) {
      for (size_t d = 0; d < Dim; ++d) {
        get(result) +=
            std::complex<double>(0., static_cast<double>(d) + 1.5) * df.get(d);
      }
    }
    return result;
  }
};

template <typename DataType, size_t Dim, typename Frame>
struct Flux2 : db::SimpleTag {
  using type = tnsr::Ij<DataType, Dim, Frame>;
  static auto flux(const MathFunctions::TensorProduct<Dim>& f,
                   const tnsr::I<DataVector, Dim, Frame>& x) {
    auto result = make_with_value<tnsr::Ij<DataType, Dim, Frame>>(x, 0.);
    const auto f_of_x = f(x);
    for (size_t d = 0; d < Dim; ++d) {
      for (size_t j = 0; j < Dim; ++j) {
        result.get(d, j) = (d + 0.5) * (j + 0.25) * get(f_of_x);
      }
    }
    return result;
  }
  static auto divergence_of_flux(const MathFunctions::TensorProduct<Dim>& f,
                                 const tnsr::I<DataVector, Dim, Frame>& x) {
    auto result = make_with_value<tnsr::i<DataType, Dim, Frame>>(x, 0.);
    const auto df = f.first_derivatives(x);
    for (size_t j = 0; j < Dim; ++j) {
      for (size_t d = 0; d < Dim; ++d) {
        result.get(j) += (d + 0.5) * (j + 0.25) * df.get(d);
      }
    }
    return result;
  }
};

template <typename DataType, size_t Dim, typename Frame>
using two_fluxes =
    tmpl::list<Flux1<DataType, Dim, Frame>, Flux2<DataType, Dim, Frame>>;

template <typename DataType, size_t Dim, typename Frame = Frame::Inertial>
void test_divergence_impl(
    const Mesh<Dim>& mesh,
    std::array<std::unique_ptr<MathFunction<1, Frame>>, Dim> functions) {
  const auto coordinate_map = make_affine_map<Dim>();
  const size_t num_grid_points = mesh.number_of_grid_points();
  const auto xi = logical_coordinates(mesh);
  const auto x = coordinate_map(xi);
  const auto inv_jacobian = coordinate_map.inv_jacobian(xi);
  const auto det_jacobian = determinant(coordinate_map.jacobian(xi));
  MathFunctions::TensorProduct<Dim> f(1.0, std::move(functions));
  using flux_tags = two_fluxes<DataType, Dim, Frame>;
  Variables<flux_tags> fluxes(num_grid_points);
  Variables<db::wrap_tags_in<Tags::div, flux_tags>> expected_div_fluxes(
      num_grid_points);
  tmpl::for_each<flux_tags>([&x, &f, &fluxes, &expected_div_fluxes](auto tag) {
    using FluxTag = tmpl::type_from<decltype(tag)>;
    get<FluxTag>(fluxes) = FluxTag::flux(f, x);
    using DivFluxTag = Tags::div<FluxTag>;
    get<DivFluxTag>(expected_div_fluxes) = FluxTag::divergence_of_flux(f, x);
  });
  const auto div_fluxes = divergence(fluxes, mesh, inv_jacobian);
  CHECK(div_fluxes.size() == expected_div_fluxes.size());
  CHECK(Dim * div_fluxes.size() == fluxes.size());
  Approx local_approx = Approx::custom().epsilon(1.e-11).scale(1.);
  CHECK_VARIABLES_CUSTOM_APPROX(div_fluxes, expected_div_fluxes, local_approx);

  // Test divergence of a single tensor
  const auto div_vector =
      divergence(get<Flux1<DataType, Dim, Frame>>(fluxes), mesh, inv_jacobian);
  const auto& expected =
      get<Tags::div<Flux1<DataType, Dim, Frame>>>(div_fluxes);
  CHECK_ITERABLE_CUSTOM_APPROX(expected, div_vector, local_approx);

  // Test logical divergence
  auto logical_fluxes =
      transform::first_index_to_different_frame(fluxes, inv_jacobian);
  Variables<db::wrap_tags_in<Tags::div, flux_tags>> logical_div_fluxes(
      num_grid_points);
  logical_divergence(make_not_null(&logical_div_fluxes), logical_fluxes, mesh);
  CHECK_VARIABLES_CUSTOM_APPROX(logical_div_fluxes, expected_div_fluxes,
                                local_approx);
}

template <typename DataType>
void test_divergence() {
  using TensorTag = Flux1<DataType, 1, Frame::Inertial>;
  TestHelpers::db::test_prefix_tag<Tags::div<TensorTag>>("div(Flux1)");

  const size_t n0 =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre> / 2;
  const size_t n1 =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre> / 2 + 1;
  const size_t n2 =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre> / 2 - 1;
  const Mesh<1> mesh_1d{
      {{n0}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};
  const Mesh<2> mesh_2d{{{n0, n1}},
                        Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> mesh_3d{{{n0, n1, n2}},
                        Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  for (size_t a = 0; a < 5; ++a) {
    std::array<std::unique_ptr<MathFunction<1, Frame::Inertial>>, 1>
        functions_1d{
            {std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(a)}};
    test_divergence_impl<DataType>(mesh_1d, std::move(functions_1d));
    for (size_t b = 0; b < 4; ++b) {
      std::array<std::unique_ptr<MathFunction<1, Frame::Inertial>>, 2>
          functions_2d{
              {std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(a),
               std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(b)}};
      test_divergence_impl<DataType>(mesh_2d, std::move(functions_2d));
      for (size_t c = 0; c < 3; ++c) {
        std::array<std::unique_ptr<MathFunction<1, Frame::Inertial>>, 3>
            functions_3d{
                {std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(a),
                 std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(b),
                 std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(c)}};
        test_divergence_impl<DataType>(mesh_3d, std::move(functions_3d));
      }
    }
  }
}

template <class MapType>
struct MapTag : db::SimpleTag {
  static constexpr size_t dim = MapType::dim;
  using target_frame = typename MapType::target_frame;
  using source_frame = typename MapType::source_frame;

  using type = MapType;
};

template <size_t Dim, typename Frame = Frame::Inertial>
void test_divergence_compute_item_impl(
    const Mesh<Dim>& mesh,
    std::array<std::unique_ptr<MathFunction<1, Frame>>, Dim> functions) {
  const auto coordinate_map = make_affine_map<Dim>();
  using map_tag = MapTag<std::decay_t<decltype(coordinate_map)>>;
  using mesh_tag = domain::Tags::Mesh<Dim>;
  using inv_jac_tag = domain::Tags::InverseJacobianCompute<
      map_tag, domain::Tags::LogicalCoordinates<Dim>>;
  using flux_tags = two_fluxes<DataVector, Dim, Frame>;
  using flux_tag = Tags::Variables<flux_tags>;
  using div_tags = db::wrap_tags_in<Tags::div, flux_tags>;
  TestHelpers::db::test_compute_tag<
      Tags::DivVariablesCompute<flux_tag, mesh_tag, inv_jac_tag>>(
      "Variables(div(Flux1),div(Flux2))");
  TestHelpers::db::test_compute_tag<Tags::DivVectorCompute<
      Flux1<DataVector, Dim, Frame>, mesh_tag, inv_jac_tag>>("div(Flux1)");

  const size_t num_grid_points = mesh.number_of_grid_points();
  const auto xi = logical_coordinates(mesh);
  const auto x = coordinate_map(xi);
  const auto inv_jacobian = coordinate_map.inv_jacobian(xi);
  MathFunctions::TensorProduct<Dim> f(1.0, std::move(functions));
  Variables<flux_tags> fluxes(num_grid_points);
  Variables<div_tags> expected_div_fluxes(num_grid_points);

  tmpl::for_each<flux_tags>([&x, &f, &fluxes, &expected_div_fluxes](auto tag) {
    using FluxTag = tmpl::type_from<decltype(tag)>;
    get<FluxTag>(fluxes) = FluxTag::flux(f, x);
    using DivFluxTag = Tags::div<FluxTag>;
    get<DivFluxTag>(expected_div_fluxes) = FluxTag::divergence_of_flux(f, x);
  });

  auto box =
      db::create<db::AddSimpleTags<mesh_tag, flux_tag, map_tag>,
                 db::AddComputeTags<
                     domain::Tags::LogicalCoordinates<Dim>, inv_jac_tag,
                     Tags::DivVariablesCompute<flux_tag, mesh_tag, inv_jac_tag>,
                     Tags::DivVectorCompute<Flux1<DataVector, Dim, Frame>,
                                            mesh_tag, inv_jac_tag>>>(
          mesh, fluxes, coordinate_map);

  const auto& div_fluxes = db::get<Tags::Variables<div_tags>>(box);

  CHECK(div_fluxes.size() == expected_div_fluxes.size());
  Approx local_approx = Approx::custom().epsilon(1.e-11).scale(1.);
  CHECK_VARIABLES_CUSTOM_APPROX(div_fluxes, expected_div_fluxes, local_approx);

  const auto& div_flux1 =
      db::get<Tags::DivVectorCompute<Flux1<DataVector, Dim, Frame>, mesh_tag,
                                     inv_jac_tag>>(box);
  const auto& expected =
      get<Tags::div<Flux1<DataVector, Dim, Frame>>>(div_fluxes);
  CHECK_ITERABLE_CUSTOM_APPROX(expected, div_flux1, local_approx);
}

void test_divergence_compute() {
  const size_t n0 =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre> / 2;
  const size_t n1 =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre> / 2 + 1;
  const size_t n2 =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre> / 2 - 1;
  const Mesh<1> mesh_1d{
      {{n0}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};
  const Mesh<2> mesh_2d{{{n0, n1}},
                        Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> mesh_3d{{{n0, n1, n2}},
                        Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  for (size_t a = 0; a < 5; ++a) {
    std::array<std::unique_ptr<MathFunction<1, Frame::Inertial>>, 1>
        functions_1d{
            {std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(a)}};
    test_divergence_compute_item_impl(mesh_1d, std::move(functions_1d));
    for (size_t b = 0; b < 4; ++b) {
      std::array<std::unique_ptr<MathFunction<1, Frame::Inertial>>, 2>
          functions_2d{
              {std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(a),
               std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(b)}};
      test_divergence_compute_item_impl(mesh_2d, std::move(functions_2d));
      for (size_t c = 0; c < 3; ++c) {
        std::array<std::unique_ptr<MathFunction<1, Frame::Inertial>>, 3>
            functions_3d{
                {std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(a),
                 std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(b),
                 std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(c)}};
        test_divergence_compute_item_impl(mesh_3d, std::move(functions_3d));
      }
    }
  }
}

template <bool spherical>
DataVector cartoon_func(const tnsr::I<DataVector, 3, Frame::Inertial>& coords) {
  if constexpr (spherical) {
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

template <bool spherical>
DataVector cartoon_dfunc(
    size_t deriv_index, const tnsr::I<DataVector, 3, Frame::Inertial>& coords) {
  if constexpr (spherical) {
    if (deriv_index == 0) {
      return 0.04 * pow(get<0>(coords), 3) + 0.9 * pow(get<0>(coords), 2) -
             0.2 * get<0>(coords) - 2.0;
    } else {
      return 0.0 * get<0>(coords);
    }
  } else {
    if (deriv_index == 0) {
      return 2.0 * get<0>(coords) * get<1>(coords);
    } else if (deriv_index == 1) {
      return 2.0 * get<1>(coords) + square(get<0>(coords));
    } else {
      return 0.0 * get<0>(coords);
    }
  }
}

template <bool spherical>
void test_cartoon(const double x_start) {
  Mesh<3> mesh;
  tnsr::I<DataVector, 3, Frame::Inertial> coords;
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      inv_jacobian;

  const Identity1D identity_cartoon_map;

  if constexpr (spherical) {
    // spherical symmetry
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
    const domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
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
    const domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                Cartoon_map_combination>
        map{{affine_x_map, affine_y_map, identity_cartoon_map}};
    inv_jacobian = map.inv_jacobian(logical_coordinates(mesh));
    coords = map(logical_coordinates(mesh));
  }

  using TempIjk =
      ::Tags::TempTensor<0, tnsr::Ijk<DataVector, 3, Frame::Inertial>>;

  using VarTags = tmpl::list<::Tags::TempI<0, 3, Frame::Inertial>,
                             ::Tags::TempIj<0, 3, Frame::Inertial>, TempIjk>;

  Variables<VarTags> vars{mesh.number_of_grid_points()};

  using div_VarTags =
      tmpl::transform<VarTags, tmpl::bind<::Tags::div, tmpl::_1>>;

  Variables<div_VarTags> expected_div_vars{mesh.number_of_grid_points()};

  // Here we create "prefactors", which serve to fill out our tensors by being
  // mulitplied by our cartoon_func(), which themselves make our tensors have
  // some nontrivial spatial derivative
  // The point of these prefactors is to ensure the tensors respect the
  // symmetry of the spacetime: it is not sufficient that each component
  // follows the symmetry, rather the entire tensor must satisfy
  // \mathcal{L}_\xi (tensor) = 0. Each rank has it's own form of prefactor
  Variables<VarTags> prefactor_vars{mesh.number_of_grid_points()};

  using TempiJkl =
      ::Tags::TempTensor<0, TensorMetafunctions::prepend_spatial_index<
                                tnsr::Ijk<DataVector, 3, Frame::Inertial>, 3,
                                UpLo::Lo, Frame::Inertial>>;

  // prepending a spatial index to VarTags (= PrefactorVarTags if it existed)
  using d_PrefactorVarTags =
      tmpl::list<::Tags::TempiJ<0, 3, Frame::Inertial>,
                 ::Tags::TempiJk<0, 3, Frame::Inertial>, TempiJkl>;
  Variables<d_PrefactorVarTags> d_prefactor_vars{mesh.number_of_grid_points()};

  const size_t dv_size = get<0>(coords).size();

  auto& vector = get<::Tags::TempI<0, 3>>(prefactor_vars);
  auto& d_vector = get<::Tags::TempiJ<0, 3>>(d_prefactor_vars);
  if constexpr (spherical) {
    // spherical case, vector/one form, using x^i
    for (size_t i = 0; i < index_dim<0>(vector); ++i) {
      vector.get(i) = coords.get(i);
    }
    // partial_i of x_j is \delta_ij
    for (size_t i = 0; i < index_dim<0>(d_vector); ++i) {
      for (size_t j = 0; j < index_dim<1>(d_vector); ++j) {
        if (i == j) {
          d_vector.get(i, j) = DataVector(dv_size, 1.0);
        } else {
          d_vector.get(i, j) = DataVector(dv_size, 0.0);
        }
      }
    }
  } else {
    // axial case, vector/one form, using x^i = (-z, 0, x) (pure rotation)
    get<0>(vector) = -1.0 * coords.get(2);
    get<1>(vector) = DataVector(dv_size, 0.0);
    get<2>(vector) = get<0>(coords);

    // \partial_i of x_j is (\delta_i2, 0, \delta_i0)
    for (size_t i = 0; i < index_dim<0>(d_vector); ++i) {
      for (size_t j = 0; j < index_dim<1>(d_vector); ++j) {
        if (i == 2 and j == 0) {
          d_vector.get(i, j) = DataVector(dv_size, -1.0);
        } else if (i == 0 and j == 2) {
          d_vector.get(i, j) = DataVector(dv_size, 1.0);
        } else {
          d_vector.get(i, j) = DataVector(dv_size, 0.0);
        }
      }
    }
  }

  auto& rank2 = get<::Tags::TempIj<0, 3>>(prefactor_vars);
  auto& d_rank2 = get<::Tags::TempiJk<0, 3>>(d_prefactor_vars);
  // filling with 0's in time components
  // filling space with (essenially) projector to tangent space of sphere
  // P_ij = \delta_ij + x_i x_j
  // can have arbitrary function in from of each term, not doing here
  // (the real projector is \delta_ij - x_i x_j / r^2)
  for (size_t i = 0; i < index_dim<0>(rank2); ++i) {
    for (size_t j = 0; j < index_dim<1>(rank2); ++j) {
      if (i == j) {
        rank2.get(i, j) =
            DataVector(dv_size, 1.0) + coords.get(i) * coords.get(j);
      } else {
        rank2.get(i, j) = coords.get(i) * coords.get(j);
      }
    }
  }
  // \partial_i of (\delta_jk + x_j x_k) is (x_j \delta_ik + x_k \delta_ij)
  for (size_t i = 0; i < index_dim<0>(d_rank2); ++i) {
    for (size_t j = 0; j < index_dim<1>(d_rank2); ++j) {
      for (size_t k = 0; k < index_dim<2>(d_rank2); ++k) {
        d_rank2.get(i, j, k) = DataVector(dv_size, 0.0);
        if (i == k) {
          d_rank2.get(i, j, k) += coords.get(j);
        }
        if (i == j) {
          d_rank2.get(i, j, k) += coords.get(k);
        }
      }
    }
  }

  auto& rank3 = get<TempIjk>(prefactor_vars);
  auto& d_rank3 = get<TempiJkl>(d_prefactor_vars);
  // x_i x_j x_k + \delta_ij x_k \delta_ik x_j \delta_jk x_i
  for (size_t i = 0; i < index_dim<0>(rank3); ++i) {
    for (size_t j = 0; j < index_dim<1>(rank3); ++j) {
      for (size_t k = 0; k < index_dim<2>(rank3); ++k) {
        rank3.get(i, j, k) = coords.get(i) * coords.get(j) * coords.get(k);
        if (i == j) {
          rank3.get(i, j, k) += coords.get(k);
        }
        if (i == k) {
          rank3.get(i, j, k) += coords.get(j);
        }
        if (j == k) {
          rank3.get(i, j, k) += coords.get(i);
        }
      }
    }
  }
  // \partial_i of (x_j x_k x_l + \delta_jk x_l \delta_jl x_k \delta_kl x_j) is
  // \delta_ij x_k x_l + \delta_ik x_j x_l + \delta_il x_j x_k +
  //   \delta_jk \delta_il + \delta_jl \delta_ik + \delta_kl \delta_ij
  for (size_t i = 0; i < index_dim<0>(d_rank3); ++i) {
    for (size_t j = 0; j < index_dim<1>(d_rank3); ++j) {
      for (size_t k = 0; k < index_dim<2>(d_rank3); ++k) {
        for (size_t l = 0; l < index_dim<3>(d_rank3); ++l) {
          d_rank3.get(i, j, k, l) = DataVector(dv_size, 0.0);
          if (i == j) {
            d_rank3.get(i, j, k, l) += coords.get(k) * coords.get(l);
          }
          if (i == k) {
            d_rank3.get(i, j, k, l) += coords.get(j) * coords.get(l);
          }
          if (i == l) {
            d_rank3.get(i, j, k, l) += coords.get(j) * coords.get(k);
          }
          if (j == k and i == l) {
            d_rank3.get(i, j, k, l) += 1.0;
          }
          if (j == l and i == k) {
            d_rank3.get(i, j, k, l) += 1.0;
          }
          if (k == l and i == j) {
            d_rank3.get(i, j, k, l) += 1.0;
          }
        }
      }
    }
  }

  tmpl::for_each<VarTags>([&vars, &prefactor_vars, &expected_div_vars,
                           &d_prefactor_vars, &coords]<typename tensor_tag>(
                              tmpl::type_<tensor_tag> /*meta*/) {
    auto& tensor = get<tensor_tag>(vars);
    auto& prefactor_tensor = get<tensor_tag>(prefactor_vars);
    using div_tensor_tag = ::Tags::div<tensor_tag>;
    auto& div_tensor = get<div_tensor_tag>(expected_div_vars);
    auto& d_prefactor_tensor =
        get<tmpl::at<d_PrefactorVarTags, tmpl::index_of<VarTags, tensor_tag>>>(
            d_prefactor_vars);

    for (size_t storage_index = 0; storage_index < tensor.size();
         ++storage_index) {
      tensor[storage_index] =
          prefactor_tensor[storage_index] * cartoon_func<spherical>(coords);

      const auto input_index = tensor.get_tensor_index(storage_index);
      std::array<size_t, tensor_tag::type::rank() - 1> output_index;
      std::copy(input_index.begin() + 1, input_index.end(),
                output_index.begin());

      const size_t deriv_index = input_index[0];
      if (deriv_index == 0) {
        div_tensor.get(output_index) = 0.0 * tensor.get(input_index);
      }
      const auto d_input_index = prepend(input_index, deriv_index);

      div_tensor.get(output_index) +=
          prefactor_tensor.get(input_index) *
              cartoon_dfunc<spherical>(deriv_index, coords) +
          cartoon_func<spherical>(coords) *
              d_prefactor_tensor.get(d_input_index);
    }
  });

  Variables<div_VarTags> div_vars{mesh.number_of_grid_points()};
  cartoon_divergence(make_not_null(&div_vars), vars, mesh, inv_jacobian,
                     coords);

  const Approx local_approx = Approx::custom().epsilon(1.0e-10).scale(1.0);
  CHECK_VARIABLES_CUSTOM_APPROX(div_vars, expected_div_vars, local_approx);
}
}  // namespace

// [[Timeout, 20]]
SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.Divergence",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  test_divergence<DataVector>();
  test_divergence<ComplexDataVector>();
  test_divergence_compute();

  test_cartoon<true>(0.0);
  test_cartoon<true>(1.0);
  test_cartoon<false>(0.0);
  test_cartoon<false>(1.0);

  BENCHMARK_ADVANCED("Divergence of vector")
  (Catch::Benchmark::Chronometer meter) {
    const Mesh<3> mesh{4, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const Affine map1d(-1.0, 1.0, -1.0, 1.0);
    const domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                                Affine3D>
        map(Affine3D{map1d, map1d, map1d});
    const auto inv_jacobian = map.inv_jacobian(logical_coordinates(mesh));
    tnsr::I<DataVector, 3> input{mesh.number_of_grid_points(), 0.};
    Scalar<DataVector> div{mesh.number_of_grid_points()};
    meter.measure([&div, &input, &mesh, &inv_jacobian]() {
      divergence(make_not_null(&div), input, mesh, inv_jacobian);
    });
  };
}
