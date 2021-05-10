// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/Tuple.hpp"

namespace FirstOrderEllipticSolutionsTestHelpers {
namespace detail {
namespace Tags {

template <typename Tag>
struct OperatorAppliedTo : db::SimpleTag, db::PrefixTag {
  using type = typename Tag::type;
  using tag = Tag;
};

}  // namespace Tags

template <typename System, typename SolutionType, typename... Maps,
          typename... FluxesArgs, typename... SourcesArgs,
          typename... PrimalFields, typename... AuxiliaryFields,
          typename... PrimalFluxes, typename... AuxiliaryFluxes>
void verify_solution_impl(
    const SolutionType& solution, const Mesh<System::volume_dim>& mesh,
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, Maps...>
        coord_map,
    const double tolerance, const std::tuple<FluxesArgs...>& fluxes_args,
    const std::tuple<SourcesArgs...>& sources_args,
    tmpl::list<PrimalFields...> /*meta*/,
    tmpl::list<AuxiliaryFields...> /*meta*/,
    tmpl::list<PrimalFluxes...> /*meta*/,
    tmpl::list<AuxiliaryFluxes...> /*meta*/) {
  using all_fields = tmpl::list<PrimalFields..., AuxiliaryFields...>;
  using all_fluxes = tmpl::list<PrimalFluxes..., AuxiliaryFluxes...>;
  using FluxesComputer = typename System::fluxes_computer;
  using SourcesComputer = typename System::sources_computer;
  CAPTURE(mesh);

  const size_t num_points = mesh.number_of_grid_points();
  const auto logical_coords = logical_coordinates(mesh);
  const auto inertial_coords = coord_map(logical_coords);
  const auto solution_fields = variables_from_tagged_tuple(
      solution.variables(inertial_coords, all_fields{}));

  // Apply operator to solution fields: -div(F) + S = f
  Variables<all_fluxes> fluxes{num_points};
  std::apply(
      [&fluxes, &solution_fields](const auto&... expanded_fluxes_args) {
        FluxesComputer::apply(make_not_null(&get<PrimalFluxes>(fluxes))...,
                              expanded_fluxes_args...,
                              get<AuxiliaryFields>(solution_fields)...);
        FluxesComputer::apply(make_not_null(&get<AuxiliaryFluxes>(fluxes))...,
                              expanded_fluxes_args...,
                              get<PrimalFields>(solution_fields)...);
      },
      fluxes_args);
  Variables<db::wrap_tags_in<Tags::OperatorAppliedTo, all_fields>>
      operator_applied_to_fields{num_points};
  divergence(make_not_null(&operator_applied_to_fields), fluxes, mesh,
             coord_map.inv_jacobian(logical_coords));
  operator_applied_to_fields *= -1.;
  std::apply(
      [&operator_applied_to_fields, &solution_fields,
       &fluxes](const auto&... expanded_sources_args) {
        SourcesComputer::apply(
            make_not_null(&get<Tags::OperatorAppliedTo<AuxiliaryFields>>(
                operator_applied_to_fields))...,
            expanded_sources_args..., get<PrimalFields>(solution_fields)...);
        SourcesComputer::apply(
            make_not_null(&get<Tags::OperatorAppliedTo<PrimalFields>>(
                operator_applied_to_fields))...,
            expanded_sources_args..., get<PrimalFields>(solution_fields)...,
            get<PrimalFluxes>(fluxes)...);
      },
      sources_args);

  // Set RHS for auxiliary equations to minus auxiliary fields, and for primal
  // equations to the solution's fixed sources f(x)
  Variables<db::wrap_tags_in<::Tags::FixedSource, all_fields>> fixed_sources{
      num_points, 0.};
  expand_pack((get<::Tags::FixedSource<AuxiliaryFields>>(fixed_sources) =
                   get<AuxiliaryFields>(solution_fields))...);
  fixed_sources *= -1.;
  fixed_sources.assign_subset(solution.variables(
      inertial_coords, tmpl::list<::Tags::FixedSource<PrimalFields>...>{}));

  // Check error norms against the given tolerance
  tmpl::for_each<all_fields>([&operator_applied_to_fields, &fixed_sources,
                              &tolerance](auto field_tag_v) {
    using field_tag = tmpl::type_from<decltype(field_tag_v)>;
    const auto& operator_applied_to_field =
        get<Tags::OperatorAppliedTo<field_tag>>(operator_applied_to_fields);
    const auto& fixed_source =
        get<::Tags::FixedSource<field_tag>>(fixed_sources);
    double l2_error_square = 0.;
    double linf_error = 0.;
    for (size_t i = 0; i < operator_applied_to_field.size(); ++i) {
      const auto error = abs(operator_applied_to_field[i] - fixed_source[i]);
      l2_error_square += alg::accumulate(square(error), 0.) / error.size();
      const double component_linf_error = *alg::max_element(error);
      if (component_linf_error > linf_error) {
        linf_error = component_linf_error;
      }
    }
    const double l2_error =
        sqrt(l2_error_square / operator_applied_to_field.size());
    CAPTURE(db::tag_name<field_tag>());
    CAPTURE(l2_error);
    CAPTURE(linf_error);
    CHECK(l2_error < tolerance);
    CHECK(linf_error < tolerance);
  });
}

}  // namespace detail

/// \ingroup TestingFrameworkGroup
/// Test that the `solution` numerically solves the `System` on the given grid
/// for the given tolerance
// @{
template <typename System, typename SolutionType, typename... Maps,
          typename... FluxesArgs, typename... SourcesArgs>
void verify_solution(
    const SolutionType& solution, const Mesh<System::volume_dim>& mesh,
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, Maps...>
        coord_map,
    const double tolerance, const std::tuple<FluxesArgs...>& fluxes_args,
    const std::tuple<SourcesArgs...>& sources_args = std::tuple<>{}) {
  detail::verify_solution_impl<System>(
      solution, mesh, coord_map, tolerance, fluxes_args, sources_args,
      typename System::primal_fields{}, typename System::auxiliary_fields{},
      typename System::primal_fluxes{}, typename System::auxiliary_fluxes{});
}

template <typename System, typename SolutionType, typename... Maps>
void verify_solution(
    const SolutionType& solution, const Mesh<System::volume_dim>& mesh,
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, Maps...>
        coord_map,
    const double tolerance) {
  using argument_tags = tmpl::remove_duplicates<
      tmpl::append<typename System::fluxes_computer::argument_tags,
                   typename System::sources_computer::argument_tags>>;
  const auto background_fields = [&solution, &mesh, &coord_map]() {
    if constexpr (tmpl::size<argument_tags>::value > 0) {
      const auto logical_coords = logical_coordinates(mesh);
      const auto inertial_coords = coord_map(logical_coords);
      const auto inv_jacobian = coord_map.inv_jacobian(logical_coords);
      return solution.variables(inertial_coords, mesh, inv_jacobian,
                                argument_tags{});
    } else {
      (void)solution;
      (void)mesh;
      (void)coord_map;
      return tuples::TaggedTuple<>{};
    }
  }();
  const auto get_items = [](const auto&... args) {
    return std::forward_as_tuple(args...);
  };
  verify_solution<System>(
      solution, mesh, coord_map, tolerance,
      tuples::apply<typename System::fluxes_computer::argument_tags>(
          get_items, background_fields),
      tuples::apply<typename System::sources_computer::argument_tags>(
          get_items, background_fields));
}
// @}

/*!
 * \ingroup TestingFrameworkGroup
 * Test that the `solution` numerically solves the `System` on the given grid
 * and that the discretization error decreases as expected for a smooth
 * function.
 *
 * \details We expect exponential convergence for a smooth solution, so the
 * tolerance is computed as
 *
 * \f{equation}
 * C_1 \exp{\left(-C_2 * N_\mathrm{points}\right)}
 * \f}
 *
 * where \f$C_1\f$ is the `tolerance_offset`, \f$C_2\f$ is the
 * `tolerance_scaling` and \f$N_\mathrm{points}\f$ is the number of grid points
 * per dimension.
 */
template <typename System, typename SolutionType,
          size_t Dim = System::volume_dim, typename... Maps,
          typename PackageFluxesArgs>
void verify_smooth_solution(
    const SolutionType& solution,
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, Maps...>&
        coord_map,
    const double tolerance_offset, const double tolerance_scaling,
    PackageFluxesArgs&& package_fluxes_args) {
  INFO("Verify smooth solution");
  for (size_t num_points = Spectral::minimum_number_of_points<
           Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>;
       num_points <=
       Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
       num_points++) {
    CAPTURE(num_points);
    const double tolerance =
        tolerance_offset * exp(-tolerance_scaling * num_points);
    CAPTURE(tolerance);
    const Mesh<Dim> mesh{num_points, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
    FirstOrderEllipticSolutionsTestHelpers::verify_solution<System>(
        solution, mesh, coord_map, tolerance, package_fluxes_args(mesh));
  }
}

/*!
 * \ingroup TestingFrameworkGroup
 * Test that the `solution` numerically solves the `System` on the given grid
 * and that the discretization error decreases as a power law.
 *
 * \details The tolerance is computed as
 *
 * \f{equation}
 * C \left(N_\mathrm{points}\right)^{-p}
 * \f}
 *
 * where \f$C\f$ is the `tolerance_offset`, \f$p\f$ is the `tolerance_pow` and
 * \f$N_\mathrm{points}\f$ is the number of grid points per dimension.
 */
template <typename System, typename SolutionType,
          size_t Dim = System::volume_dim, typename... Maps>
void verify_solution_with_power_law_convergence(
    const SolutionType& solution,
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, Maps...>&
        coord_map,
    const double tolerance_offset, const double tolerance_pow) {
  INFO("Verify solution with power-law convergence");
  for (size_t num_points = Spectral::minimum_number_of_points<
           Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>;
       num_points <=
       Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
       num_points++) {
    CAPTURE(num_points);
    const double tolerance = tolerance_offset * pow(num_points, -tolerance_pow);
    CAPTURE(tolerance);
    FirstOrderEllipticSolutionsTestHelpers::verify_solution<System>(
        solution,
        Mesh<Dim>{num_points, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto},
        coord_map, tolerance);
  }
}

}  // namespace FirstOrderEllipticSolutionsTestHelpers
