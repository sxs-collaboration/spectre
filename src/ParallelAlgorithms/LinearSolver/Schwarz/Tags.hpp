// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "Options/Options.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Weighting.hpp"
#include "Utilities/TMPL.hpp"

namespace LinearSolver::Schwarz {

/// Option tags related to the Schwarz solver
namespace OptionTags {

template <typename OptionsGroup>
struct MaxOverlap {
  using type = size_t;
  using group = OptionsGroup;
  static constexpr Options::String help =
      "Number of points a subdomain can overlap with its neighbor";
};

template <typename SolverType, typename OptionsGroup>
struct SubdomainSolver {
  using type = SolverType;
  using group = OptionsGroup;
  static constexpr Options::String help =
      "Options for the linear solver on subdomains";
};

}  // namespace OptionTags

/// Tags related to the Schwarz solver
namespace Tags {

/// Number of points a subdomain can overlap with its neighbor
template <typename OptionsGroup>
struct MaxOverlap : db::SimpleTag {
  static std::string name() noexcept {
    return "MaxOverlap(" + Options::name<OptionsGroup>() + ")";
  }
  using type = size_t;

  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::MaxOverlap<OptionsGroup>>;
  static type create_from_options(const type& value) noexcept { return value; }
};

// @{
/// The extents of a neighbor's subdomain into the element
template <size_t Dim, typename OptionsGroup>
struct IntrudingExtents : db::SimpleTag {
  using type = std::array<size_t, Dim>;
};

template <size_t Dim, typename OptionsGroup>
struct IntrudingExtentsCompute : db::ComputeTag,
                                 IntrudingExtents<Dim, OptionsGroup> {
  using base = IntrudingExtents<Dim, OptionsGroup>;
  using return_type = tmpl::type_from<base>;
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<Dim>, MaxOverlap<OptionsGroup>>;
  static constexpr auto function(
      const gsl::not_null<return_type*> intruding_extents,
      const Mesh<Dim>& mesh, const size_t max_overlap) noexcept {
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(*intruding_extents, d) =
          LinearSolver::Schwarz::overlap_extent(mesh.extents(d), max_overlap);
    }
  }
};
// @}

/// The `Tag` on the overlap region with each neighbor, i.e. on a region
/// extruding from the central element.
///
/// Note that data on an overlap with a neighbor is typically oriented according
/// to the neighbor's orientation, so re-orientation needs to happen whenever
/// the data cross element boundaries.
template <typename Tag, size_t Dim, typename OptionsGroup>
struct Overlaps : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    return "Overlaps(" + db::tag_name<Tag>() + ", " +
           Options::name<OptionsGroup>() + ")";
  }
  using tag = Tag;
  using type = OverlapMap<Dim, db::item_type<Tag>>;
};

// @{
/// The width of an intruding overlap in element-logical coordinates
template <size_t Dim, typename OptionsGroup>
struct IntrudingOverlapWidths : db::SimpleTag {
  using type = std::array<double, Dim>;
};

template <size_t Dim, typename OptionsGroup>
struct IntrudingOverlapWidthsCompute
    : db::ComputeTag,
      IntrudingOverlapWidths<Dim, OptionsGroup> {
  using base = IntrudingOverlapWidths<Dim, OptionsGroup>;
  using return_type = tmpl::type_from<base>;
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<Dim>, IntrudingExtents<Dim, OptionsGroup>>;
  static constexpr auto function(
      const gsl::not_null<return_type*> intruding_overlap_widths,
      const Mesh<Dim>& mesh,
      const std::array<size_t, Dim>& intruding_overlap_extents) noexcept {
    for (size_t d = 0; d < Dim; ++d) {
      const auto& collocation_points =
          Spectral::collocation_points(mesh.slice_through(d));
      gsl::at(*intruding_overlap_widths, d) =
          LinearSolver::Schwarz::overlap_width(
              gsl::at(intruding_overlap_extents, d), collocation_points);
    }
  }
};
// @}

/// Weighting field for combining data from multiple overlapping subdomains
template <typename OptionsGroup>
struct Weight : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim, typename OptionsGroup>
struct IntrudingOverlapWeightCompute : db::ComputeTag, Weight<OptionsGroup> {
  using base = Weight<OptionsGroup>;
  using return_type = tmpl::type_from<base>;
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
                 IntrudingExtents<Dim, OptionsGroup>,
                 domain::Tags::Direction<Dim>,
                 domain::Tags::Coordinates<Dim, Frame::Logical>,
                 IntrudingOverlapWidths<Dim, OptionsGroup>>;
  using volume_tags =
      tmpl::list<domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
                 IntrudingExtents<Dim, OptionsGroup>,
                 domain::Tags::Coordinates<Dim, Frame::Logical>,
                 IntrudingOverlapWidths<Dim, OptionsGroup>>;
  static constexpr auto function(
      const gsl::not_null<return_type*> intruding_overlap_weight,
      const Mesh<Dim>& mesh, const Element<Dim>& element,
      const std::array<size_t, Dim>& intruding_extents,
      const Direction<Dim>& direction,
      const tnsr::I<DataVector, Dim, Frame::Logical>& logical_coords,
      const std::array<double, Dim> intruding_overlap_widths) noexcept {
    const size_t dim = direction.dimension();
    if (gsl::at(intruding_extents, dim) > 0) {
      const auto intruding_logical_coords =
          LinearSolver::Schwarz::data_on_overlap(
              logical_coords, mesh.extents(), gsl::at(intruding_extents, dim),
              direction);
      *intruding_overlap_weight = LinearSolver::Schwarz::intruding_weight(
          intruding_logical_coords, direction, intruding_overlap_widths,
          element.neighbors().at(direction).size(),
          element.external_boundaries());
    }
  }
};

template <size_t Dim, typename OptionsGroup>
struct ElementWeightCompute : db::ComputeTag, Weight<OptionsGroup> {
  using base = Weight<OptionsGroup>;
  using return_type = tmpl::type_from<base>;
  using argument_tags =
      tmpl::list<domain::Tags::Element<Dim>,
                 domain::Tags::Coordinates<Dim, Frame::Logical>,
                 IntrudingOverlapWidths<Dim, OptionsGroup>,
                 MaxOverlap<OptionsGroup>>;
  static constexpr auto function(
      const gsl::not_null<return_type*> element_weight,
      const Element<Dim>& element,
      const tnsr::I<DataVector, Dim, Frame::Logical>& logical_coords,
      const std::array<double, Dim>& intruding_overlap_widths,
      const size_t max_overlap) noexcept {
    // For max_overlap > 0 all overlaps will have non-zero extents, so we don't
    // need to check their extents individually
    if (LIKELY(max_overlap > 0)) {
      *element_weight = LinearSolver::Schwarz::element_weight(
          logical_coords, intruding_overlap_widths,
          element.external_boundaries());
    } else {
      *element_weight = make_with_value<return_type>(logical_coords, 1.);
    }
  }
};

/// This quantity and the `Weight` on the element should sum to one on all grid
/// points. Residual values indicate that overlap data from neighboring
/// subdomains and data on the element are combined in a non-conservative way.
template <size_t Dim, typename OptionsGroup>
struct SummedIntrudingOverlapWeights : db::ComputeTag {
  static std::string name() noexcept { return "SummedIntrudingOverlapWeights"; }
  using return_type = Scalar<DataVector>;
  using type = return_type;
  using argument_tags =
      tmpl::list<domain::Tags::Interface<domain::Tags::InternalDirections<Dim>,
                                         Weight<OptionsGroup>>,
                 domain::Tags::Mesh<Dim>, IntrudingExtents<Dim, OptionsGroup>>;
  static auto function(
      const gsl::not_null<return_type*> summed_intruding_overlap_weights,
      const std::unordered_map<Direction<Dim>, Scalar<DataVector>>&
          all_intruding_weights,
      const Mesh<Dim>& mesh,
      const std::array<size_t, Dim>& all_intruding_extents) noexcept {
    *summed_intruding_overlap_weights =
        Scalar<DataVector>{mesh.number_of_grid_points(), 0.};
    for (const auto& direction_and_intruding_weight : all_intruding_weights) {
      const auto& direction = direction_and_intruding_weight.first;
      const auto& intruding_weight = direction_and_intruding_weight.second;
      // Extend to full extents
      using temp_tag = ::Tags::TempScalar<0>;
      Variables<tmpl::list<temp_tag>> temp_vars{get(intruding_weight).size()};
      get<temp_tag>(temp_vars) = intruding_weight;
      temp_vars = LinearSolver::Schwarz::extended_overlap_data(
          temp_vars, mesh.extents(),
          gsl::at(all_intruding_extents, direction.dimension()), direction);
      // Contribute to conserved weight
      get(*summed_intruding_overlap_weights) += get(get<temp_tag>(temp_vars));
    }
  }
};

/// The serial linear solver used to solve subdomain operators
template <typename OptionsGroup>
struct SubdomainSolverBase : db::BaseTag {};

/// The serial linear solver of type `SolverType` used to solve subdomain
/// operators
template <typename SolverType, typename OptionsGroup>
struct SubdomainSolver : SubdomainSolverBase<OptionsGroup>, db::SimpleTag {
  static std::string name() noexcept {
    return "SubdomainSolver(" + Options::name<OptionsGroup>() + ")";
  }
  using type = SolverType;

  static constexpr bool pass_metavariables = false;
  using option_tags =
      tmpl::list<OptionTags::SubdomainSolver<SolverType, OptionsGroup>>;
  static type create_from_options(const type& value) noexcept { return value; }
};

}  // namespace Tags

}  // namespace LinearSolver::Schwarz
