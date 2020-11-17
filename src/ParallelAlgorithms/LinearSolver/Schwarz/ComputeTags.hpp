// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Weighting.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace LinearSolver::Schwarz::Tags {

/// The number of points a neighbor's subdomain extends into the element
template <size_t Dim, typename OptionsGroup>
struct IntrudingExtentsCompute : db::ComputeTag,
                                 IntrudingExtents<Dim, OptionsGroup> {
  using base = IntrudingExtents<Dim, OptionsGroup>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<Dim>, MaxOverlap<OptionsGroup>>;
  static constexpr void function(
      const gsl::not_null<return_type*> intruding_extents,
      const Mesh<Dim>& mesh, const size_t max_overlap) noexcept {
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(*intruding_extents, d) =
          LinearSolver::Schwarz::overlap_extent(mesh.extents(d), max_overlap);
    }
  }
};

/// The width in element-logical coordinates that a neighbor's subdomain extends
/// into the element
template <size_t Dim, typename OptionsGroup>
struct IntrudingOverlapWidthsCompute
    : db::ComputeTag,
      IntrudingOverlapWidths<Dim, OptionsGroup> {
  using base = IntrudingOverlapWidths<Dim, OptionsGroup>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<Dim>, IntrudingExtents<Dim, OptionsGroup>>;
  static constexpr void function(
      const gsl::not_null<return_type*> intruding_overlap_widths,
      const Mesh<Dim>& mesh,
      const std::array<size_t, Dim>& intruding_extents) noexcept {
    for (size_t d = 0; d < Dim; ++d) {
      const auto& collocation_points =
          Spectral::collocation_points(mesh.slice_through(d));
      gsl::at(*intruding_overlap_widths, d) =
          LinearSolver::Schwarz::overlap_width(gsl::at(intruding_extents, d),
                                               collocation_points);
    }
  }
};

/// Weighting field for data on the element
template <size_t Dim, typename OptionsGroup>
struct ElementWeightCompute : db::ComputeTag, Weight<OptionsGroup> {
  using base = Weight<OptionsGroup>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<domain::Tags::Element<Dim>,
                 domain::Tags::Coordinates<Dim, Frame::Logical>,
                 IntrudingOverlapWidths<Dim, OptionsGroup>,
                 MaxOverlap<OptionsGroup>>;
  static constexpr void function(
      const gsl::not_null<return_type*> element_weight,
      const Element<Dim>& element,
      const tnsr::I<DataVector, Dim, Frame::Logical>& logical_coords,
      const std::array<double, Dim>& intruding_overlap_widths,
      const size_t max_overlap) noexcept {
    // For max_overlap > 0 all overlaps will have non-zero extents on an LGL
    // mesh (because it has at least 2 points per dimension), so we don't need
    // to check their extents are non-zero individually
    if (LIKELY(max_overlap > 0)) {
      LinearSolver::Schwarz::element_weight(element_weight, logical_coords,
                                            intruding_overlap_widths,
                                            element.external_boundaries());
    } else {
      *element_weight = make_with_value<return_type>(logical_coords, 1.);
    }
  }
};

/// Weighting field for data on neighboring subdomains that overlap with the
/// element. Intended to be used with `domain::Tags::InterfaceCompute`.
template <size_t Dim, typename OptionsGroup>
struct IntrudingOverlapWeightCompute : db::ComputeTag, Weight<OptionsGroup> {
  using base = Weight<OptionsGroup>;
  using return_type = typename base::type;
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
  static constexpr void function(
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
      LinearSolver::Schwarz::intruding_weight(
          intruding_overlap_weight, intruding_logical_coords, direction,
          intruding_overlap_widths, element.neighbors().at(direction).size(),
          element.external_boundaries());
    }
  }
};

/*!
 * \brief A diagnostic quantity to check that weights are conserved
 *
 * \see `LinearSolver::Schwarz::Tags::SummedIntrudingOverlapWeights`
 */
template <size_t Dim, typename OptionsGroup>
struct SummedIntrudingOverlapWeightsCompute
    : db::ComputeTag,
      SummedIntrudingOverlapWeights<OptionsGroup> {
  using base = SummedIntrudingOverlapWeights<OptionsGroup>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<domain::Tags::Interface<domain::Tags::InternalDirections<Dim>,
                                         Weight<OptionsGroup>>,
                 domain::Tags::Mesh<Dim>, IntrudingExtents<Dim, OptionsGroup>>;
  static void function(
      const gsl::not_null<return_type*> summed_intruding_overlap_weights,
      const std::unordered_map<Direction<Dim>, Scalar<DataVector>>&
          all_intruding_weights,
      const Mesh<Dim>& mesh,
      const std::array<size_t, Dim>& all_intruding_extents) noexcept {
    destructive_resize_components(summed_intruding_overlap_weights,
                                  mesh.number_of_grid_points());
    get(*summed_intruding_overlap_weights) = 0.;
    for (const auto& [direction, intruding_weight] : all_intruding_weights) {
      // Extend intruding weight to full extents
      // There's not function to extend a single tensor, so we create a
      // temporary Variables. This is only a diagnostic quantity that won't be
      // used in production code, so it doesn't need to be particularly
      // efficient and we don't need to add an overload of
      // `LinearSolver::Schwarz::extended_overlap_data` just for this purpose.
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

}  // namespace LinearSolver::Schwarz::Tags
