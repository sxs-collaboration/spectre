// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "DataStructures/Variables.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/Minmod.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
//#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Element.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "Options/Options.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
template <size_t VolumeDim>
class Direction;
template <size_t VolumeDim>
class ElementId;
template <size_t>
class Mesh;

namespace PUP {
class er;
}  // namespace PUP

namespace SlopeLimiters {
template <size_t VolumeDim, typename TagsToLimit>
class SimpleWeno;
}  // namespace SlopeLimiters

namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
template <size_t VolumeDim>
struct Element;
template <size_t VolumeDim>
struct Mesh;
}  // namespace Tags
/// \endcond

namespace SimpleWeno_detail {
// Compute a simple smoothness indicator
template <size_t VolumeDim, typename... Tags>
tuples::TaggedTuple<Minmod_detail::to_tensor_double<Tags>...>
smoothness_indicator(const Variables<tmpl::list<Tags...>>& vars,
                     const Mesh<VolumeDim>& mesh) noexcept {
  const auto logical_derivs_of_vars =
      logical_partial_derivatives<tmpl::list<Tags...>>(vars, mesh);
  tuples::TaggedTuple<Minmod_detail::to_tensor_double<Tags>...> result{};
  DataVector buffer(vars.number_of_grid_points());
  tmpl::for_each<tmpl::list<Tags...>>(
      [&result, &buffer, &logical_derivs_of_vars, &mesh ](auto tag) noexcept {
        using Tag = tmpl::type_from<decltype(tag)>;
        // loop over tensor components
        for (size_t i = 0; i < Tag::type::size(); ++i) {
          buffer = 0.0;
          for (size_t d = 0; d < VolumeDim; ++d) {
            buffer += square(get<Tag>(gsl::at(logical_derivs_of_vars, d))[i]);
          }
          get<Minmod_detail::to_tensor_double<Tag>>(result)[i] =
              definite_integral(buffer, mesh);
        }
      });
  return result;
}

// Implements the simple WENO limiter for one Tensor<DataVector>.
template <size_t VolumeDim, typename Tag, typename TagList>
void limit_one_tensor(
    const gsl::not_null<db::item_type<Tag>*> tensor_to_limit,
    const Mesh<VolumeDim>& mesh,
    const db::item_type<Minmod_detail::to_tensor_double<Tag>>&
        local_smoothness_indicator,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        Variables<TagList>,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_vars,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        db::item_type<Minmod_detail::to_tensor_double<Tag>>,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_smoothness_indicators,
    const double neighbor_linear_weight) noexcept {
  const double central_linear_weight =
      1.0 - pow<VolumeDim>(2) * neighbor_linear_weight;
  // Limit one tensor component at a time
  for (size_t i = 0; i < tensor_to_limit->size(); ++i) {
    // Un-normalized weights
    const double unnormalized_weight =
        central_linear_weight / square(1.e-6 + local_smoothness_indicator[i]);
    std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, double,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
        neighbor_unnormalized_weights;
    for (const auto& neighbor_and_smoothness_indicator :
         neighbor_smoothness_indicators) {
      const auto& neighbor = neighbor_and_smoothness_indicator.first;
      const auto& neighbor_smoothness_indicator =
          neighbor_and_smoothness_indicator.second;
      neighbor_unnormalized_weights.insert(std::make_pair(
          neighbor, neighbor_linear_weight /
                        square(1.e-6 + neighbor_smoothness_indicator[i])));
    }

    // Normalized weights
    double normalization = unnormalized_weight;
    for (const auto& neighbor_and_weight : neighbor_unnormalized_weights) {
      normalization += neighbor_and_weight.second;
    }
    const double weight = unnormalized_weight / normalization;
    auto neighbor_weights = neighbor_unnormalized_weights;
    for (auto& neighbor_and_weight : neighbor_weights) {
      neighbor_and_weight.second /= normalization;
    }

    // Linear combination
    const double mean = mean_value((*tensor_to_limit)[i], mesh);
    (*tensor_to_limit)[i] = mean + weight * ((*tensor_to_limit)[i] - mean);
    for (const auto& neighbor_and_weight : neighbor_weights) {
      const auto& neighbor = neighbor_and_weight.first;
      const auto& neighbor_tensor_component =
          get<Tag>(neighbor_vars.at(neighbor))[i];
      const double neighbor_mean = mean_value(neighbor_tensor_component, mesh);
      (*tensor_to_limit)[i] += neighbor_and_weight.second *
                               (neighbor_tensor_component - neighbor_mean);
    }
  }
}
}  // namespace SimpleWeno_detail

namespace SlopeLimiters {
/// \ingroup SlopeLimitersGroup
/// \brief A generic simple WENO slope limiter
///
/// Implements the simple WENO slope limiter from...
/// TODO():
template <size_t VolumeDim, typename... Tags>
class SimpleWeno<VolumeDim, tmpl::list<Tags...>> {
 public:
  struct NeighborWeight {
    using type = double;
    static type default_value() noexcept { return 0.001; }
    static type lower_bound() noexcept { return 0.0; }
    static type upper_bound() noexcept { return 0.1; }
    static constexpr OptionString help = {
        "Linear weight of neighboring-element solution"};
  };
  /// \brief Turn the limiter off
  ///
  /// This option exists to temporarily disable the limiter for debugging
  /// purposes. For problems where limiting is not needed, the preferred
  /// approach is to not compile the limiter into the executable.
  struct DisableForDebugging {
    using type = bool;
    static type default_value() noexcept { return false; }
    static constexpr OptionString help = {"Disable the limiter"};
  };
  using options = tmpl::list<NeighborWeight, DisableForDebugging>;
  static constexpr OptionString help = {"A simple WENO slope limiter."};

  /// \brief Constuct a SimpleWeno slope limiter
  explicit SimpleWeno(const double neighbor_linear_weight,
                      const bool disable_for_debugging = false) noexcept
      : neighbor_linear_weight_(neighbor_linear_weight),
        disable_for_debugging_(disable_for_debugging) {}

  SimpleWeno() noexcept = default;
  SimpleWeno(const SimpleWeno& /*rhs*/) = default;
  SimpleWeno& operator=(const SimpleWeno& /*rhs*/) = default;
  SimpleWeno(SimpleWeno&& /*rhs*/) noexcept = default;
  SimpleWeno& operator=(SimpleWeno&& /*rhs*/) noexcept = default;
  ~SimpleWeno() = default;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | neighbor_linear_weight_;
    p | disable_for_debugging_;
  }

  /// \brief Data to send to neighbor elements.
  struct PackagedData {
    Variables<tmpl::list<Tags...>> volume_data;
    Mesh<VolumeDim> mesh;
    std::array<double, VolumeDim> element_size;
    // TODO(): add some measure of refinement depth

    // clang-tidy: google-runtime-references
    void pup(PUP::er& p) noexcept {  // NOLINT
      p | volume_data;
      p | mesh;
      p | element_size;
    }
  };

  using package_argument_tags = tmpl::list<Tags..., ::Tags::Mesh<VolumeDim>,
                                           ::Tags::SizeOfElement<VolumeDim>>;

  /// \brief Package data for sending to neighbor elements.
  void package_data(const gsl::not_null<PackagedData*>& packaged_data,
                    const db::item_type<Tags>&... tensors,
                    const Mesh<VolumeDim>& mesh,
                    const std::array<double, VolumeDim>& element_size,
                    const OrientationMap<VolumeDim>& orientation_map) const
      noexcept {
    if (disable_for_debugging_) {
      // Do not initialize packaged_data
      return;
    }

    (packaged_data->volume_data).initialize(mesh.number_of_grid_points());
    const auto wrap_copy_tensor = [&packaged_data](
        auto tag, const auto& tensor) noexcept {
      // TODO(): reorient
      get<decltype(tag)>(packaged_data->volume_data) = tensor;
      return '0';
    };
    expand_pack(wrap_copy_tensor(Tags{}, tensors)...);
    packaged_data->mesh = orientation_map(mesh);
    packaged_data->element_size =
        orientation_map.permute_from_neighbor(element_size);
  }

  using limit_tags = tmpl::list<Tags...>;
  using limit_argument_tags =
      tmpl::list<::Tags::Element<VolumeDim>, ::Tags::Mesh<VolumeDim>,
                 ::Tags::Coordinates<VolumeDim, Frame::Logical>,
                 ::Tags::SizeOfElement<VolumeDim>>;

  /// \brief Limits the solution on the element.
  bool operator()(
      const gsl::not_null<std::add_pointer_t<db::item_type<Tags>>>... tensors,
      const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
      const tnsr::I<DataVector, VolumeDim, Frame::Logical>& /*logical_coords*/,
      const std::array<double, VolumeDim>& element_size,
      const std::unordered_map<
          std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
          boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
          neighbor_data) const noexcept {
    if (disable_for_debugging_) {
      // Do not modify input tensors
      return false;
    }

    // Troubled-cell detection for SimpleWeno flags the cell for limiting if
    // any component of any tensor needs limiting.
    bool cell_is_troubled = false;

    // TODO: this is copied from Minmod.hpp.... find way to avoid duplication
    // Allocate temporary buffer to be used in `limit_one_tensor` where we
    // otherwise make 1 + 2 * VolumeDim allocations per tensor component for
    // MUSCL and LambdaPi1, and 1 + 4 * VolumeDim allocations per tensor
    // component for LambdaPiN.
    const size_t half_number_boundary_points = alg::accumulate(
        alg::iota(std::array<size_t, VolumeDim>{{}}, 0_st),
        0_st, [&mesh](const size_t state, const size_t d) noexcept {
          return state + mesh.slice_away(d).number_of_grid_points();
        });
    std::unique_ptr<double[], decltype(&free)> temp_buffer(
        static_cast<double*>(
            malloc(sizeof(double) * (mesh.number_of_grid_points() +
                                     half_number_boundary_points))),
        &free);
    size_t alloc_offset = 0;
    DataVector u_lin(temp_buffer.get() + alloc_offset,
                     mesh.number_of_grid_points());
    alloc_offset += mesh.number_of_grid_points();
    std::array<DataVector, VolumeDim> temp_boundary_buffer{};
    for (size_t d = 0; d < VolumeDim; ++d) {
      const size_t num_points = mesh.slice_away(d).number_of_grid_points();
      temp_boundary_buffer[d].set_data_ref(temp_buffer.get() + alloc_offset,
                                           num_points);
      alloc_offset += num_points;
    }
    // Compute the slice indices once since this is (surprisingly) expensive
    const auto indices_and_buffer = volume_and_slice_indices(mesh.extents());
    const auto volume_and_slice_indices = indices_and_buffer.second;

    const auto wrap_tci_one_tensor = [
      &cell_is_troubled, &element, &mesh, &element_size, &neighbor_data, &u_lin,
      &temp_boundary_buffer, &volume_and_slice_indices
    ](auto tag, const auto& tensor) noexcept {
      if (cell_is_troubled) {
        // If a prior tag flagged cell for limiting, no need to check this tag
        return '0';
      }
      const auto neighbor_sizes = [&neighbor_data]() noexcept {
        FixedHashMap<
            maximum_number_of_neighbors(VolumeDim),
            std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
            std::array<double, VolumeDim>,
            boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
            result;
        for (const auto& neighbor_and_data : neighbor_data) {
          result.insert(std::make_pair(neighbor_and_data.first,
                                       neighbor_and_data.second.element_size));
        }
        return result;
      }
      ();
      for (size_t t = 0; t < tensor->size(); ++t) {
        const auto neighbor_tensor_component =
            [&neighbor_data, &t ]() noexcept {
          FixedHashMap<
              maximum_number_of_neighbors(VolumeDim),
              std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, double,
              boost::hash<
                  std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
              result;
          for (const auto& neighbor_and_data : neighbor_data) {
            result.insert(std::make_pair(
                neighbor_and_data.first,
                mean_value(
                    get<decltype(tag)>(neighbor_and_data.second.volume_data)[t],
                    neighbor_and_data.second.mesh)));
          }
          return result;
        }
        ();
        double u_mean;
        std::array<double, VolumeDim> u_limited_slopes{};
        cell_is_troubled =
            cell_is_troubled or
            Minmod_detail::minmod_troubled_cell_indicator(
                make_not_null(&((*tensor)[t])), make_not_null(&u_mean),
                make_not_null(&u_limited_slopes), make_not_null(&u_lin),
                make_not_null(&temp_boundary_buffer), volume_and_slice_indices,
                MinmodType::LambdaPiN,
                0.0,  // TODO(): verify TVBM
                element, mesh, element_size, neighbor_tensor_component,
                neighbor_sizes);
        if (cell_is_troubled) {
          // No need to check further components
          break;
        }
      }
      return '0';
    };
    expand_pack(wrap_tci_one_tensor(Tags{}, tensors)...);

    if (not cell_is_troubled) {
      // No limiting is needed
      return false;
    }

    // Extrapolate all data
    std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        Variables<tmpl::list<Tags...>>,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
        extrapolated_neighbor_vars;
    for (const auto& neighbor_and_data : neighbor_data) {
      const auto& neighbor = neighbor_and_data.first;
      const auto& data = neighbor_and_data.second;

      // Interpolate from neighbor onto self
      // Actually implement this by interpolating ---
      // - from the neighbor mesh
      // - onto self mesh, in self direction, with coord offset
      const auto& source_mesh = data.mesh;
      const auto target_1d_logical_coords = [&neighbor, &mesh ]() noexcept {
        const auto& direction = neighbor.first;
        auto result = std::array<DataVector, VolumeDim>{{{}}};
        for (size_t d = 0; d < VolumeDim; ++d) {
          gsl::at(result, d) =
              get<0>(logical_coordinates(mesh.slice_through(d)));
          if (d == direction.dimension()) {
            gsl::at(result, d) +=
                (direction.side() == Side::Upper ? -2.0 : 2.0);
          }
        }
        return result;
      }
      ();
      const intrp::RegularGrid<VolumeDim> interpolant(source_mesh,
                                                      target_1d_logical_coords);
      extrapolated_neighbor_vars.insert(
          std::make_pair(neighbor, interpolant.interpolate(data.volume_data)));
    }

    // Compute all smoothness indicators
    Variables<tmpl::list<Tags...>> local_vars(mesh.number_of_grid_points());
    const auto copy_into_vars = [&local_vars](auto tag,
                                              const auto& tensor) noexcept {
      get<decltype(tag)>(local_vars) = *tensor;
      return '0';
    };
    expand_pack(copy_into_vars(Tags{}, tensors)...);
    const auto local_smoothness_indicators =
        SimpleWeno_detail::smoothness_indicator(local_vars, mesh);

    std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        tuples::TaggedTuple<Minmod_detail::to_tensor_double<Tags>...>,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
        neighbor_smoothness_indicators{};
    for (const auto& neighbor_and_vars : extrapolated_neighbor_vars) {
      const auto& neighbor = neighbor_and_vars.first;
      const auto& vars = neighbor_and_vars.second;
      neighbor_smoothness_indicators.insert(std::make_pair(
          neighbor, SimpleWeno_detail::smoothness_indicator(vars, mesh)));
    }

    const auto wrap_limit_one_tensor = [
      this, &extrapolated_neighbor_vars, &mesh, &local_smoothness_indicators,
      &neighbor_smoothness_indicators
    ](auto tag, const auto& tensor) noexcept {
      using DoubleTag = Minmod_detail::to_tensor_double<decltype(tag)>;
      const typename DoubleTag::type tag_smoothness =
          get<DoubleTag>(local_smoothness_indicators);
      std::unordered_map<
          std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
          typename DoubleTag::type,
          boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
          neighbor_tag_smoothness{};
      for (const auto& neighbor_and_tuple : neighbor_smoothness_indicators) {
        const auto& neighbor = neighbor_and_tuple.first;
        const auto& tuple = neighbor_and_tuple.second;
        neighbor_tag_smoothness.insert(
            std::make_pair(neighbor, get<DoubleTag>(tuple)));
      }
      SimpleWeno_detail::limit_one_tensor<VolumeDim, decltype(tag)>(
          tensor, mesh, tag_smoothness, extrapolated_neighbor_vars,
          neighbor_tag_smoothness, neighbor_linear_weight_);
      return '0';
    };
    expand_pack(wrap_limit_one_tensor(Tags{}, tensors)...);
    return true;  // cell_is_troubled
  }

  const double& neighbor_linear_weight() const noexcept {
    return neighbor_linear_weight_;
  }
  const bool& disable_for_debugging() const noexcept {
    return disable_for_debugging_;
  }

 private:
  double neighbor_linear_weight_;
  bool disable_for_debugging_;
};

template <size_t VolumeDim, typename TagList>
SPECTRE_ALWAYS_INLINE bool operator==(
    const SimpleWeno<VolumeDim, TagList>& lhs,
    const SimpleWeno<VolumeDim, TagList>& rhs) noexcept {
  return lhs.neighbor_linear_weight() == rhs.neighbor_linear_weight() and
         lhs.disable_for_debugging() == rhs.disable_for_debugging();
}

template <size_t VolumeDim, typename TagList>
SPECTRE_ALWAYS_INLINE bool operator!=(
    const SimpleWeno<VolumeDim, TagList>& lhs,
    const SimpleWeno<VolumeDim, TagList>& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace SlopeLimiters
