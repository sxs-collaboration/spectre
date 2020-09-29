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

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/SizeOfElement.hpp"
#include "Domain/Structure/Element.hpp"  // IWYU pragma: keep
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/HwenoImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodTci.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodType.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/SimpleWenoImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoGridHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoType.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Options.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
template <size_t VolumeDim>
class Direction;
template <size_t VolumeDim>
class ElementId;

namespace PUP {
class er;
}  // namespace PUP

namespace Limiters {
template <size_t VolumeDim, typename TagsToLimit>
class Weno;
}  // namespace Limiters
/// \endcond

namespace Limiters {
/// \ingroup LimitersGroup
/// \brief A compact-stencil WENO limiter for DG
///
/// Implements the simple WENO limiter of \cite Zhong2013 and the Hermite WENO
/// (HWENO) limiter of \cite Zhu2016 for an arbitrary set of tensors. These
/// limiters require communication only between nearest-neighbor elements, but
/// preserve the full order of the DG solution when the solution is smooth.
/// Full volume data is communicated between neighbors.
///
/// The limiter uses the minmod-based TVB troubled-cell indicator (TCI) of
/// \cite Cockburn1999 to identify elements that need limiting. The simple
/// WENO implementation follows the paper: it checks the TCI independently
/// to each tensor component, so that only certain tensor components may be
/// limited. The HWENO implementation checks the TCI for all tensor components,
/// and if any single component is troubled, then all components are limited.
/// Note that the HWENO paper, because it specializes the limiter to the
/// Newtonian Euler fluid system, uses a more sophisticated TCI that is adapted
/// to the particulars of the fluid system. We instead use the TVB indicator
/// because it is easily applied to a general set of tensors.
///
/// For each tensor component to limit, the new solution is obtained by WENO
/// reconstruction --- a linear combination of the local DG solution and a
/// "modified" solution from each neighbor element. For the simple WENO limiter,
/// the modified solution is obtained by simply extrapolating the neighbor
/// solution onto the troubled element. For the HWENO limiter, the modified
/// solution is obtained by a least-squares fit to the solution across multiple
/// neighboring elements.
///
/// To reconstruct the WENO solution from the local solution and the modified
/// neighbor solutions, the standard WENO procedure is followed. We use the
/// oscillation indicator of \cite Dumbser2007, modified for use on the
/// square/cube grids of SpECTRE. We favor this indicator because portions of
/// the work can be precomputed, leading to an oscillation measure that is
/// efficient to evaluate.
///
/// \warning
/// Limitations:
/// - Does not support h- or p-refinement; this is enforced with ASSERTIONS.
/// - Does not support curved elements; this is not enforced.
template <size_t VolumeDim, typename... Tags>
class Weno<VolumeDim, tmpl::list<Tags...>> {
 public:
  /// \brief The WenoType
  ///
  /// One of `Limiters::WenoType`. See the `Limiters::Weno`
  /// documentation for details.
  struct Type {
    using type = WenoType;
    static constexpr Options::String help = {"Type of WENO limiter"};
  };
  /// \brief The linear weight given to each neighbor
  ///
  /// This linear weight gets combined with the oscillation indicator to
  /// compute the weight for each WENO estimated solution. The standard value
  /// in the literature is 0.001; larger values may be better suited for
  /// problems with strong shocks, and smaller values may be better suited to
  /// smooth problems.
  struct NeighborWeight {
    using type = double;
    static type lower_bound() noexcept { return 1e-6; }
    static type upper_bound() noexcept { return 0.1; }
    static constexpr Options::String help = {
        "Linear weight for each neighbor element's solution"};
  };
  /// \brief The TVB constant for the minmod TCI
  ///
  /// See `Limiters::Minmod` documentation for details.
  struct TvbConstant {
    using type = double;
    static type lower_bound() noexcept { return 0.0; }
    static constexpr Options::String help = {"TVB constant 'm'"};
  };
  /// \brief Turn the limiter off
  ///
  /// This option exists to temporarily disable the limiter for debugging
  /// purposes. For problems where limiting is not needed, the preferred
  /// approach is to not compile the limiter into the executable.
  struct DisableForDebugging {
    using type = bool;
    static type default_value() noexcept { return false; }
    static constexpr Options::String help = {"Disable the limiter"};
  };
  using options =
      tmpl::list<Type, NeighborWeight, TvbConstant, DisableForDebugging>;
  static constexpr Options::String help = {"A WENO limiter for DG"};

  Weno(WenoType weno_type, double neighbor_linear_weight, double tvb_constant,
       bool disable_for_debugging = false) noexcept;

  Weno() noexcept = default;
  Weno(const Weno& /*rhs*/) = default;
  Weno& operator=(const Weno& /*rhs*/) = default;
  Weno(Weno&& /*rhs*/) noexcept = default;
  Weno& operator=(Weno&& /*rhs*/) noexcept = default;
  ~Weno() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

  /// \brief Data to send to neighbor elements
  struct PackagedData {
    Variables<tmpl::list<Tags...>> volume_data;
    tuples::TaggedTuple<::Tags::Mean<Tags>...> means;
    Mesh<VolumeDim> mesh;
    std::array<double, VolumeDim> element_size =
        make_array<VolumeDim>(std::numeric_limits<double>::signaling_NaN());

    // NOLINTNEXTLINE(google-runtime-references)
    void pup(PUP::er& p) noexcept {
      p | volume_data;
      p | means;
      p | mesh;
      p | element_size;
    }
  };

  using package_argument_tags =
      tmpl::list<Tags..., domain::Tags::Mesh<VolumeDim>,
                 domain::Tags::SizeOfElement<VolumeDim>>;

  /// \brief Package data for sending to neighbor elements
  void package_data(gsl::not_null<PackagedData*> packaged_data,
                    const typename Tags::type&... tensors,
                    const Mesh<VolumeDim>& mesh,
                    const std::array<double, VolumeDim>& element_size,
                    const OrientationMap<VolumeDim>& orientation_map) const
      noexcept;

  using limit_tags = tmpl::list<Tags...>;
  using limit_argument_tags =
      tmpl::list<domain::Tags::Mesh<VolumeDim>,
                 domain::Tags::Element<VolumeDim>,
                 domain::Tags::SizeOfElement<VolumeDim>>;

  /// \brief Limit the solution on the element
  bool operator()(
      const gsl::not_null<std::add_pointer_t<typename Tags::type>>... tensors,
      const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
      const std::array<double, VolumeDim>& element_size,
      const std::unordered_map<
          std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
          boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
          neighbor_data) const noexcept;

 private:
  template <size_t LocalDim, typename LocalTagList>
  // NOLINTNEXTLINE(readability-redundant-declaration) false positive
  friend bool operator==(const Weno<LocalDim, LocalTagList>& lhs,
                         const Weno<LocalDim, LocalTagList>& rhs) noexcept;

  WenoType weno_type_;
  double neighbor_linear_weight_;
  double tvb_constant_;
  bool disable_for_debugging_;
};

template <size_t VolumeDim, typename... Tags>
Weno<VolumeDim, tmpl::list<Tags...>>::Weno(
    const WenoType weno_type, const double neighbor_linear_weight,
    const double tvb_constant, const bool disable_for_debugging) noexcept
    : weno_type_(weno_type),
      neighbor_linear_weight_(neighbor_linear_weight),
      tvb_constant_(tvb_constant),
      disable_for_debugging_(disable_for_debugging) {}

template <size_t VolumeDim, typename... Tags>
// NOLINTNEXTLINE(google-runtime-references)
void Weno<VolumeDim, tmpl::list<Tags...>>::pup(PUP::er& p) noexcept {
  p | weno_type_;
  p | neighbor_linear_weight_;
  p | tvb_constant_;
  p | disable_for_debugging_;
}

template <size_t VolumeDim, typename... Tags>
void Weno<VolumeDim, tmpl::list<Tags...>>::package_data(
    const gsl::not_null<PackagedData*> packaged_data,
    const typename Tags::type&... tensors, const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size,
    const OrientationMap<VolumeDim>& orientation_map) const noexcept {
  // By always initializing the PackagedData Variables member, we avoid an
  // assertion that arises from having a default-constructed Variables in a
  // disabled limiter. There is a performance cost, because the package_data()
  // function does non-zero work even for a disabled limiter... but since the
  // limiter should never be disabled in a production simulation, this cost
  // should never matter.
  (packaged_data->volume_data).initialize(mesh.number_of_grid_points());

  if (UNLIKELY(disable_for_debugging_)) {
    // Do not initialize packaged_data
    // (except for the Variables member "volume_data", see above)
    return;
  }

  const auto wrap_compute_means = [&mesh, &packaged_data](
                                      auto tag, const auto tensor) noexcept {
    for (size_t i = 0; i < tensor.size(); ++i) {
      // Compute the mean using the local orientation of the tensor and mesh.
      get<::Tags::Mean<decltype(tag)>>(packaged_data->means)[i] =
          mean_value(tensor[i], mesh);
    }
    return '0';
  };
  expand_pack(wrap_compute_means(Tags{}, tensors)...);

  packaged_data->element_size =
      orientation_map.permute_from_neighbor(element_size);

  const auto wrap_copy_tensor = [&packaged_data](auto tag,
                                                 const auto tensor) noexcept {
    get<decltype(tag)>(packaged_data->volume_data) = tensor;
    return '0';
  };
  expand_pack(wrap_copy_tensor(Tags{}, tensors)...);
  packaged_data->volume_data = orient_variables(
      packaged_data->volume_data, mesh.extents(), orientation_map);

  packaged_data->mesh = orientation_map(mesh);
}

template <size_t VolumeDim, typename... Tags>
bool Weno<VolumeDim, tmpl::list<Tags...>>::operator()(
    const gsl::not_null<std::add_pointer_t<typename Tags::type>>... tensors,
    const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
    const std::array<double, VolumeDim>& element_size,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data) const noexcept {
  if (UNLIKELY(disable_for_debugging_)) {
    // Do not modify input tensors
    return false;
  }

  // Enforce restrictions on h-refinement, p-refinement
  if (UNLIKELY(alg::any_of(element.neighbors(),
                           [](const auto& direction_neighbors) noexcept {
                             return direction_neighbors.second.size() != 1;
                           }))) {
    ERROR("The Weno limiter does not yet support h-refinement");
    // Removing this limitation will require:
    // - Generalizing the computation of the modified neighbor solutions.
    // - Generalizing the WENO weighted sum for multiple neighbors in each
    //   direction.
  }
  alg::for_each(neighbor_data, [&mesh](const auto& neighbor_and_data) noexcept {
    if (UNLIKELY(neighbor_and_data.second.mesh != mesh)) {
      ERROR("The Weno limiter does not yet support p-refinement");
      // Removing this limitation will require generalizing the
      // computation of the modified neighbor solutions.
    }
  });

  if (weno_type_ == WenoType::Hweno) {
    // Troubled-cell detection for HWENO flags the element for limiting if any
    // component of any tensor needs limiting.
    const bool cell_is_troubled =
        Tci::tvb_minmod_indicator<VolumeDim, PackagedData, Tags...>(
            tvb_constant_, (*tensors)..., mesh, element, element_size,
            neighbor_data);
    if (not cell_is_troubled) {
      // No limiting is needed
      return false;
    }

    std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, DataVector,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
        modified_neighbor_solution_buffer{};
    for (const auto& neighbor_and_data : neighbor_data) {
      const auto& neighbor = neighbor_and_data.first;
      modified_neighbor_solution_buffer.insert(
          make_pair(neighbor, DataVector(mesh.number_of_grid_points())));
    }

    EXPAND_PACK_LEFT_TO_RIGHT(Weno_detail::hweno_impl<Tags>(
        make_not_null(&modified_neighbor_solution_buffer), tensors,
        neighbor_linear_weight_, mesh, element, neighbor_data));
    return true;  // cell_is_troubled

  } else if (weno_type_ == WenoType::SimpleWeno) {
    // Buffers and pre-computations for TCI
    Minmod_detail::BufferWrapper<VolumeDim> tci_buffer(mesh);
    const auto effective_neighbor_sizes =
        Minmod_detail::compute_effective_neighbor_sizes(element, neighbor_data);

    // Buffers for simple WENO implementation
    std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        intrp::RegularGrid<VolumeDim>,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
        interpolator_buffer{};
    std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, DataVector,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
        modified_neighbor_solution_buffer{};

    bool some_component_was_limited = false;

    const auto wrap_minmod_tci_and_simple_weno_impl =
        [this, &some_component_was_limited, &tci_buffer, &interpolator_buffer,
         &modified_neighbor_solution_buffer, &mesh, &element, &element_size,
         &neighbor_data,
         &effective_neighbor_sizes](auto tag, const auto tensor) noexcept {
          for (size_t tensor_storage_index = 0;
               tensor_storage_index < tensor->size(); ++tensor_storage_index) {
            // Check TCI
            const auto effective_neighbor_means =
                Minmod_detail::compute_effective_neighbor_means<decltype(tag)>(
                    tensor_storage_index, element, neighbor_data);
            const bool component_needs_limiting = Tci::tvb_minmod_indicator(
                make_not_null(&tci_buffer), tvb_constant_,
                (*tensor)[tensor_storage_index], mesh, element, element_size,
                effective_neighbor_means, effective_neighbor_sizes);

            if (component_needs_limiting) {
              if (modified_neighbor_solution_buffer.empty()) {
                // Allocate the neighbor solution buffers only if the limiter is
                // triggered. This reduces allocation when no limiting occurs.
                for (const auto& neighbor_and_data : neighbor_data) {
                  const auto& neighbor = neighbor_and_data.first;
                  modified_neighbor_solution_buffer.insert(make_pair(
                      neighbor, DataVector(mesh.number_of_grid_points())));
                }
              }
              Weno_detail::simple_weno_impl<decltype(tag)>(
                  make_not_null(&interpolator_buffer),
                  make_not_null(&modified_neighbor_solution_buffer), tensor,
                  neighbor_linear_weight_, tensor_storage_index, mesh, element,
                  neighbor_data);
              some_component_was_limited = true;
            }
          }
          return '0';
        };
    expand_pack(wrap_minmod_tci_and_simple_weno_impl(Tags{}, tensors)...);
    return some_component_was_limited;  // cell_is_troubled
  } else {
    ERROR("WENO limiter not implemented for WenoType: " << weno_type_);
  }

  return false;  // cell_is_troubled
}

template <size_t LocalDim, typename LocalTagList>
bool operator==(const Weno<LocalDim, LocalTagList>& lhs,
                const Weno<LocalDim, LocalTagList>& rhs) noexcept {
  return lhs.weno_type_ == rhs.weno_type_ and
         lhs.neighbor_linear_weight_ == rhs.neighbor_linear_weight_ and
         lhs.tvb_constant_ == rhs.tvb_constant_ and
         lhs.disable_for_debugging_ == rhs.disable_for_debugging_;
}

template <size_t VolumeDim, typename TagList>
bool operator!=(const Weno<VolumeDim, TagList>& lhs,
                const Weno<VolumeDim, TagList>& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Limiters
