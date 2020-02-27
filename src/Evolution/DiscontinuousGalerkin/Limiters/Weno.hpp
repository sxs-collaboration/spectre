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

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Element.hpp"  // IWYU pragma: keep
#include "Domain/Mesh.hpp"
#include "Domain/OrientationMapHelpers.hpp"
#include "Domain/SizeOfElement.hpp"
#include "Domain/Tags.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/HwenoModifiedSolution.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodTci.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodType.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoGridHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoType.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
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
/// (HWENO) limiter of \cite Zhu2016. These limiters require communication only
/// between nearest-neighbor elements, but preserve the full order of the DG
/// solution when the solution is smooth. Full volume data is communicated
/// between neighbors.
///
/// The limiter uses a Minmod-based troubled-cell indicator to identify elements
/// that need limiting. The \f$\Lambda\Pi^N\f$ limiter of \cite Cockburn1999 is
/// used. Note that the HWENO paper recommends a more sophisticated
/// troubled-cell indicator instead, but the specific choice of indicator should
/// not be too important for a high-order WENO limiter.
///
/// On any identified "troubled" elements, the limited solution is obtained by
/// WENO reconstruction --- a linear combination of the local DG solution and a
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
    static constexpr OptionString help = {"Type of WENO limiter"};
  };
  /// \brief The linear weight given to each neighbor
  ///
  /// This linear weight gets combined with the oscillation indicator to
  /// compute the weight for each WENO estimated solution. Larger values are
  /// better suited for problems with strong shocks, and smaller values are
  /// better suited to smooth problems.
  struct NeighborWeight {
    using type = double;
    static type default_value() noexcept { return 0.001; }
    static type lower_bound() noexcept { return 1e-6; }
    static type upper_bound() noexcept { return 0.1; }
    static constexpr OptionString help = {
        "Linear weight for each neighbor element's solution"};
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
  using options = tmpl::list<Type, NeighborWeight, DisableForDebugging>;
  static constexpr OptionString help = {"A WENO limiter for DG"};

  Weno(WenoType weno_type, double neighbor_linear_weight,
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
                    const db::const_item_type<Tags>&... tensors,
                    const Mesh<VolumeDim>& mesh,
                    const std::array<double, VolumeDim>& element_size,
                    const OrientationMap<VolumeDim>& orientation_map) const
      noexcept;

  using limit_tags = tmpl::list<Tags...>;
  using limit_argument_tags =
      tmpl::list<domain::Tags::Element<VolumeDim>,
                 domain::Tags::Mesh<VolumeDim>,
                 domain::Tags::SizeOfElement<VolumeDim>>;

  /// \brief Limit the solution on the element
  bool operator()(
      const gsl::not_null<std::add_pointer_t<db::item_type<Tags>>>... tensors,
      const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
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
  bool disable_for_debugging_;
};

template <size_t VolumeDim, typename... Tags>
Weno<VolumeDim, tmpl::list<Tags...>>::Weno(
    const WenoType weno_type, const double neighbor_linear_weight,
    const bool disable_for_debugging) noexcept
    : weno_type_(weno_type),
      neighbor_linear_weight_(neighbor_linear_weight),
      disable_for_debugging_(disable_for_debugging) {}

template <size_t VolumeDim, typename... Tags>
// NOLINTNEXTLINE(google-runtime-references)
void Weno<VolumeDim, tmpl::list<Tags...>>::pup(PUP::er& p) noexcept {
  p | weno_type_;
  p | neighbor_linear_weight_;
  p | disable_for_debugging_;
}

template <size_t VolumeDim, typename... Tags>
void Weno<VolumeDim, tmpl::list<Tags...>>::package_data(
    const gsl::not_null<PackagedData*> packaged_data,
    const db::const_item_type<Tags>&... tensors, const Mesh<VolumeDim>& mesh,
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

  const auto wrap_compute_means =
      [&mesh, &packaged_data ](auto tag, const auto tensor) noexcept {
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
    const gsl::not_null<std::add_pointer_t<db::item_type<Tags>>>... tensors,
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
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

  // Troubled-cell detection for WENO flags the element for limiting if any
  // component of any tensor needs limiting.
  const double tci_tvb_constant = 0.0;
  const bool cell_is_troubled =
      Tci::tvb_minmod_indicator<VolumeDim, PackagedData, Tags...>(
          (*tensors)..., neighbor_data, tci_tvb_constant, element, mesh,
          element_size);

  if (not cell_is_troubled) {
    // No limiting is needed
    return false;
  }

  // Compute the modified solutions from each neighbor, for each tensor
  // component. For this step, each WenoType requires a different treatment.
  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
      Variables<tmpl::list<Tags...>>,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      modified_neighbor_solutions;

  if (weno_type_ == WenoType::Hweno) {
    // For each neighbor, the HWENO fits are done one tensor at a time.
    for (const auto& neighbor_and_data : neighbor_data) {
      modified_neighbor_solutions[neighbor_and_data.first].initialize(
          mesh.number_of_grid_points());
    }
    const auto wrap_hweno_neighbor_solution_one_tensor =
        [&element, &mesh, &neighbor_data, &
         modified_neighbor_solutions ](auto tag, const auto tensor) noexcept {
      for (const auto& neighbor_and_data : neighbor_data) {
        const auto& primary_neighbor = neighbor_and_data.first;
        auto& modified_tensor = get<decltype(tag)>(
            modified_neighbor_solutions.at(primary_neighbor));
        hweno_modified_neighbor_solution<decltype(tag)>(
            make_not_null(&modified_tensor), *tensor, element, mesh,
            neighbor_data, primary_neighbor);
      }
      return '0';
    };
    expand_pack(wrap_hweno_neighbor_solution_one_tensor(Tags{}, tensors)...);
  } else if (weno_type_ == WenoType::SimpleWeno) {
    // For each neighbor, the simple WENO data is obtained by extrapolation,
    // with a constant offset added to preserve the correct mean value.
    // The extrapolation step is done on the entire Variables:
    for (const auto& neighbor_and_data : neighbor_data) {
      const auto& neighbor = neighbor_and_data.first;
      const auto& direction = neighbor.first;
      const auto& data = neighbor_and_data.second;
      const auto& source_mesh = data.mesh;
      const auto target_1d_logical_coords =
          Weno_detail::local_grid_points_in_neighbor_logical_coords(
              mesh, source_mesh, element, direction);
      const intrp::RegularGrid<VolumeDim> interpolant(source_mesh, mesh,
                                                      target_1d_logical_coords);
      modified_neighbor_solutions.insert(
          std::make_pair(neighbor, interpolant.interpolate(data.volume_data)));
    }

    // Then the correction is added one tensor component at a time:
    const auto wrap_mean_correction = [&mesh, &modified_neighbor_solutions ](
        auto tag, const auto tensor) noexcept {
      for (size_t i = 0; i < tensor->size(); ++i) {
        const double local_mean = mean_value((*tensor)[i], mesh);
        for (auto& kv : modified_neighbor_solutions) {
          DataVector& neighbor_component_to_correct =
              get<decltype(tag)>(kv.second)[i];
          const double neighbor_mean =
              mean_value(neighbor_component_to_correct, mesh);
          neighbor_component_to_correct += local_mean - neighbor_mean;
        }
      }
      return '0';
    };
    expand_pack(wrap_mean_correction(Tags{}, tensors)...);
  } else {
    ERROR("WENO limiter not implemented for WenoType: " << weno_type_);
  }

  // Reconstruct WENO solution from local solution and modified neighbor
  // solutions.
  const auto wrap_reconstruct_one_tensor =
      [ this, &mesh, &
        modified_neighbor_solutions ](auto tag, const auto tensor) noexcept {
    Weno_detail::reconstruct_from_weighted_sum<decltype(tag)>(
        tensor, mesh, neighbor_linear_weight_, modified_neighbor_solutions,
        Weno_detail::DerivativeWeight::Unity);
    return '0';
  };
  expand_pack(wrap_reconstruct_one_tensor(Tags{}, tensors)...);
  return true;  // cell_is_troubled
}

template <size_t LocalDim, typename LocalTagList>
bool operator==(const Weno<LocalDim, LocalTagList>& lhs,
                const Weno<LocalDim, LocalTagList>& rhs) noexcept {
  return lhs.weno_type_ == rhs.weno_type_ and
         lhs.neighbor_linear_weight_ == rhs.neighbor_linear_weight_ and
         lhs.disable_for_debugging_ == rhs.disable_for_debugging_;
}

template <size_t VolumeDim, typename TagList>
bool operator!=(const Weno<VolumeDim, TagList>& lhs,
                const Weno<VolumeDim, TagList>& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace Limiters
