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
#include "Evolution/DiscontinuousGalerkin/Limiters/HwenoImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodTci.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodType.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/SimpleWenoImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoGridHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoType.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "Options/String.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
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
/// \endcond

namespace Limiters {
/// \ingroup LimitersGroup
/// \brief A compact-stencil WENO limiter for DG
///
/// Implements the simple WENO limiter of \cite Zhong2013 and the Hermite WENO
/// (HWENO) limiter of \cite Zhu2016. The implementation is system-agnostic and
/// can act on an arbitrary set of tensors.
///
/// #### Summary of the compact-stencil WENO algorithms:
//
/// The compact-stencil WENO limiters require communication only between
/// nearest-neighbor elements, but aim to preserve the full order of the DG
/// solution when the solution is smooth. To achieve this, full volume data is
/// communicated between neighbors.
//
/// For each tensor component to limit, the new solution is obtained by a
/// standard WENO procedure --- the new solution is a linear combination of
/// different polynomials, with weights chosen so that the smoother (i.e., less
/// oscillatory) polynomials contribute the most to the sum.
///
/// For the simple WENO and HWENO limiters, the polynomials used are the local
/// DG solution as well as a "modified" solution from each neighbor element. For
/// the simple WENO limiter, the modified solution is obtained by simply
/// extrapolating the neighbor solution onto the troubled element. For the HWENO
/// limiter, the modified solution is obtained by a least-squares fit to the
/// solution across multiple neighboring elements.
///
/// #### Notes on the SpECTRE implemention of the WENO limiters:
///
/// There are a few differences between the limiters as implemented in SpECTRE
/// and as presented in the references. We list them here and discuss them
/// further below.
/// 1. The choice of basis to represent the DG solution
/// 2. The system-agnostic implementation
/// 3. The oscillation indicator
///
/// Finally, in 4., we will discuss the geometric limitations of the
/// implementation (which are not a deviation from the references).
///
/// ##### 1. The choice of basis
///
/// SpECTRE uses a Legendre basis, rather than the polynomial basis that we
/// understand to be used in the references. Because the construction of the
/// modified neighbor solutions and the WENO sum is geometrically motivated, the
/// overall algorithm should work similarly. However, the precise numerics may
/// differ.
///
/// ##### 2. The system-agnostic implementation
//
/// This implementation can act on an arbitrary set of tensors. To reach this
/// generality, our HWENO implementation uses a different troubled-cell
/// indicator (TCI) than the reference, which instead specializes the TCI to the
/// Newtonian Euler system of equations.
///
/// This implementation uses the minmod-based TVB TCI of \cite Cockburn1999 to
/// identify elements that need limiting. The simple WENO implementation follows
/// its reference: it checks the TCI independently for each tensor component, so
/// that only certain tensor components may be limited. The HWENO implementation
/// checks the TVB TCI for all tensor components, and if any single component is
/// troubled, then all components of all tensors are limited.
///
/// When the evolution system has multiple evolved variables, the recommendation
/// of the references is to apply the limiter to the system's characteristic
/// variables to reduce spurious post-limiting oscillations. In SpECTRE,
/// applying the limiter to the characteristic variables requires specializing
/// the limiter to each evolution system. The system-specific limiter can also
/// implement a system-specific TCI (as the HWENO reference does) to more
/// precisely trigger the limiter.
///
/// ##### 3. The oscillation indicator
///
/// We use the oscillation indicator of \cite Dumbser2007, modified for use on
/// the square/cube grids of SpECTRE. We favor this indicator because portions
/// of the work can be precomputed, leading to an oscillation measure that is
/// efficient to evaluate.
///
/// ##### 4. The geometric limitations
///
/// Does not support non-Legendre bases; this is checked in DEBUG mode. In
/// principle other bases could be supported, but this would require
/// generalizing the many internal algorithms that assume a Legendre basis.
///
/// Does not support h- or p-refinement; this is checked always. In principle
/// this could be supported. The modified neighbor solution algorithm would
/// need to be generalized (reasonable for simple WENO, extremely tedious for
/// HWENO), and the sum of neighbor solutions may need to be updated as well.
///
/// Does not support curved elements; this is not enforced. The code will run
/// but we make no guarantees about the results. Specifically, the limiter acts
/// in the `Frame::ElementLogical` coordinates, because in these coordinates it
/// is straightforward to formulate the algorithm. This means the limiter can
/// operate on generic deformed grids --- however, some things can start to
/// break down, especially on strongly deformed grids:
/// 1. When the Jacobian (from `Frame::ElementLogical` to `Frame::Inertial`)
///    varies across the element, then the limiter fails to be conservative.
///    This is because the integral of a tensor `u` over the element will change
///    after the limiter activates on `u`.
/// 2. When computing the modified neighbor solution for the WENO sum, the
///    extrapolation or fitting procedure may not properly account for the
///    coordinates of the source data. If the coordinate map of the neighbor
///    differs from that of the local element, then the logical-coordinate
///    representation of the neighbor data may be incorrect. This may be a
///    large error at Block boundaries with discontinuous map changes, and may
///    be a small error from smoothly-varying maps that are not sufficiently
///    resolved from one element to the next.
template <size_t VolumeDim, typename TagsToLimit>
class Weno;

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
    static type lower_bound() { return 1e-6; }
    static type upper_bound() { return 0.1; }
    static constexpr Options::String help = {
        "Linear weight for each neighbor element's solution"};
  };
  /// \brief The TVB constant for the minmod TCI
  ///
  /// See `Limiters::Minmod` documentation for details.
  struct TvbConstant {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {"TVB constant 'm'"};
  };
  /// \brief Turn the limiter off
  ///
  /// This option exists to temporarily disable the limiter for debugging
  /// purposes. For problems where limiting is not needed, the preferred
  /// approach is to not compile the limiter into the executable.
  struct DisableForDebugging {
    using type = bool;
    static type suggested_value() { return false; }
    static constexpr Options::String help = {"Disable the limiter"};
  };
  using options =
      tmpl::list<Type, NeighborWeight, TvbConstant, DisableForDebugging>;
  static constexpr Options::String help = {"A WENO limiter for DG"};

  Weno(WenoType weno_type, double neighbor_linear_weight, double tvb_constant,
       bool disable_for_debugging = false);

  Weno() = default;
  Weno(const Weno& /*rhs*/) = default;
  Weno& operator=(const Weno& /*rhs*/) = default;
  Weno(Weno&& /*rhs*/) = default;
  Weno& operator=(Weno&& /*rhs*/) = default;
  ~Weno() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  /// \brief Data to send to neighbor elements
  struct PackagedData {
    Variables<tmpl::list<Tags...>> volume_data;
    tuples::TaggedTuple<::Tags::Mean<Tags>...> means;
    Mesh<VolumeDim> mesh;
    std::array<double, VolumeDim> element_size =
        make_array<VolumeDim>(std::numeric_limits<double>::signaling_NaN());

    // NOLINTNEXTLINE(google-runtime-references)
    void pup(PUP::er& p) {
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
                    const OrientationMap<VolumeDim>& orientation_map) const;

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
          neighbor_data) const;

 private:
  template <size_t LocalDim, typename LocalTagList>
  // NOLINTNEXTLINE(readability-redundant-declaration) false positive
  friend bool operator==(const Weno<LocalDim, LocalTagList>& lhs,
                         const Weno<LocalDim, LocalTagList>& rhs);

  WenoType weno_type_;
  double neighbor_linear_weight_;
  double tvb_constant_;
  bool disable_for_debugging_;
};

template <size_t VolumeDim, typename... Tags>
Weno<VolumeDim, tmpl::list<Tags...>>::Weno(const WenoType weno_type,
                                           const double neighbor_linear_weight,
                                           const double tvb_constant,
                                           const bool disable_for_debugging)
    : weno_type_(weno_type),
      neighbor_linear_weight_(neighbor_linear_weight),
      tvb_constant_(tvb_constant),
      disable_for_debugging_(disable_for_debugging) {}

template <size_t VolumeDim, typename... Tags>
// NOLINTNEXTLINE(google-runtime-references)
void Weno<VolumeDim, tmpl::list<Tags...>>::pup(PUP::er& p) {
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
    const OrientationMap<VolumeDim>& orientation_map) const {
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

  const auto wrap_compute_means = [&mesh, &packaged_data](auto tag,
                                                          const auto tensor) {
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

  const auto wrap_copy_tensor = [&packaged_data](auto tag, const auto tensor) {
    get<decltype(tag)>(packaged_data->volume_data) = tensor;
    return '0';
  };
  expand_pack(wrap_copy_tensor(Tags{}, tensors)...);
  packaged_data->volume_data = orient_variables(
      packaged_data->volume_data, mesh.extents(), orientation_map);

  // Warning: the WENO limiter is currently only tested with aligned meshes.
  // The orientation of the mesh, the `element_size` computed above, and the
  // variables should be carefully tested when used with domains that involve
  // orientation maps
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
        neighbor_data) const {
  if (UNLIKELY(disable_for_debugging_)) {
    // Do not modify input tensors
    return false;
  }

  // Check that basis is LGL or LG
  // A Legendre basis is assumed for the oscillation indicator (used in both
  // SimpleWeno and Hweno) and in the Hweno reconstruction.
  ASSERT(mesh.basis() == make_array<VolumeDim>(Spectral::Basis::Legendre),
         "Unsupported basis: " << mesh);
  ASSERT(mesh.quadrature() ==
                 make_array<VolumeDim>(Spectral::Quadrature::GaussLobatto) or
             mesh.quadrature() ==
                 make_array<VolumeDim>(Spectral::Quadrature::Gauss),
         "Unsupported quadrature: " << mesh);

  // Enforce restrictions on h-refinement, p-refinement
  if (UNLIKELY(
          alg::any_of(element.neighbors(), [](const auto& direction_neighbors) {
            return direction_neighbors.second.size() != 1;
          }))) {
    ERROR("The Weno limiter does not yet support h-refinement");
    // Removing this limitation will require:
    // - Generalizing the computation of the modified neighbor solutions.
    // - Generalizing the WENO weighted sum for multiple neighbors in each
    //   direction.
  }
  alg::for_each(neighbor_data, [&mesh](const auto& neighbor_and_data) {
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
         &effective_neighbor_sizes](auto tag, const auto tensor) {
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
                const Weno<LocalDim, LocalTagList>& rhs) {
  return lhs.weno_type_ == rhs.weno_type_ and
         lhs.neighbor_linear_weight_ == rhs.neighbor_linear_weight_ and
         lhs.tvb_constant_ == rhs.tvb_constant_ and
         lhs.disable_for_debugging_ == rhs.disable_for_debugging_;
}

template <size_t VolumeDim, typename TagList>
bool operator!=(const Weno<VolumeDim, TagList>& lhs,
                const Weno<VolumeDim, TagList>& rhs) {
  return not(lhs == rhs);
}

}  // namespace Limiters
