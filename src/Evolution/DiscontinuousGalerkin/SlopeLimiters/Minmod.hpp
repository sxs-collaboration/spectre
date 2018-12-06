// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Element.hpp"  // IWYU pragma: keep
#include "Domain/MaxNumberOfNeighbors.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "Options/Options.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
template <size_t VolumeDim>
class Direction;
template <size_t VolumeDim>
class ElementId;
template <size_t VolumeDim>
class Mesh;
template <size_t VolumeDim>
class OrientationMap;

namespace PUP {
class er;
}  // namespace PUP

namespace SlopeLimiters {
template <size_t VolumeDim, typename TagsToLimit>
class Minmod;
}  // namespace SlopeLimiters

namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
template <size_t VolumeDim>
struct Element;
template <size_t VolumeDim>
struct Mesh;
template <size_t VolumeDim>
struct SizeOfElement;
}  // namespace Tags
/// \endcond

namespace SlopeLimiters {
/// \ingroup SlopeLimitersGroup
/// \brief Possible types of the minmod slope limiter.
///
/// \see SlopeLimiters::Minmod
enum class MinmodType { LambdaPi1, LambdaPiN, Muscl };
}  // namespace SlopeLimiters

namespace Minmod_detail {
// Encodes the return status of the minmod_tvbm function.
struct MinmodResult {
  const double value;
  const bool activated;
};

// The TVBM-corrected minmod function, see e.g. Cockburn reference Eq. 2.26.
MinmodResult minmod_tvbm(double a, double b, double c,
                         double tvbm_scale) noexcept;

// Implements the minmod troubled-cell detector for one component of a
// Tensor<DataVector> at a time.
template <size_t VolumeDim>
bool minmod_troubled_cell_indicator(
    gsl::not_null<DataVector*> tensor_component, gsl::not_null<double*> u_mean,
    gsl::not_null<std::array<double, VolumeDim>*> u_limited_slopes,
    gsl::not_null<DataVector*> u_lin,
    gsl::not_null<std::array<DataVector, VolumeDim>*> temp_boundary_buffer,
    const std::array<std::pair<gsl::span<std::pair<size_t, size_t>>,
                               gsl::span<std::pair<size_t, size_t>>>,
                     VolumeDim>& volume_and_slice_indices,
    const SlopeLimiters::MinmodType& minmod_type, double tvbm_constant,
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    const std::array<double, VolumeDim>& element_size,
    const FixedHashMap<
        maximum_number_of_neighbors(VolumeDim),
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, double,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_tensor_component,
    const FixedHashMap<
        maximum_number_of_neighbors(VolumeDim),
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        std::array<double, VolumeDim>,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_sizes) noexcept;

// Implements the minmod limiter for one Tensor<DataVector>.
//
// The interface is designed to erase the tensor structure information, because
// this way the implementation can be moved out of the header file. This is
// achieved by receiving Tensor<DataVector>::iterators into the tensor to limit,
// and Tensor<double>::iterators into the neighbor tensors.
//
// Note: because the interface erases the tensor structure information, we can
// no longer rely on the compiler to enforce that the local and neighbor tensors
// share the same Structure.
template <size_t VolumeDim>
bool limit_one_tensor(
    gsl::not_null<DataVector*> tensor_begin,
    gsl::not_null<DataVector*> tensor_end, gsl::not_null<DataVector*> u_lin,
    gsl::not_null<std::array<DataVector, VolumeDim>*> temp_boundary_buffer,
    const std::array<std::pair<gsl::span<std::pair<size_t, size_t>>,
                               gsl::span<std::pair<size_t, size_t>>>,
                     VolumeDim>& volume_and_slice_indices,
    const SlopeLimiters::MinmodType& minmod_type, double tvbm_constant,
    const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
    const tnsr::I<DataVector, VolumeDim, Frame::Logical>& logical_coords,
    const std::array<double, VolumeDim>& element_size,
    const FixedHashMap<
        maximum_number_of_neighbors(VolumeDim),
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        gsl::not_null<const double*>,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_tensor_begin,
    const FixedHashMap<
        maximum_number_of_neighbors(VolumeDim),
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        std::array<double, VolumeDim>,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_sizes) noexcept;

template <typename Tag>
struct to_tensor_double : db::PrefixTag, db::SimpleTag {
  using type = TensorMetafunctions::swap_type<double, db::item_type<Tag>>;
  using tag = Tag;
  static std::string name() noexcept {
    return "TensorDouble(" + Tag::name() + ")";
  }
};
}  // namespace Minmod_detail

namespace SlopeLimiters {
/// \ingroup SlopeLimitersGroup
/// \brief A generic Minmod slope limiter
///
/// Implements the minmod-based slope limiter from
/// \ref cockburn_ref "Cockburn (1999)", Section 2.4.
/// Three types of minmod limiter from the reference are implemented:
/// \f$\Lambda\Pi^1\f$, \f$\Lambda\Pi^N\f$, and MUSCL.
///
/// This minmod limiter has a generic implementation that can work on an
/// arbitrary set of tensors. The minmod limiting algorithm is applied to each
/// component of each tensor independently. In general, the limiter linearizes
/// the tensors on every DG element, each time it is applied; additionally, the
/// limiter may reduce the spatial slope of some tensor components if the data
/// look like they may contain oscillations.
///
/// The key features differentiating the three minmod limiter types are:
/// 1. The `Muscl` limiter is the most dissipative; it more aggressively reduces
///    the slopes of the data. This limiter may better handle strong shocks, but
///    also produces the most broadening of features.
/// 2. The `LambdaPiN` limiter is the least aggressive; its "troubled cell"
///    detector tries to avoid limiting in DG elements where the data look
///    smooth enough. Where `LambdaPiN` is able to avoid limiting, the data are
///    _not_ linearized, and the post-limiter data are identical to the
///    pre-limiter data.
/// 3. The `LambdaPi1` limiter is a middle-ground option between the other two.
///    It does not try to avoid limiting as much as `LambdaPiN`, but it allows
///    larger slopes in the data than `Muscl`.
///
/// For all three types of minmod limiter the "total variation bound in the
/// means" (TVBM) correction is implemented, enabling the limiter to avoid
/// limiting away smooth extrema in the solution that would otherwise look like
/// spurious oscillations. The limiter will not reduce the slope (but will still
/// linearize) on elements where the slope is less than \f$m h^2\f$, where
/// \f$m\f$ is the TVBM constant and \f$h\f$ is the size of the DG element.
///
/// The limiter acts in the `Frame::Logical` coordinates, because in these
/// coordinates it is straightforward to formulate the algorithm. This means the
/// limiter can operate on generic deformed grids. However, if the grid is too
/// strongly deformed, some things can start to break down:
/// 1. When an element is deformed so that the Jacobian (from `Frame::Logical`
///    to `Frame::Inertial`) varies across the element, then the limiter fails
///    to be conservative. In other words, the integral of a tensor `u` over the
///    element will change after the limiter activates on `u`. This error is
///    typically small.
/// 2. When there is a sudden change in the size of the elements (perhaps at an
///    h-refinement boundary, or at the boundary between two blocks with very
///    different mappings), a smooth solution in `Frame::Inertial` can appear
///    to have a kink in `Frame::Logical`. The Minmod implementation includes
///    some (untested) tweaks that try to reduce spurious limiter activations
///    near these fake kinks.
///
/// When an element has multiple neighbors in any direction, an effective mean
/// and neighbor size in this direction are computed by averaging over the
/// multiple neighbors. This simple generalization of the minmod limiter enables
/// it to operate on h-refined grids.
///
/// \tparam VolumeDim The number of spatial dimensions.
/// \tparam Tags A typelist of tags specifying the tensors to limit.
///
/// \anchor cockburn_ref [1] B. Cockburn,
/// Discontinuous Galerkin Methods for Convection-Dominated Problems,
/// [Springer (1999)](https://doi.org/10.1007/978-3-662-03882-6_2)
template <size_t VolumeDim, typename... Tags>
class Minmod<VolumeDim, tmpl::list<Tags...>> {
 public:
  /// \brief The MinmodType
  ///
  /// One of `SlopeLimiters::MinmodType`. See `SlopeLimiters::Minmod`
  /// documentation for details.
  struct Type {
    using type = MinmodType;
    static constexpr OptionString help = {"Type of minmod"};
  };
  /// \brief The TVBM constant
  ///
  /// See `SlopeLimiters::Minmod` documentation for details.
  struct TvbmConstant {
    using type = double;
    static type default_value() noexcept { return 0.0; }
    static type lower_bound() noexcept { return 0.0; }
    static constexpr OptionString help = {"TVBM constant 'm'"};
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
  using options = tmpl::list<Type, TvbmConstant, DisableForDebugging>;
  static constexpr OptionString help = {
      "A minmod-based slope limiter.\n"
      "The different types of minmod are more or less aggressive in trying\n"
      "to reduce slopes. The TVBM correction allows the limiter to ignore\n"
      "'small' slopes, and helps to avoid limiting of smooth extrema in the\n"
      "solution.\n"};

  /// \brief Constuct a Minmod slope limiter
  ///
  /// \param minmod_type The type of Minmod slope limiter.
  /// \param tvbm_constant The value of the TVBM constant (default: 0).
  /// \param disable_for_debugging Switch to turn the limiter off (default:
  //         false).
  explicit Minmod(const MinmodType minmod_type,
                  const double tvbm_constant = 0.0,
                  const bool disable_for_debugging = false) noexcept
      : minmod_type_(minmod_type),
        tvbm_constant_(tvbm_constant),
        disable_for_debugging_(disable_for_debugging) {
    ASSERT(tvbm_constant >= 0.0, "The TVBM constant must be non-negative.");
  }

  Minmod() noexcept = default;
  Minmod(const Minmod& /*rhs*/) = default;
  Minmod& operator=(const Minmod& /*rhs*/) = default;
  Minmod(Minmod&& /*rhs*/) noexcept = default;
  Minmod& operator=(Minmod&& /*rhs*/) noexcept = default;
  ~Minmod() = default;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | minmod_type_;
    p | tvbm_constant_;
    p | disable_for_debugging_;
  }

  const MinmodType& minmod_type() const noexcept { return minmod_type_; }
  const double& tvbm_constant() const noexcept { return tvbm_constant_; }
  const bool& disable_for_debugging() const noexcept {
    return disable_for_debugging_;
  }

  /// \brief Data to send to neighbor elements.
  struct PackagedData {
    tuples::TaggedTuple<Minmod_detail::to_tensor_double<Tags>...> means;
    std::array<double, VolumeDim> element_size =
        make_array<VolumeDim>(std::numeric_limits<double>::signaling_NaN());

    // clang-tidy: google-runtime-references
    void pup(PUP::er& p) noexcept {  // NOLINT
      p | means;
      p | element_size;
    }
  };

  using package_argument_tags = tmpl::list<Tags..., ::Tags::Mesh<VolumeDim>,
                                           ::Tags::SizeOfElement<VolumeDim>>;

  /// \brief Package data for sending to neighbor elements.
  ///
  /// The following quantities are stored in `PackagedData` and communicated
  /// between neighboring elements:
  /// - the cell-averaged mean of each tensor component, and
  /// - the size of the cell along each logical coordinate direction.
  ///
  /// \param packaged_data The data package to fill with this element's values.
  /// \param tensors The tensors to be averaged and packaged.
  /// \param mesh The mesh on which the tensor values are measured.
  /// \param element_size The size of the element in inertial coordinates, along
  ///        each dimension of logical coordinates.
  /// \param orientation_map The orientation of the neighbor
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

    const auto wrap_compute_means =
        [&mesh, &packaged_data ](auto tag, const auto& tensor) noexcept {
      for (size_t i = 0; i < tensor.size(); ++i) {
        // Compute the mean using the local orientation of the tensor and mesh:
        // this avoids the work of reorienting the tensor while giving the same
        // result.
        get<Minmod_detail::to_tensor_double<decltype(tag)>>(
            packaged_data->means)[i] = mean_value(tensor[i], mesh);
      }
      return '0';
    };
    expand_pack(wrap_compute_means(Tags{}, tensors)...);
    packaged_data->element_size =
        orientation_map.permute_from_neighbor(element_size);
  }

  using limit_tags = tmpl::list<Tags...>;
  using limit_argument_tags =
      tmpl::list<::Tags::Element<VolumeDim>, ::Tags::Mesh<VolumeDim>,
                 ::Tags::Coordinates<VolumeDim, Frame::Logical>,
                 ::Tags::SizeOfElement<VolumeDim>>;

  /// \brief Limits the solution on the element.
  ///
  /// For each component of each tensor, the limiter will (in general) linearize
  /// the data, then possibly reduce its slope, dimension-by-dimension, until it
  /// no longer looks oscillatory.
  ///
  /// \param tensors The tensors to be limited.
  /// \param element The element on which the tensors to limit live.
  /// \param mesh The mesh on which the tensor values are measured.
  /// \param logical_coords The logical coordinates of the mesh gridpoints.
  /// \param element_size The size of the element, in the inertial coordinates.
  /// \param neighbor_data The data from each neighbor.
  ///
  /// \return whether the limiter modified the solution or not.
  ///
  /// \note The return value is false if the limiter knows it has not modified
  /// the solution. True return values can indicate:
  /// - The solution was limited to reduce the slope, whether by a large factor
  ///   or by a factor only roundoff away from unity.
  /// - The solution was linearized but not limited.
  /// - The solution is identical to the input, if the input was a linear
  ///   function on a higher-order mesh, so that the limiter cannot know that
  ///   the linearization step did not actually modify the data. This is
  ///   somewhat contrived and is unlikely to occur outside of code tests or
  ///   test cases with very clean initial data.
  bool operator()(
      const gsl::not_null<std::add_pointer_t<db::item_type<Tags>>>... tensors,
      const Element<VolumeDim>& element, const Mesh<VolumeDim>& mesh,
      const tnsr::I<DataVector, VolumeDim, Frame::Logical>& logical_coords,
      const std::array<double, VolumeDim>& element_size,
      const std::unordered_map<
          std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
          boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
          neighbor_data) const noexcept {
    if (disable_for_debugging_) {
      // Do not modify input tensors
      return false;
    }

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

    bool limiter_activated = false;
    const auto wrap_limit_one_tensor = [
      this, &limiter_activated, &element, &mesh, &logical_coords, &element_size,
      &neighbor_data, &u_lin, &temp_boundary_buffer, &indices_and_buffer
    ](auto tag, const auto& tensor) noexcept {
      // Because we hide the types of Tags from limit_one_tensor (we do this so
      // that its implementation isn't templated on Tags and can be moved out of
      // this header file), we cannot pass it PackagedData as currently
      // implemented. So we unpack everything from PackagedData. In the future
      // we may want a PackagedData type that erases types inherently, as this
      // would avoid the need for unpacking as done here.
      //
      // Get iterators into the local and neighbor tensors, because these are
      // independent from the structure of the tensor being limited.
      const auto tensor_begin = make_not_null(tensor->begin());
      const auto tensor_end = make_not_null(tensor->end());
      const auto neighbor_tensor_begin = [&neighbor_data]() noexcept {
        FixedHashMap<
            maximum_number_of_neighbors(VolumeDim),
            std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
            gsl::not_null<const double*>,
            boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
            result;
        for (const auto& neighbor_and_data : neighbor_data) {
          result.insert(std::make_pair(
              neighbor_and_data.first,
              make_not_null(get<Minmod_detail::to_tensor_double<decltype(tag)>>(
                                neighbor_and_data.second.means)
                                .cbegin())));
        }
        return result;
      }
      ();
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

      limiter_activated =
          Minmod_detail::limit_one_tensor<VolumeDim>(
              tensor_begin, tensor_end, &u_lin, &temp_boundary_buffer,
              indices_and_buffer.second, minmod_type_, tvbm_constant_, element,
              mesh, logical_coords, element_size, neighbor_tensor_begin,
              neighbor_sizes) or
          limiter_activated;
      return '0';
    };
    expand_pack(wrap_limit_one_tensor(Tags{}, tensors)...);
    return limiter_activated;
  }

 private:
  MinmodType minmod_type_;
  double tvbm_constant_;
  bool disable_for_debugging_;
};

template <size_t VolumeDim, typename TagList>
SPECTRE_ALWAYS_INLINE bool operator==(
    const Minmod<VolumeDim, TagList>& lhs,
    const Minmod<VolumeDim, TagList>& rhs) noexcept {
  return lhs.minmod_type() == rhs.minmod_type() and
         lhs.tvbm_constant() == rhs.tvbm_constant() and
         lhs.disable_for_debugging() == rhs.disable_for_debugging();
}

template <size_t VolumeDim, typename TagList>
SPECTRE_ALWAYS_INLINE bool operator!=(
    const Minmod<VolumeDim, TagList>& lhs,
    const Minmod<VolumeDim, TagList>& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace SlopeLimiters

template <>
struct create_from_yaml<SlopeLimiters::MinmodType> {
  static SlopeLimiters::MinmodType create(const Option& options);
};
