// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/MinmodType.hpp"
#include "Options/Options.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
template <size_t VolumeDim>
class Direction;
template <size_t VolumeDim>
class Element;
template <size_t VolumeDim>
class ElementId;
template <size_t VolumeDim>
class Mesh;
template <size_t VolumeDim>
class OrientationMap;

namespace boost {
template <class T>
struct hash;
}  // namespace boost

namespace gsl {
template <class T>
class not_null;
}  // namespace gsl

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
/// \brief A general minmod slope limiter
///
/// Provides an implementation for the three minmod-based generalized slope
/// limiters from \cite Cockburn1999 Sec. 2.4: \f$\Lambda\Pi^1\f$,
/// \f$\Lambda\Pi^N\f$, and MUSCL. Below we summarize these three limiters, but
/// the reader should refer to the reference for full details. The limiter has a
/// general implementation that can work on an arbitrary set of tensors; the
/// limiting algorithm is applied to each component of each tensor
/// independently.
///
/// The MUSCL and \f$\Lambda\Pi^1\f$ limiters are both intended for use on
/// piecewise-linear solutions, i.e., on linear-order elements with two points
/// per dimension. These limiters operate by reducing the spatial slope of the
/// tensor components if the data look like they may contain oscillations.
/// Between these two, MUSCL is more dissipative --- it more aggressively
/// reduces the slopes of the data, so it may better handle strong shocks, but
/// it correspondingly produces the most broadening of features in the solution.
///
/// Note that we do not _require_ the MUSCL and \f$\Lambda\Pi^1\f$ limiters to
/// be used on linear-order elements. However, when they are used on a
/// higher-resolution grid, the limiters act to linearize the solution (by
/// discarding higher-order mode content) whether or not the slopes must be
/// reduced.
///
/// The \f$\Lambda\Pi^N\f$ limiter is intended for use with higher-order
/// elements (with more than two points per dimension), where the solution is a
/// piecewise polynomial of higher-than-linear order. This limiter generalizes
/// \f$\Lambda\Pi^1\f$: the post-limiter solution is the linearized solution of
/// \f$\Lambda\Pi^1\f$ in the case that the slopes must be reduced, but is the
/// original (higher-order) data in the case that the slopes are acceptable.
///
/// For all three types of minmod limiter the "total variation bound in the
/// means" (TVBM) correction is implemented, enabling the limiter to avoid
/// limiting away smooth extrema in the solution that would otherwise look like
/// spurious oscillations. The limiter will not reduce the slope (but may still
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
template <size_t VolumeDim, typename... Tags>
class Minmod<VolumeDim, tmpl::list<Tags...>> {
 public:
  /// \brief The MinmodType
  ///
  /// One of `SlopeLimiters::MinmodType`. See `SlopeLimiters::Minmod`
  /// documentation for details. Note in particular that on grids with more than
  /// two points per dimension, the recommended type is `LambdaPiN`.
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
  explicit Minmod(MinmodType minmod_type, double tvbm_constant = 0.0,
                  bool disable_for_debugging = false) noexcept;

  Minmod() noexcept = default;
  Minmod(const Minmod& /*rhs*/) = default;
  Minmod& operator=(const Minmod& /*rhs*/) = default;
  Minmod(Minmod&& /*rhs*/) noexcept = default;
  Minmod& operator=(Minmod&& /*rhs*/) noexcept = default;
  ~Minmod() = default;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

  // To facilitate testing
  /// \cond
  const MinmodType& minmod_type() const noexcept { return minmod_type_; }
  /// \endcond

  /// \brief Data to send to neighbor elements.
  struct PackagedData {
    tuples::TaggedTuple<::Tags::Mean<Tags>...> means;
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
      noexcept;

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
          neighbor_data) const noexcept;

 private:
  template <size_t LocalDim, typename LocalTagList>
  // NOLINTNEXTLINE(readability-redundant-declaration) false positive
  friend bool operator==(const Minmod<LocalDim, LocalTagList>& lhs,
                         const Minmod<LocalDim, LocalTagList>& rhs) noexcept;

  MinmodType minmod_type_;
  double tvbm_constant_;
  bool disable_for_debugging_;
};

template <size_t VolumeDim, typename TagList>
bool operator!=(const Minmod<VolumeDim, TagList>& lhs,
                const Minmod<VolumeDim, TagList>& rhs) noexcept;

}  // namespace SlopeLimiters
