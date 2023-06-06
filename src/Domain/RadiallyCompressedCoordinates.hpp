// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Tags.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace domain {

/// @{
/*!
 * \ingroup ComputationalDomainGroup
 * \brief Coordinates suitable for visualizing large radii by compressing them
 * logarithmically or inversely.
 *
 * Rescales the coordinates $\boldsymbol{x}$ as
 *
 * \begin{equation}
 * \hat{\boldsymbol{x}} = \frac{\hat{r}}{r} \boldsymbol{x}
 * \text{,}
 * \end{equation}
 *
 * for $r > r_0$, where $r=\sqrt{x^2+y^2+z^2}$ is the Euclidean coordinate
 * radius and $r_0$ is the `inner_radius`.
 * The coordinates are compressed from $r \in [r_0, r_1]$ to $\hat{r} \in [r_0,
 * \hat{r}_1]$, where the `outer_radius` $r_1$ can be incomprehensibly large
 * like $10^9$ and the compressed outer radius $\hat{r}_1$ is reasonably small
 * so it can be visualized well. We choose
 *
 * \begin{equation}
 * \hat{r}_1 = r_0 \log_{10}(r_1)
 * \end{equation}
 *
 * so the compressed outer radius is a multiple of the inner radius and
 * increases with the outer radius as well, but exponentials are tamed.
 *
 * The radial compression map $\hat{r}(r)$ is just the inverse of the
 * `domain::CoordinateMaps::Interval` map, which is also used to distribute grid
 * points radially. Therefore, radial grid points will be distributed linearly
 * in the radially compressed coordinates if you use the same `compression`
 * distribution that you used to distribute radial grid points in the
 * `CoordsFrame`.
 *
 * \see domain::CoordinateMaps::Interval
 */
template <typename DataType, size_t Dim, typename CoordsFrame>
void radially_compressed_coordinates(
    gsl::not_null<tnsr::I<DataType, Dim, CoordsFrame>*> result,
    const tnsr::I<DataType, Dim, CoordsFrame>& coordinates, double inner_radius,
    double outer_radius, CoordinateMaps::Distribution compression);
template <typename DataType, size_t Dim, typename CoordsFrame>
tnsr::I<DataType, Dim, CoordsFrame> radially_compressed_coordinates(
    const tnsr::I<DataType, Dim, CoordsFrame>& coordinates, double inner_radius,
    double outer_radius, CoordinateMaps::Distribution compression);
/// @}

/// Options for radially compressed coordinates
///
/// \see radially_compressed_coordinates
struct RadiallyCompressedCoordinatesOptions {
  static constexpr Options::String help =
      "Define radially compressed coordinates for visualizing large outer "
      "radii.";
  struct InnerRadius {
    using type = double;
    static constexpr Options::String help =
        "Radially compressed coordinates begin at this radius, and coincide "
        "with the original coordinates for smaller radii.";
  };
  struct OuterRadius {
    using type = double;
    static constexpr Options::String help =
        "Outer radius of the domain which will be compressed down to a "
        "comprehensible radius, namely to r_inner * log10(r_outer).";
  };
  struct Compression {
    using type = CoordinateMaps::Distribution;
    static constexpr Options::String help =
        "Compression mode: 'Logarithmic' or 'Inverse'. If you use the same "
        "mode that you used to distribute radial grid points then the "
        "grid points will be distributed linearly in the radially compressed "
        "coordinates.";
  };
  using options = tmpl::list<InnerRadius, OuterRadius, Compression>;
  void pup(PUP::er& p);
  double inner_radius;
  double outer_radius;
  CoordinateMaps::Distribution compression;
};

namespace OptionTags {

struct RadiallyCompressedCoordinates {
  using type = Options::Auto<domain::RadiallyCompressedCoordinatesOptions,
                             Options::AutoLabel::None>;
  static constexpr Options::String help =
      "Define radially compressed coordinates for visualizing large outer "
      "radii.";
};

}  // namespace OptionTags

namespace Tags {

/// Options for radially compressed coordinates, or `std::nullopt` if none
/// were specified in the input file
///
/// \see radially_compressed_coordinates
struct RadiallyCompressedCoordinatesOptions : db::SimpleTag {
  using type = std::optional<domain::RadiallyCompressedCoordinatesOptions>;
  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::RadiallyCompressedCoordinates>;
  static type create_from_options(type value) { return value; };
};

/// @{
/*!
 * \brief Coordinates suitable for visualizing large radii by compressing them
 * logarithmically or inversely.
 *
 * The coordinate map reduces to the identity if no options for radially
 * compressed coordinates were specified in the input file.
 *
 * \see radially_compressed_coordinates
 */
template <size_t Dim, typename CoordsFrame>
struct RadiallyCompressedCoordinates : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, CoordsFrame>;
};

template <size_t Dim, typename CoordsFrame>
struct RadiallyCompressedCoordinatesCompute
    : RadiallyCompressedCoordinates<Dim, CoordsFrame>,
      db::ComputeTag {
  using base = RadiallyCompressedCoordinates<Dim, CoordsFrame>;
  using return_type = tnsr::I<DataVector, Dim, CoordsFrame>;
  using argument_tags = tmpl::list<Tags::Coordinates<Dim, CoordsFrame>,
                                   Tags::RadiallyCompressedCoordinatesOptions>;
  static void function(
      const gsl::not_null<tnsr::I<DataVector, Dim, CoordsFrame>*> result,
      const tnsr::I<DataVector, Dim, CoordsFrame>& coords,
      const std::optional<domain::RadiallyCompressedCoordinatesOptions>&
          options) {
    if (options.has_value()) {
      radially_compressed_coordinates(result, coords, options->inner_radius,
                                      options->outer_radius,
                                      options->compression);
    } else {
      *result = coords;
    }
  }
};
/// @}

}  // namespace Tags
}  // namespace domain
