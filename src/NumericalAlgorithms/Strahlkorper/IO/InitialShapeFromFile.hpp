// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>

#include "NumericalAlgorithms/Strahlkorper/InitialShape.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace ylm {
template <typename Frame>
class Strahlkorper;

namespace InitialShapes {
/*!
 * \ingroup SurfacesGroup
 * \brief An initial Strahlkorper shape read from an H5 file.
 */
template <typename Frame>
class FromFile : public InitialShape<Frame> {
 public:
  struct H5Filename {
    using type = std::string;
    static constexpr Options::String help = {
        "H5 file containing the Strahlkorper coefficients"};
  };
  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help = {
        "Subfile (without leading slash or .dat extension) within "
        "the H5 file that contains the Strahlkorper coefficients"};
  };
  struct Time {
    using type = double;
    static constexpr Options::String help = {
        "Time at which to read the Strahlkorper coefficients"};
  };
  struct TimeEpsilon {
    using type = double;
    static constexpr Options::String help = {
        "Tolerance for matching the requested time to read the Strahlkorper "
        "coefficients"};
  };
  struct CheckFrame {
    using type = bool;
    static constexpr Options::String help = {
        "Whether to check that the frame in the file matches the requested "
        "frame"};
  };

  using options =
      tmpl::list<H5Filename, SubfileName, Time, TimeEpsilon, CheckFrame>;
  static constexpr Options::String help = {
      "Construct a Strahlkorper from coefficients read from an H5 file."};
  static std::string name() { return "FromFile"; }

  FromFile() = default;
  FromFile(std::string h5_filename, std::string subfile_name, double time,
           double time_epsilon, bool check_frame);

  /// \cond
  explicit FromFile(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(FromFile);  // NOLINT
  /// \endcond

  Strahlkorper<Frame> strahlkorper(
      size_t l_max, const Options::Context& context) const override;

  void pup(PUP::er& p) override;

 private:
  std::string h5_filename_{};
  std::string subfile_name_{};
  double time_{};
  double time_epsilon_{};
  bool check_frame_{};
};
}  // namespace InitialShapes
}  // namespace ylm
