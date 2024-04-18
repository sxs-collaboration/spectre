// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/IO/FillYlmLegendAndData.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"

namespace ylm {
template <typename Frame>
void fill_ylm_legend_and_data(
    const gsl::not_null<std::vector<std::string>*> legend,
    const gsl::not_null<std::vector<double>*> data,
    const ylm::Strahlkorper<Frame>& strahlkorper, const double time,
    const size_t max_l) {
  ASSERT(max_l >= strahlkorper.l_max(),
         "The Lmax of the Ylm data to write ("
             << strahlkorper.l_max()
             << ") is larger than the maximum value that l can be (" << max_l
             << ").");
  const std::array<double, 3> expansion_center =
      strahlkorper.expansion_center();
  const std::string frame{get_output(Frame{})};
  // we only store half and thus only write half of the coefficients
  const size_t num_coefficients =
      ylm::Spherepack::spectral_size(max_l, max_l) / 2;
  // time + 3 dims of expansion center + Lmax = 5 columns
  const size_t num_columns = num_coefficients + 5;

  legend->reserve(num_columns);
  data->reserve(num_columns);

  legend->emplace_back("Time");
  data->emplace_back(time);
  legend->emplace_back(frame + "ExpansionCenter_x");
  data->emplace_back(expansion_center[0]);
  legend->emplace_back(frame + "ExpansionCenter_y");
  data->emplace_back(expansion_center[1]);
  legend->emplace_back(frame + "ExpansionCenter_z");
  data->emplace_back(expansion_center[2]);
  legend->emplace_back("Lmax");
  data->emplace_back(strahlkorper.l_max());

  const DataVector& ylm_coefficients = strahlkorper.coefficients();
  // fill coefficients for l in [0, l_max]
  // l_max == m_max
  ylm::SpherepackIterator iter(strahlkorper.l_max(), strahlkorper.l_max());
  for (size_t l = 0; l <= strahlkorper.l_max(); l++) {
    for (int m = -static_cast<int>(l); m <= static_cast<int>(l); m++) {
      legend->emplace_back(MakeString{} << "coef(" << l << "," << m << ")");

      iter.set(l, m);
      data->emplace_back(ylm_coefficients[iter()]);
    }
  }
  // fill coefficients for l in [l_max + 1, max_l],
  // i.e. higher order coefficients beyond this Strahlkorper's l_max == 0.0
  data->resize(num_columns, 0.0);
  for (size_t l = strahlkorper.l_max() + 1; l <= max_l; l++) {
    for (int m = -static_cast<int>(l); m <= static_cast<int>(l); m++) {
      legend->emplace_back(MakeString{} << "coef(" << l << "," << m << ")");
    }
  }
}
}  // namespace ylm

#define FRAMETYPE(instantiation_data) BOOST_PP_TUPLE_ELEM(0, instantiation_data)

#define INSTANTIATE(_, instantiation_data)                                  \
  template void ylm::fill_ylm_legend_and_data(                              \
      gsl::not_null<std::vector<std::string>*> legend,                      \
      gsl::not_null<std::vector<double>*> data,                             \
      const ylm::Strahlkorper<FRAMETYPE(instantiation_data)>& strahlkorper, \
      double time, size_t max_l);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Grid, Frame::Inertial, Frame::Distorted))

#undef INSTANTIATE
#undef FRAMETYPE
