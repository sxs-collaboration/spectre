// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Spectral.hpp"

#include <algorithm>
#include <cmath>
#include <ostream>
#include <type_traits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/StaticCache.hpp"

namespace Spectral {

std::ostream& operator<<(std::ostream& os, const Basis& basis) {
  switch (basis) {
    case Basis::Legendre:
      return os << "Legendre";
    case Basis::Chebyshev:
      return os << "Chebyshev";
    case Basis::FiniteDifference:
      return os << "FiniteDifference";
    case Basis::SphericalHarmonic:
      return os << "SphericalHarmonic";
    default:
      ERROR("Invalid basis");
  }
}

std::ostream& operator<<(std::ostream& os, const Quadrature& quadrature) {
  switch (quadrature) {
    case Quadrature::Gauss:
      return os << "Gauss";
    case Quadrature::GaussLobatto:
      return os << "GaussLobatto";
    case Quadrature::CellCentered:
      return os << "CellCentered";
    case Quadrature::FaceCentered:
      return os << "FaceCentered";
    case Quadrature::Equiangular:
      return os << "Equiangular";
    default:
      ERROR("Invalid quadrature");
  }
}

Basis to_basis(const std::string& basis) {
  if ("Chebyshev" == basis) {
    return Spectral::Basis::Chebyshev;
  } else if ("Legendre" == basis) {
    return Spectral::Basis::Legendre;
  } else if ("FiniteDifference" == basis) {
    return Spectral::Basis::FiniteDifference;
  } else if ("SphericalHarmonic" == basis) {
    return Spectral::Basis::SphericalHarmonic;
  }
  ERROR("Unknown basis " << basis);
}

Quadrature to_quadrature(const std::string& quadrature) {
  if ("Gauss" == quadrature) {
    return Spectral::Quadrature::Gauss;
  } else if ("GaussLobatto" == quadrature) {
    return Spectral::Quadrature::GaussLobatto;
  } else if ("CellCentered" == quadrature) {
    return Spectral::Quadrature::CellCentered;
  } else if ("FaceCentered" == quadrature) {
    return Spectral::Quadrature::FaceCentered;
  } else if ("Equiangular" == quadrature) {
    return Spectral::Quadrature::Equiangular;
  }
  ERROR("Unknown quadrature " << quadrature);
}

template <Basis BasisType>
Matrix spectral_indefinite_integral_matrix(size_t num_points);

namespace {

// Caching mechanism

template <Basis BasisType, Quadrature QuadratureType,
          typename SpectralQuantityGenerator>
const auto& precomputed_spectral_quantity(const size_t num_points) {
  constexpr size_t max_num_points =
      Spectral::maximum_number_of_points<BasisType>;
  constexpr size_t min_num_points =
      Spectral::minimum_number_of_points<BasisType, QuadratureType>;
  ASSERT(num_points >= min_num_points,
         "Tried to work with less than the minimum number of collocation "
         "points for this quadrature.");
  ASSERT(num_points <= max_num_points,
         "Exceeded maximum number of collocation points.");
  // We compute the quantity for all possible `num_point`s the first time this
  // function is called and keep the data around for the lifetime of the
  // program. The computation is handled by the call operator of the
  // `SpectralQuantityType` instance.
  static const auto precomputed_data =
      make_static_cache<CacheRange<min_num_points, max_num_points + 1>>(
          SpectralQuantityGenerator{});
  return precomputed_data(num_points);
}

template <Basis BasisType, Quadrature QuadratureType>
struct CollocationPointsAndWeightsGenerator {
  std::pair<DataVector, DataVector> operator()(const size_t num_points) const {
    return compute_collocation_points_and_weights<BasisType, QuadratureType>(
        num_points);
  }
};

// Computation of basis-agnostic quantities

template <Basis BasisType, Quadrature QuadratureType>
struct QuadratureWeightsGenerator {
  DataVector operator()(const size_t num_points) const {
    const auto& pts_and_weights = precomputed_spectral_quantity<
        BasisType, QuadratureType,
        CollocationPointsAndWeightsGenerator<BasisType, QuadratureType>>(
        num_points);
    return pts_and_weights.second *
           compute_inverse_weight_function_values<BasisType>(
               pts_and_weights.first);
  }
};

template <Basis BasisType, Quadrature QuadratureType>
struct BarycentricWeightsGenerator {
  DataVector operator()(const size_t num_points) const {
    // Algorithm 30 in Kopriva, p. 75
    // This is valid for any collocation points.
    const DataVector& x =
        collocation_points<BasisType, QuadratureType>(num_points);
    DataVector bary_weights(num_points, 1.);
    for (size_t j = 1; j < num_points; j++) {
      for (size_t k = 0; k < j; k++) {
        bary_weights[k] *= x[k] - x[j];
        bary_weights[j] *= x[j] - x[k];
      }
    }
    for (size_t j = 0; j < num_points; j++) {
      bary_weights[j] = 1. / bary_weights[j];
    }
    return bary_weights;
  }
};

// We don't need this as part of the public interface, but precompute it since
// `interpolation_matrix` needs it at runtime.
template <Basis BasisType, Quadrature QuadratureType>
const DataVector& barycentric_weights(const size_t num_points) {
  return precomputed_spectral_quantity<
      BasisType, QuadratureType,
      BarycentricWeightsGenerator<BasisType, QuadratureType>>(num_points);
}

template <Basis BasisType, Quadrature QuadratureType>
struct DifferentiationMatrixGenerator {
  Matrix operator()(const size_t num_points) const {
    // Algorithm 37 in Kopriva, p. 82
    // It is valid for any collocation points and barycentric weights.
    const DataVector& collocation_pts =
        collocation_points<BasisType, QuadratureType>(num_points);
    Matrix diff_matrix(num_points, num_points);
    if constexpr (BasisType == Spectral::Basis::FiniteDifference) {
      ASSERT(QuadratureType == Spectral::Quadrature::CellCentered,
             "Currently only support cell-centered SBP FD derivatives. Most "
             "likely supporting cell- or vertex-centered just requires "
             "removing this ASSERT and adding tests.");

      // The summation by parts weights come from:
      // arXiv:gr-qc/0512001
      // https://arxiv.org/pdf/gr-qc/0512001.pdf
      //
      // New, efficient, and accurate high order derivative and dissipation
      // operators satisfying summation by parts, and applications in
      // three-dimensional multi-block evolutions
      //
      // by Peter Diener, Ernst Nils Dorband, Erik Schnetter, and Manuel Tiglio
      if (num_points >= 10) {
        // D_{6-5}
        //
        // Left boundary points
        diff_matrix(0, 0) = -2.465354921110524023660777656111276003457;
        diff_matrix(0, 1) = 6.092129526663144141964665936667656020742;
        diff_matrix(0, 2) = -7.730323816657860354911664841669140051855;
        diff_matrix(0, 3) = 6.973765088877147139882219788892186735807;
        diff_matrix(0, 4) = -3.980323816657860354911664841669140051855;
        diff_matrix(0, 5) = 1.292129526663144141964665936667656020742;
        diff_matrix(0, 6) = -0.1820215877771906903274443227779426701237;
        for (size_t i = 7; i < num_points; ++i) {
          diff_matrix(0, i) = 0.0;
        }

        diff_matrix(1, 0) = -0.2234725650784319828746535134412736890421;
        diff_matrix(1, 1) = -0.9329308121107134563129925525068570679651;
        diff_matrix(1, 2) = 1.586820596545839371759081303802027231274;
        diff_matrix(1, 3) = -0.3647002340377160216914505558624668821400;
        diff_matrix(1, 4) = -0.2666957784872806143914117440166232718819;
        diff_matrix(1, 5) = 0.3112949048634705032101261273629794071371;
        diff_matrix(1, 6) = -0.1404504214762266650000768489896480092493;
        diff_matrix(1, 7) = 0.03488568514730479833596013512958238764128;
        diff_matrix(1, 8) = -0.004964021886392518344179263072091597647654;
        diff_matrix(1, 9) = 0.0002126465201465853095969115943714918742904;

        diff_matrix(2, 0) = 0.1582216737061633151406179477554921935333;
        diff_matrix(2, 1) = -1.137049298003377811733609086574457439398;
        diff_matrix(2, 2) = 1.212364522932578587741649981040340946798;
        diff_matrix(2, 3) = -0.9562288729513894906148167047868730813830;
        diff_matrix(2, 4) = 1.066548057336766350478498057851678826640;
        diff_matrix(2, 5) = -0.3478788551267041838265477441805600110467;
        diff_matrix(2, 6) = -0.03133923293520187620333693909408071632123;
        diff_matrix(2, 7) = 0.04098845955755862691072597869183962277781;
        diff_matrix(2, 8) = -0.005963188634687155197078928402509551508436;
        diff_matrix(2, 9) = 0.0003367341182936373038974376991292099082999;

        diff_matrix(3, 0) = 0.02915734641890708196910927068736798144670;
        diff_matrix(3, 1) = -0.1169665089768926152768236581512624861308;
        diff_matrix(3, 2) = -0.1112219092451476301503253995474190870412;
        diff_matrix(3, 3) = -0.7924486261248032107393766820001361351677;
        diff_matrix(3, 4) = 1.266650704820613624987450232358951199911;
        diff_matrix(3, 5) = -0.2899273290506621673153239836530375587273;
        diff_matrix(3, 6) = 0.002515684257201926199329020583484434062150;
        diff_matrix(3, 7) = 0.01329713961871764653006682056620518602804;
        diff_matrix(3, 8) = -0.001124464399630667352932212208930962568134;
        diff_matrix(3, 9) = 0.00006796268169601114882659136477742818715059;

        diff_matrix(4, 0) = -0.04582150000326981674750984653096293434777;
        diff_matrix(4, 1) = 0.2240986548857151482718685516611524323427;
        diff_matrix(4, 2) = -0.3246718493011818141660859125588209338018;
        diff_matrix(4, 3) = -0.3929792921782506986152017485694441380503;
        diff_matrix(4, 4) = 0.1166355818729375628072830916953646214341;
        diff_matrix(4, 5) = 0.3449626905957060254933930895775644438105;
        diff_matrix(4, 6) = 0.1430419813354607083034935179267283951745;
        diff_matrix(4, 7) = -0.07764802499372607792980458731991885121073;
        diff_matrix(4, 8) = 0.01332439335504217034559288889042994978834;
        diff_matrix(4, 9) = -0.0009426355684332077630290447720929851395193;

        diff_matrix(5, 0) = 0.003172814452954821196677290327889903944225;
        diff_matrix(5, 1) = 0.00001061446045061551877105554145609103530766;
        diff_matrix(5, 2) = -0.08747763580209736614983637747947172321794;
        diff_matrix(5, 3) = 0.3975827322299876034907453299884380895682;
        diff_matrix(5, 4) = -1.148835072393422871630425744497391344782;
        diff_matrix(5, 5) = 0.3583006649535242306065761818925080902380;
        diff_matrix(5, 6) = 0.5647665154270147564019144982190032455071;
        diff_matrix(5, 7) = -0.09698196887272109736153117076061707705561;
        diff_matrix(5, 8) = 0.008843905091972988427261446924164441884143;
        diff_matrix(5, 9) = 0.0006174304523363194998474898440202828786385;

        diff_matrix(6, 0) = -0.008639107540858839028043929986084287776394;
        diff_matrix(6, 1) = 0.04722773954485212324714352753530343274219;
        diff_matrix(6, 2) = -0.1008747537650261142294540111407681552350;
        diff_matrix(6, 3) = 0.08043834953845218736895768965086958762389;
        diff_matrix(6, 4) = 0.1295138674713300902982857323205417604553;
        diff_matrix(6, 5) = -0.7909424166489541737614153656634872155367;
        diff_matrix(6, 6) = 0.03807866847647628589685997987877954466259;
        diff_matrix(6, 7) = 0.7367055699548196242687865288427927434250;
        diff_matrix(6, 8) = -0.1480235854665196220062411065981933720158;
        diff_matrix(6, 9) = 0.01651566843542843794512095516024596165494;
        for (size_t i = 1; i < 7; ++i) {
          for (size_t j = 10; j < num_points; ++j) {
            diff_matrix(i, j) = 0.0;
          }
        }

        // central points
        for (size_t i = 7; i < num_points - 7; ++i) {
          for (size_t j = 0; j < i - 3; ++j) {
            diff_matrix(i, j) = 0.0;
          }
          diff_matrix(i, i - 3) = -1.0 / 60.0;
          diff_matrix(i, i - 2) = 0.15;
          diff_matrix(i, i - 1) = -0.75;
          diff_matrix(i, i) = 0.0;
          diff_matrix(i, i + 1) = 0.75;
          diff_matrix(i, i + 2) = -0.15;
          diff_matrix(i, i + 3) = 1.0 / 60.0;
          for (size_t j = i + 4; j < num_points; ++j) {
            diff_matrix(i, j) = 0.0;
          }
        }

        // Right boundary points. Reverse of left boundary, with opposite sign.
        for (size_t i = std::max(num_points - 7, 7_st); i < num_points; ++i) {
          for (size_t j = 0; j < num_points; ++j) {
            diff_matrix(i, j) =
                -diff_matrix(num_points - i - 1, num_points - j - 1);
          }
        }
      } else if (num_points >= 7) {
        for (size_t i = 0; i < num_points; ++i) {
          for (size_t j = 0; j < num_points; ++j) {
            diff_matrix(i, j) = 0.0;
          }
        }
        // D_{4-3}
        //
        // Left boundary points
        diff_matrix(0, 0) = -2.09329763466349871588733;
        diff_matrix(0, 1) = 4.0398572053206615302160;
        diff_matrix(0, 2) = -3.0597858079809922953240;
        diff_matrix(0, 3) = 1.37319053865399486354933;
        diff_matrix(0, 4) = -0.25996430133016538255400;
        for (size_t i = 5; i < num_points; ++i) {
          diff_matrix(0, i) = 0.0;
        }

        diff_matrix(1, 0) = -0.31641585285940445272297;
        diff_matrix(1, 1) = -0.53930788973980422327388;
        diff_matrix(1, 2) = 0.98517732028644343383297;
        diff_matrix(1, 3) = -0.05264665989297578146709;
        diff_matrix(1, 4) = -0.113807251750624235013258;
        diff_matrix(1, 5) = 0.039879767889849911803103;
        diff_matrix(1, 6) = -0.0028794339334846531588787;

        diff_matrix(2, 0) = 0.13026916185021164524452;
        diff_matrix(2, 1) = -0.87966858995059249256890;
        diff_matrix(2, 2) = 0.38609640961100070000134;
        diff_matrix(2, 3) = 0.31358369072435588745988;
        diff_matrix(2, 4) = 0.085318941913678384633511;
        diff_matrix(2, 5) = -0.039046615792734640274641;
        diff_matrix(2, 6) = 0.0034470016440805155042908;

        diff_matrix(3, 0) = -0.01724512193824647912172;
        diff_matrix(3, 1) = 0.16272288227127504381134;
        diff_matrix(3, 2) = -0.81349810248648813029217;
        diff_matrix(3, 3) = 0.13833269266479833215645;
        diff_matrix(3, 4) = 0.59743854328548053399616;
        diff_matrix(3, 5) = -0.066026434346299887619324;
        diff_matrix(3, 6) = -0.0017244594505194129307249;

        diff_matrix(4, 0) = -0.00883569468552192965061;
        diff_matrix(4, 1) = 0.03056074759203203857284;
        diff_matrix(4, 2) = 0.05021168274530854232278;
        diff_matrix(4, 3) = -0.66307364652444929534068;
        diff_matrix(4, 4) = 0.014878787464005191116088;
        diff_matrix(4, 5) = 0.65882706381707471953820;
        diff_matrix(4, 6) = -0.082568940408449266558615;
        for (size_t i = 1; i < 5; ++i) {
          for (size_t j = 7; j < num_points; ++j) {
            diff_matrix(i, j) = 0.0;
          }
        }

        // central points
        //
        // Coded up for completeness and so we can test different (low-order)
        // operators more easily. This won't always be used.
        for (size_t i = 5; i < num_points - 5; ++i) {
          for (size_t j = 0; j < i - 2; ++j) {
            diff_matrix(i, j) = 0.0;
          }
          diff_matrix(i, i - 2) = 1.0 / 12.0;
          diff_matrix(i, i - 1) = -2.0 / 3.0;
          diff_matrix(i, i) = 0.0;
          diff_matrix(i, i + 1) = 2.0 / 3.0;
          diff_matrix(i, i + 2) = -1.0 / 12.0;
          for (size_t j = i + 3; j < num_points; ++j) {
            diff_matrix(i, j) = 0.0;
          }
        }

        // Right boundary points. Reverse of left boundary, with opposite sign.
        for (size_t i = std::max(num_points - 5, 5_st); i < num_points; ++i) {
          for (size_t j = 0; j < num_points; ++j) {
            diff_matrix(i, j) =
                -diff_matrix(num_points - i - 1, num_points - j - 1);
          }
        }
      } else if (num_points == 6) {
        // D_{4-2}
        //
        // Left boundary points
        diff_matrix(0, 0) = -24.0 / 17.0;
        diff_matrix(0, 1) = 59.0 / 34.0;
        diff_matrix(0, 2) = -4.0 / 17.0;
        diff_matrix(0, 3) = -3.0 / 34.0;
        for (size_t i = 4; i < num_points; ++i) {
          diff_matrix(0, i) = 0.0;
        }

        diff_matrix(1, 0) = -1.0 / 2.0;
        diff_matrix(1, 1) = 0.0;
        diff_matrix(1, 2) = 1.0 / 2.0;
        for (size_t i = 3; i < num_points; ++i) {
          diff_matrix(1, i) = 0.0;
        }

        diff_matrix(2, 0) = 4.0 / 43.0;
        diff_matrix(2, 1) = -59.0 / 86.0;
        diff_matrix(2, 2) = 0.0;
        diff_matrix(2, 3) = 59.0 / 86.0;
        diff_matrix(2, 4) = -4.0 / 43.0;
        for (size_t i = 5; i < num_points; ++i) {
          diff_matrix(2, i) = 0.0;
        }

        diff_matrix(3, 0) = 3.0 / 98.0;
        diff_matrix(3, 1) = 0.0;
        diff_matrix(3, 2) = -59.0 / 98.0;
        diff_matrix(3, 3) = 0.0;
        diff_matrix(3, 4) = 32.0 / 49.0;
        diff_matrix(3, 5) = -4.0 / 49.0;

        // Coded up for completeness and so we can test different (low-order)
        // operators more easily. This won't always be used.
        for (size_t i = 6; i < num_points; ++i) {
          diff_matrix(3, i) = 0.0;
        }

        // central points
        //
        // Coded up for completeness and so we can test different (low-order)
        // operators more easily. This won't always be used.
        for (size_t i = 4; i < num_points - 4; ++i) {
          for (size_t j = 0; j < i - 2; ++j) {
            diff_matrix(i, j) = 0.0;
          }
          diff_matrix(i, i - 2) = 1.0 / 12.0;
          diff_matrix(i, i - 1) = -2.0 / 3.0;
          diff_matrix(i, i) = 0.0;
          diff_matrix(i, i + 1) = 2.0 / 3.0;
          diff_matrix(i, i + 2) = -1.0 / 12.0;
          for (size_t j = i + 3; j < num_points; ++j) {
            diff_matrix(i, j) = 0.0;
          }
        }

        // Right boundary points. Reverse of left boundary, with opposite sign.
        for (size_t i = std::max(num_points - 4, 4_st); i < num_points; ++i) {
          for (size_t j = 0; j < num_points; ++j) {
            diff_matrix(i, j) =
                -diff_matrix(num_points - i - 1, num_points - j - 1);
          }
        }
      } else if (num_points > 1) {
        // D_{2-1}
        //
        diff_matrix(0, 0) = -1.0;
        diff_matrix(0, 1) = 1.0;
        for (size_t i = 2; i < num_points; ++i) {
          diff_matrix(0, i) = 0.0;
        }

        for (size_t i = 1; i < num_points - 1; ++i) {
          for (size_t j = 0; j < i - 1; ++j) {
            diff_matrix(i, j) = 0.0;
          }
          diff_matrix(i, i - 1) = -0.5;
          diff_matrix(i, i) = 0.0;
          diff_matrix(i, i + 1) = 0.5;
          for (size_t j = i + 2; j < num_points; ++j) {
            diff_matrix(i, j) = 0.0;
          }
        }

        // Right boundary points. Reverse of left boundary, with opposite
        // sign.
        for (size_t i = num_points - 1; i < num_points; ++i) {
          for (size_t j = 0; j < num_points; ++j) {
            diff_matrix(i, j) =
                -diff_matrix(num_points - i - 1, num_points - j - 1);
          }
        }
      } else {
        diff_matrix(0, 0) = 0.0;
        return diff_matrix;
      }
      const double inv_delta = 1.0 / (collocation_pts[1] - collocation_pts[0]);
      for (size_t i = 0; i < num_points; ++i) {
        for (size_t j = 0; j < num_points; ++j) {
          diff_matrix(i, j) *= inv_delta;
        }
      }
    } else {
      const DataVector& bary_weights =
          barycentric_weights<BasisType, QuadratureType>(num_points);
      for (size_t i = 0; i < num_points; ++i) {
        double& diagonal = diff_matrix(i, i) = 0.0;
        for (size_t j = 0; j < num_points; ++j) {
          if (LIKELY(i != j)) {
            diff_matrix(i, j) =
                bary_weights[j] /
                (bary_weights[i] * (collocation_pts[i] - collocation_pts[j]));
            diagonal -= diff_matrix(i, j);
          }
        }
      }
    }
    return diff_matrix;
  }
};

template <Basis BasisType, Quadrature QuadratureType>
struct WeakFluxDifferentiationMatrixGenerator {
  Matrix operator()(const size_t num_points) const {
    if (BasisType != Basis::Legendre) {
      // We cannot use a `static_assert` because of the way our instantiation
      // macros work for creating matrices
      ERROR(
          "Weak differentiation matrix only implemented for Legendre "
          "polynomials.");
    }
    const DataVector& weights =
        quadrature_weights<BasisType, QuadratureType>(num_points);

    Matrix weak_diff_matrix =
        DifferentiationMatrixGenerator<BasisType, QuadratureType>{}.operator()(
            num_points);
    transpose(weak_diff_matrix);
    for (size_t i = 0; i < num_points; ++i) {
      for (size_t j = 0; j < num_points; ++j) {
        weak_diff_matrix(i, j) *= weights[j] / weights[i];
      }
    }
    return weak_diff_matrix;
  }
};

template <Basis BasisType, Quadrature QuadratureType>
struct IntegrationMatrixGenerator {
  Matrix operator()(const size_t num_points) const {
    return Spectral::modal_to_nodal_matrix<BasisType, QuadratureType>(
               num_points) *
           spectral_indefinite_integral_matrix<BasisType>(num_points) *
           Spectral::nodal_to_modal_matrix<BasisType, QuadratureType>(
               num_points);
  }
};

template <Basis BasisType, Quadrature QuadratureType>
struct ModalToNodalMatrixGenerator {
  Matrix operator()(const size_t num_points) const {
    // To obtain the Vandermonde matrix we need to compute the basis function
    // values at the collocation points. Constructing the matrix proceeds
    // the same for any basis.
    const DataVector& collocation_pts =
        collocation_points<BasisType, QuadratureType>(num_points);
    Matrix vandermonde_matrix(num_points, num_points);
    for (size_t j = 0; j < num_points; j++) {
      const auto& basis_function_values =
          compute_basis_function_value<BasisType>(j, collocation_pts);
      for (size_t i = 0; i < num_points; i++) {
        vandermonde_matrix(i, j) = basis_function_values[i];
      }
    }
    return vandermonde_matrix;
  }
};

template <Basis BasisType, Quadrature QuadratureType>
struct NodalToModalMatrixGenerator {
  Matrix operator()(const size_t num_points) const {
    // Numerically invert the matrix for this generic case
    return inv(modal_to_nodal_matrix<BasisType, QuadratureType>(num_points));
  }
};

template <Basis BasisType>
struct NodalToModalMatrixGenerator<BasisType, Quadrature::Gauss> {
  using data_type = Matrix;
  Matrix operator()(const size_t num_points) const {
    // For Gauss quadrature we implement the analytic expression
    // \f$\mathcal{V}^{-1}_{ij}=\mathcal{V}_{ji}\frac{w_j}{\gamma_i}\f$
    // (see description of `nodal_to_modal_matrix`).
    const DataVector& weights =
        precomputed_spectral_quantity<
            BasisType, Quadrature::Gauss,
            CollocationPointsAndWeightsGenerator<BasisType, Quadrature::Gauss>>(
            num_points)
            .second;
    const Matrix& vandermonde_matrix =
        modal_to_nodal_matrix<BasisType, Quadrature::Gauss>(num_points);
    Matrix vandermonde_inverse(num_points, num_points);
    // This should be vectorized when the functionality is implemented.
    for (size_t i = 0; i < num_points; i++) {
      for (size_t j = 0; j < num_points; j++) {
        vandermonde_inverse(i, j) =
            vandermonde_matrix(j, i) * weights[j] /
            compute_basis_function_normalization_square<BasisType>(i);
      }
    }
    return vandermonde_inverse;
  }
};

template <Basis BasisType, Quadrature QuadratureType>
struct LinearFilterMatrixGenerator {
  Matrix operator()(const size_t num_points) const {
    // We implement the expression
    // \f$\mathcal{V}^{-1}\cdot\mathrm{diag}(1,1,0,0,...)\cdot\mathcal{V}\f$
    // (see description of `linear_filter_matrix`)
    // which multiplies the first two columns of
    // `nodal_to_modal_matrix` with the first two rows of
    // `modal_to_nodal_matrix`.
    Matrix lin_filter(num_points, num_points);
    dgemm_(
        'N', 'N', num_points, num_points, std::min(size_t{2}, num_points), 1.0,
        modal_to_nodal_matrix<BasisType, QuadratureType>(num_points).data(),
        modal_to_nodal_matrix<BasisType, QuadratureType>(num_points).spacing(),
        nodal_to_modal_matrix<BasisType, QuadratureType>(num_points).data(),
        nodal_to_modal_matrix<BasisType, QuadratureType>(num_points).spacing(),
        0.0, lin_filter.data(), lin_filter.spacing());
    return lin_filter;
  }
};

}  // namespace

// Public interface

template <Basis BasisType, Quadrature QuadratureType>
const DataVector& collocation_points(const size_t num_points) {
  return precomputed_spectral_quantity<
             BasisType, QuadratureType,
             CollocationPointsAndWeightsGenerator<BasisType, QuadratureType>>(
             num_points)
      .first;
}

template <Basis BasisType, Quadrature QuadratureType>
const DataVector& quadrature_weights(const size_t num_points) {
  return precomputed_spectral_quantity<
      BasisType, QuadratureType,
      QuadratureWeightsGenerator<BasisType, QuadratureType>>(num_points);
}

// clang-tidy: Macro arguments should be in parentheses, but we want to append
// template parameters here.
#define PRECOMPUTED_SPECTRAL_QUANTITY(function_name, return_type, \
                                      generator_name)             \
  template <Basis BasisType, Quadrature QuadratureType>           \
  const return_type& function_name(const size_t num_points) {     \
    return precomputed_spectral_quantity<                         \
        BasisType, QuadratureType,                                \
        generator_name<BasisType, QuadratureType>>(/* NOLINT */   \
                                                   num_points);   \
  }

PRECOMPUTED_SPECTRAL_QUANTITY(differentiation_matrix, Matrix,
                              DifferentiationMatrixGenerator)
PRECOMPUTED_SPECTRAL_QUANTITY(weak_flux_differentiation_matrix, Matrix,
                              WeakFluxDifferentiationMatrixGenerator)
PRECOMPUTED_SPECTRAL_QUANTITY(integration_matrix, Matrix,
                              IntegrationMatrixGenerator)
PRECOMPUTED_SPECTRAL_QUANTITY(modal_to_nodal_matrix, Matrix,
                              ModalToNodalMatrixGenerator)
PRECOMPUTED_SPECTRAL_QUANTITY(nodal_to_modal_matrix, Matrix,
                              NodalToModalMatrixGenerator)
PRECOMPUTED_SPECTRAL_QUANTITY(linear_filter_matrix, Matrix,
                              LinearFilterMatrixGenerator)

#undef PRECOMPUTED_SPECTRAL_QUANTITY

template <Basis BasisType, Quadrature QuadratureType, typename T>
Matrix interpolation_matrix(const size_t num_points, const T& target_points) {
  constexpr size_t max_num_points =
      Spectral::maximum_number_of_points<BasisType>;
  constexpr size_t min_num_points =
      Spectral::minimum_number_of_points<BasisType, QuadratureType>;
  ASSERT(num_points >= min_num_points,
         "Tried to work with less than the minimum number of collocation "
         "points for this quadrature.");
  ASSERT(num_points <= max_num_points,
         "Exceeded maximum number of collocation points.");
  const DataVector& collocation_pts =
      collocation_points<BasisType, QuadratureType>(num_points);
  const DataVector& bary_weights =
      barycentric_weights<BasisType, QuadratureType>(num_points);
  const size_t num_target_points = get_size(target_points);
  Matrix interp_matrix(num_target_points, num_points);
  // Algorithm 32 in Kopriva, p. 76
  // This is valid for any collocation points.
  for (size_t k = 0; k < num_target_points; k++) {
    // Check where no interpolation is necessary since a target point
    // matches the original collocation points
    bool row_has_match = false;
    for (size_t j = 0; j < num_points; j++) {
      interp_matrix(k, j) = 0.0;
      if (equal_within_roundoff(get_element(target_points, k),
                                collocation_pts[j])) {
        interp_matrix(k, j) = 1.0;
        row_has_match = true;
      }
    }
    // Perform interpolation for non-matching points
    if (not row_has_match) {
      double sum = 0.0;
      for (size_t j = 0; j < num_points; j++) {
        interp_matrix(k, j) = bary_weights[j] / (get_element(target_points, k) -
                                                 collocation_pts[j]);
        sum += interp_matrix(k, j);
      }
      for (size_t j = 0; j < num_points; j++) {
        interp_matrix(k, j) /= sum;
      }
    }
  }
  return interp_matrix;
}

template <Basis BasisType, Quadrature QuadratureType>
const std::pair<Matrix, Matrix>& boundary_interpolation_matrices(
    const size_t num_points) {
  static_assert(BasisType == Spectral::Basis::Legendre);
  static_assert(
      QuadratureType == Spectral::Quadrature::Gauss,
      "We only compute the boundary interpolation for Gauss quadrature "
      "since for Gauss-Lobatto you can just copy values off the volume.");
  static const auto cache = make_static_cache<
      CacheRange<Spectral::minimum_number_of_points<BasisType, QuadratureType>,
                 Spectral::maximum_number_of_points<BasisType>>>(
      [](const size_t local_num_points) {
        return std::pair<Matrix, Matrix>{
            interpolation_matrix<BasisType, QuadratureType>(local_num_points,
                                                            -1.0),
            interpolation_matrix<BasisType, QuadratureType>(local_num_points,
                                                            1.0)};
      });
  return cache(num_points);
}

const std::pair<Matrix, Matrix>& boundary_interpolation_matrices(
    const Mesh<1>& mesh) {
  ASSERT(mesh.basis(0) == Spectral::Basis::Legendre,
         "We only support DG with a Legendre basis.");
  ASSERT(mesh.quadrature(0) == Spectral::Quadrature::Gauss,
         "We only compute the boundary interpolation for Gauss quadrature "
         "since for Gauss-Lobatto you can just copy values off the volume.");
  return boundary_interpolation_matrices<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::Gauss>(
      mesh.extents(0));
}

template <Basis BasisType, Quadrature QuadratureType>
const std::pair<DataVector, DataVector>& boundary_interpolation_term(
    const size_t num_points) {
  static_assert(BasisType == Spectral::Basis::Legendre);
  static_assert(
      QuadratureType == Spectral::Quadrature::Gauss,
      "We only compute the time derivative correction interpolation for Gauss "
      "quadrature since for Gauss-Lobatto you can just copy values off the "
      "volume.");
  static const auto cache = make_static_cache<
      CacheRange<Spectral::minimum_number_of_points<BasisType, QuadratureType>,
                 Spectral::maximum_number_of_points<BasisType>>>(
      [](const size_t local_num_points) {
        const Matrix interp_matrix =
            interpolation_matrix<BasisType, Quadrature::GaussLobatto>(
                local_num_points == 1 ? 2 : local_num_points,
                collocation_points<BasisType, Quadrature::Gauss>(
                    local_num_points));

        std::pair<DataVector, DataVector> result{DataVector{local_num_points},
                                                 DataVector{local_num_points}};
        for (size_t i = 0; i < local_num_points; ++i) {
          result.first[i] = interp_matrix(i, 0);
          result.second[i] = interp_matrix(i, interp_matrix.columns() - 1);
        }
        return result;
      });
  return cache(num_points);
}

const std::pair<DataVector, DataVector>& boundary_interpolation_term(
    const Mesh<1>& mesh) {
  ASSERT(mesh.basis(0) == Spectral::Basis::Legendre,
         "We only support DG with a Legendre basis.");
  ASSERT(
      mesh.quadrature(0) == Spectral::Quadrature::Gauss,
      "We only compute the time derivative correction interpolation for Gauss "
      "quadrature since for Gauss-Lobatto you can just copy values off the "
      "volume.");
  return boundary_interpolation_term<Spectral::Basis::Legendre,
                                     Spectral::Quadrature::Gauss>(
      mesh.extents(0));
}

template <Basis BasisType, Quadrature QuadratureType>
const std::pair<DataVector, DataVector>& boundary_lifting_term(
    const size_t num_points) {
  static_assert(BasisType == Spectral::Basis::Legendre);
  static_assert(
      QuadratureType == Spectral::Quadrature::Gauss,
      "We only compute the boundary lifting for Gauss quadrature "
      "since for Gauss-Lobatto you can just copy values off the volume.");
  static const auto cache = make_static_cache<
      CacheRange<Spectral::minimum_number_of_points<BasisType, QuadratureType>,
                 Spectral::maximum_number_of_points<BasisType>>>(
      [](const size_t local_num_points) {
        const auto& matrices =
            boundary_interpolation_matrices<BasisType, QuadratureType>(
                local_num_points);
        std::pair<DataVector, DataVector> result{DataVector{local_num_points},
                                                 DataVector{local_num_points}};
        for (size_t i = 0; i < local_num_points; ++i) {
          result.first[i] = matrices.first(0, i);
          result.second[i] = matrices.second(0, i);
        }
        const DataVector& quad_weights =
            quadrature_weights<BasisType, QuadratureType>(local_num_points);
        result.first /= quad_weights;
        result.second /= quad_weights;
        return result;
      });
  return cache(num_points);
}

const std::pair<DataVector, DataVector>& boundary_lifting_term(
    const Mesh<1>& mesh) {
  ASSERT(mesh.basis(0) == Spectral::Basis::Legendre,
         "We only support DG with a Legendre basis.");
  ASSERT(mesh.quadrature(0) == Spectral::Quadrature::Gauss,
         "We only compute the boundary lifting for Gauss quadrature "
         "since for Gauss-Lobatto you can just copy values off the volume.");
  return boundary_lifting_term<Spectral::Basis::Legendre,
                               Spectral::Quadrature::Gauss>(mesh.extents(0));
}

namespace {

template <typename F>
decltype(auto) get_spectral_quantity_for_mesh(F&& f, const Mesh<1>& mesh) {
  const auto num_points = mesh.extents(0);
  // Switch on runtime values of basis and quadrature to select
  // corresponding template specialization. For basis functions spanning
  // multiple dimensions we can generalize this function to take a
  // higher-dimensional Mesh.
  switch (mesh.basis(0)) {
    case Basis::Legendre:
      switch (mesh.quadrature(0)) {
        case Quadrature::Gauss:
          return f(std::integral_constant<Basis, Basis::Legendre>{},
                   std::integral_constant<Quadrature, Quadrature::Gauss>{},
                   num_points);
        case Quadrature::GaussLobatto:
          return f(
              std::integral_constant<Basis, Basis::Legendre>{},
              std::integral_constant<Quadrature, Quadrature::GaussLobatto>{},
              num_points);
        default:
          ERROR("Missing quadrature case for spectral quantity");
      }
    case Basis::Chebyshev:
      switch (mesh.quadrature(0)) {
        case Quadrature::Gauss:
          return f(std::integral_constant<Basis, Basis::Chebyshev>{},
                   std::integral_constant<Quadrature, Quadrature::Gauss>{},
                   num_points);
        case Quadrature::GaussLobatto:
          return f(
              std::integral_constant<Basis, Basis::Chebyshev>{},
              std::integral_constant<Quadrature, Quadrature::GaussLobatto>{},
              num_points);
        default:
          ERROR("Missing quadrature case for spectral quantity");
      }
    case Basis::FiniteDifference:
      switch (mesh.quadrature(0)) {
        case Quadrature::CellCentered:
          return f(
              std::integral_constant<Basis, Basis::FiniteDifference>{},
              std::integral_constant<Quadrature, Quadrature::CellCentered>{},
              num_points);
        case Quadrature::FaceCentered:
          return f(
              std::integral_constant<Basis, Basis::FiniteDifference>{},
              std::integral_constant<Quadrature, Quadrature::FaceCentered>{},
              num_points);
        default:
          ERROR(
              "Only CellCentered and FaceCentered are supported for finite "
              "difference quadrature.");
      }
    default:
      ERROR("Missing basis case for spectral quantity. The missing basis is: "
            << mesh.basis(0));
  }
}

}  // namespace

// clang-tidy: Macro arguments should be in parentheses, but we want to append
// template parameters here.
#define SPECTRAL_QUANTITY_FOR_MESH(function_name, return_type)           \
  const return_type& function_name(const Mesh<1>& mesh) {                \
    return get_spectral_quantity_for_mesh(                               \
        [](const auto basis, const auto quadrature,                      \
           const size_t num_points) -> const return_type& {              \
          return function_name</* NOLINT */ decltype(basis)::value,      \
                               decltype(quadrature)::value>(num_points); \
        },                                                               \
        mesh);                                                           \
  }

SPECTRAL_QUANTITY_FOR_MESH(collocation_points, DataVector)
SPECTRAL_QUANTITY_FOR_MESH(quadrature_weights, DataVector)
SPECTRAL_QUANTITY_FOR_MESH(differentiation_matrix, Matrix)
SPECTRAL_QUANTITY_FOR_MESH(weak_flux_differentiation_matrix, Matrix)
SPECTRAL_QUANTITY_FOR_MESH(integration_matrix, Matrix)
SPECTRAL_QUANTITY_FOR_MESH(modal_to_nodal_matrix, Matrix)
SPECTRAL_QUANTITY_FOR_MESH(nodal_to_modal_matrix, Matrix)
SPECTRAL_QUANTITY_FOR_MESH(linear_filter_matrix, Matrix)

#undef SPECTRAL_QUANTITY_FOR_MESH

template <typename T>
Matrix interpolation_matrix(const Mesh<1>& mesh, const T& target_points) {
  return get_spectral_quantity_for_mesh(
      [target_points](const auto basis, const auto quadrature,
                      const size_t num_points) -> Matrix {
        return interpolation_matrix<decltype(basis)::value,
                                    decltype(quadrature)::value>(num_points,
                                                                 target_points);
      },
      mesh);
}

}  // namespace Spectral

#define BASIS(data) BOOST_PP_TUPLE_ELEM(0, data)
#define QUAD(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE(_, data)                                               \
  template const DataVector&                                               \
      Spectral::collocation_points<BASIS(data), QUAD(data)>(size_t);       \
  template const DataVector&                                               \
      Spectral::quadrature_weights<BASIS(data), QUAD(data)>(size_t);       \
  template const Matrix&                                                   \
      Spectral::differentiation_matrix<BASIS(data), QUAD(data)>(size_t);   \
  template const Matrix&                                                   \
      Spectral::weak_flux_differentiation_matrix<BASIS(data), QUAD(data)>( \
          size_t);                                                         \
  template const Matrix&                                                   \
      Spectral::integration_matrix<BASIS(data), QUAD(data)>(size_t);       \
  template const Matrix&                                                   \
      Spectral::nodal_to_modal_matrix<BASIS(data), QUAD(data)>(size_t);    \
  template const Matrix&                                                   \
      Spectral::modal_to_nodal_matrix<BASIS(data), QUAD(data)>(size_t);    \
  template const Matrix&                                                   \
      Spectral::linear_filter_matrix<BASIS(data), QUAD(data)>(size_t);     \
  template Matrix Spectral::interpolation_matrix<BASIS(data), QUAD(data)>( \
      size_t, const DataVector&);                                          \
  template Matrix Spectral::interpolation_matrix<BASIS(data), QUAD(data)>( \
      size_t, const std::vector<double>&);
template Matrix Spectral::interpolation_matrix(const Mesh<1>&,
                                               const DataVector&);
template Matrix Spectral::interpolation_matrix(const Mesh<1>&,
                                               const std::vector<double>&);
template Matrix Spectral::interpolation_matrix(const Mesh<1>&,
                                               const double&);
GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Spectral::Basis::Chebyshev, Spectral::Basis::Legendre),
                        (Spectral::Quadrature::Gauss,
                         Spectral::Quadrature::GaussLobatto))

#undef BASIS
#undef QUAD
#undef INSTANTIATE

template const DataVector&
    Spectral::collocation_points<Spectral::Basis::FiniteDifference,
                                 Spectral::Quadrature::CellCentered>(size_t);
template const DataVector&
    Spectral::quadrature_weights<Spectral::Basis::FiniteDifference,
                                 Spectral::Quadrature::CellCentered>(size_t);
template const DataVector&
    Spectral::collocation_points<Spectral::Basis::FiniteDifference,
                                 Spectral::Quadrature::FaceCentered>(size_t);
template const DataVector&
    Spectral::quadrature_weights<Spectral::Basis::FiniteDifference,
                                 Spectral::Quadrature::FaceCentered>(size_t);
template const Matrix& Spectral::differentiation_matrix<
    Spectral::Basis::FiniteDifference, Spectral::Quadrature::CellCentered>(
    size_t);

template <>
Spectral::Quadrature
Options::create_from_yaml<Spectral::Quadrature>::create<void>(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  try {
    return Spectral::to_quadrature(type_read);
  } catch (const std::exception& /*e*/) {
    PARSE_ERROR(
        options.context(),
        "Failed to convert \""
            << type_read
            << "\" to Spectral::Quadrature. Must be one "
               "of Gauss, GaussLobatto, CellCentered, or FaceCentered.");
  }
}

template <>
Spectral::Basis Options::create_from_yaml<Spectral::Basis>::create<void>(
    const Options::Option& options) {
  const auto type_read = options.parse_as<std::string>();
  try {
    return Spectral::to_basis(type_read);
  } catch (const std::exception& /*e*/) {
    PARSE_ERROR(options.context(),
                "Failed to convert \""
                    << type_read
                    << "\" to Spectral::Basis. Must be one "
                       "of Chebyshev, Legendre, or FiniteDifference.");
  }
}
