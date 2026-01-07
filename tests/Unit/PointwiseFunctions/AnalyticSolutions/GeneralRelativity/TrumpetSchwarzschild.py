# Distributed under the MIT License.
# See LICENSE.txt for details.

import importlib.util


def check_import() -> bool:
    has_integrate = importlib.util.find_spec("scipy.integrate") is not None
    has_optimize = importlib.util.find_spec("scipy.optimize") is not None

    if has_integrate and has_optimize:
        try:
            module = importlib.import_module("scipy.integrate")
            getattr(module, "tanhsinh")
            module = importlib.import_module("scipy.optimize")
            getattr(module, "root_scalar")
        except (ImportError, AttributeError):
            return False

    return has_integrate and has_optimize


def trumpet_schwarzschild_variables(x, t, mass, n):
    import numpy as np

    # for fixed pts check, we hardcode values obtained by the same python
    # functions above
    if np.array_equal(
        x,
        np.array([0.5680968170041150, 0.4970452133901940, 0.2830205552189050]),
    ):
        return {
            "Lapse": 0.2102394708993822,
            "dt(Lapse)": 0.0000000000000000,
            "deriv(Lapse)": np.array(
                [0.1629867952800197, 0.1426021129056149, 0.0811985068614934]
            ),
            "Shift": np.array(
                [0.0918707439523423, 0.0803805129078502, 0.0457691509325556]
            ),
            "dt(Shift)": np.array(
                [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
            ),
            "deriv(Shift)": np.array(
                [
                    [
                        0.1185516082013154,
                        -0.0377664561206673,
                        -0.0215044488750177,
                    ],
                    [
                        -0.0377664561206673,
                        0.1286736794202534,
                        -0.0188148974963263,
                    ],
                    [
                        -0.0215044488750177,
                        -0.0188148974963263,
                        0.1510033859095480,
                    ],
                ]
            ),
            "SpatialMetric": np.array(
                [
                    [
                        16.2297675079970318,
                        0.0000000000000000,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        16.2297675079970318,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        0.0000000000000000,
                        16.2297675079970318,
                    ],
                ]
            ),
            "dt(SpatialMetric)": np.array(
                [
                    [
                        0.0000000000000000,
                        0.0000000000000000,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        0.0000000000000000,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        0.0000000000000000,
                        0.0000000000000000,
                    ],
                ]
            ),
            "deriv(SpatialMetric)": np.array(
                [
                    [
                        [
                            -22.4089328123191152,
                            -0.0000000000000000,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -22.4089328123191152,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -0.0000000000000000,
                            -22.4089328123191152,
                        ],
                    ],
                    [
                        [
                            -19.6062580499636816,
                            -0.0000000000000000,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -19.6062580499636816,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -0.0000000000000000,
                            -19.6062580499636816,
                        ],
                    ],
                    [
                        [
                            -11.1639220931592629,
                            -0.0000000000000000,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -11.1639220931592629,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -0.0000000000000000,
                            -11.1639220931592629,
                        ],
                    ],
                ]
            ),
            "SqrtDetSpatialMetric": 65.3835426193290488,
            "ExtrinsicCurvature": np.array(
                [
                    [
                        -0.7075730434939116,
                        -2.9154411387039083,
                        -1.6600698438599941,
                    ],
                    [
                        -2.9154411387039083,
                        0.0738162069640711,
                        -1.4524456837047197,
                    ],
                    [
                        -1.6600698438599941,
                        -1.4524456837047197,
                        1.7975931138062093,
                    ],
                ]
            ),
            "InverseSpatialMetric": np.array(
                [
                    [
                        0.0616151771433116,
                        0.0000000000000000,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        0.0616151771433116,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        0.0000000000000000,
                        0.0616151771433116,
                    ],
                ]
            ),
        }
    elif np.array_equal(
        x,
        np.array([4.1805906002464903, 3.0296489917597702, 4.8486321865578601]),
    ):
        return {
            "Lapse": 0.7570367089032295,
            "dt(Lapse)": 0.0000000000000000,
            "deriv(Lapse)": np.array(
                [0.0169496724561394, 0.0122833262038080, 0.0196581620351931]
            ),
            "Shift": np.array(
                [0.0389517377879063, 0.0282280912915661, 0.0451760690342816]
            ),
            "dt(Shift)": np.array(
                [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
            ),
            "deriv(Shift)": np.array(
                [
                    [
                        0.0022751608103024,
                        -0.0051033824348369,
                        -0.0081674228272535,
                    ],
                    [
                        -0.0051033824348369,
                        0.0056188903879402,
                        -0.0059188824498637,
                    ],
                    [
                        -0.0081674228272535,
                        -0.0059188824498637,
                        -0.0001552630897424,
                    ],
                ]
            ),
            "SpatialMetric": np.array(
                [
                    [
                        1.6912774453505286,
                        0.0000000000000000,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        1.6912774453505286,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        0.0000000000000000,
                        1.6912774453505286,
                    ],
                ]
            ),
            "dt(SpatialMetric)": np.array(
                [
                    [
                        0.0000000000000000,
                        0.0000000000000000,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        0.0000000000000000,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        0.0000000000000000,
                        0.0000000000000000,
                    ],
                ]
            ),
            "deriv(SpatialMetric)": np.array(
                [
                    [
                        [
                            -0.0684887677153134,
                            -0.0000000000000000,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -0.0684887677153134,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -0.0000000000000000,
                            -0.0684887677153134,
                        ],
                    ],
                    [
                        [
                            -0.0496334001333051,
                            -0.0000000000000000,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -0.0496334001333051,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -0.0000000000000000,
                            -0.0496334001333051,
                        ],
                    ],
                    [
                        [
                            -0.0794329977067292,
                            -0.0000000000000000,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -0.0794329977067292,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -0.0000000000000000,
                            -0.0794329977067292,
                        ],
                    ],
                ]
            ),
            "SqrtDetSpatialMetric": 2.1994914891050010,
            "ExtrinsicCurvature": np.array(
                [
                    [
                        0.0000254763514301,
                        -0.0114013435617176,
                        -0.0182466422723242,
                    ],
                    [
                        -0.0114013435617176,
                        0.0074956219223144,
                        -0.0132232324686586,
                    ],
                    [
                        -0.0182466422723242,
                        -0.0132232324686586,
                        -0.0054042750416586,
                    ],
                ]
            ),
            "InverseSpatialMetric": np.array(
                [
                    [
                        0.5912690450340297,
                        0.0000000000000000,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        0.5912690450340297,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        0.0000000000000000,
                        0.5912690450340297,
                    ],
                ]
            ),
        }
    elif np.array_equal(
        x,
        np.array(
            [51.5006439411830002, 33.8820166534259002, 13.9650020497035996]
        ),
    ):
        return {
            "Lapse": 0.9688533839546105,
            "dt(Lapse)": 0.0000000000000000,
            "deriv(Lapse)": np.array(
                [0.0003951648239540, 0.0002599769657511, 0.0001071535645805]
            ),
            "Shift": np.array(
                [0.0015024323928771, 0.0009884427739242, 0.0004074021185062]
            ),
            "dt(Shift)": np.array(
                [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
            ),
            "deriv(Shift)": np.array(
                [
                    [
                        -0.0000268204747819,
                        -0.0000368378801109,
                        -0.0000151833073137,
                    ],
                    [
                        -0.0000368378801109,
                        0.0000049376222514,
                        -0.0000099890221148,
                    ],
                    [
                        -0.0000151833073137,
                        -0.0000099890221148,
                        0.0000250559483605,
                    ],
                ]
            ),
            "SpatialMetric": np.array(
                [
                    [
                        1.0647991578520684,
                        0.0000000000000000,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        1.0647991578520684,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        0.0000000000000000,
                        1.0647991578520684,
                    ],
                ]
            ),
            "dt(SpatialMetric)": np.array(
                [
                    [
                        0.0000000000000000,
                        0.0000000000000000,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        0.0000000000000000,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        0.0000000000000000,
                        0.0000000000000000,
                    ],
                ]
            ),
            "deriv(SpatialMetric)": np.array(
                [
                    [
                        [
                            -0.0008550051140581,
                            -0.0000000000000000,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -0.0008550051140581,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -0.0000000000000000,
                            -0.0008550051140581,
                        ],
                    ],
                    [
                        [
                            -0.0005625035979427,
                            -0.0000000000000000,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -0.0005625035979427,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -0.0000000000000000,
                            -0.0005625035979427,
                        ],
                    ],
                    [
                        [
                            -0.0002318446383693,
                            -0.0000000000000000,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -0.0002318446383693,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -0.0000000000000000,
                            -0.0002318446383693,
                        ],
                    ],
                ]
            ),
            "SqrtDetSpatialMetric": 1.0987567307255579,
            "ExtrinsicCurvature": np.array(
                [
                    [
                        -0.0000304751383919,
                        -0.0000404859438680,
                        -0.0000166869137361,
                    ],
                    [
                        -0.0000404859438680,
                        0.0000044279703096,
                        -0.0000109782372769,
                    ],
                    [
                        -0.0000166869137361,
                        -0.0000109782372769,
                        0.0000265386188899,
                    ],
                ]
            ),
            "InverseSpatialMetric": np.array(
                [
                    [
                        0.9391442438940482,
                        0.0000000000000000,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        0.9391442438940482,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        0.0000000000000000,
                        0.9391442438940482,
                    ],
                ]
            ),
        }
    elif np.array_equal(
        x,
        np.array([0.0487902462212653, 0.0127925396464956, 0.0508251522757783]),
    ):
        return {
            "Lapse": 0.0180386580327798,
            "dt(Lapse)": 0.0000000000000000,
            "deriv(Lapse)": np.array(
                [0.1844554327385223, 0.0483632205014491, 0.1921485580235478]
            ),
            "Shift": np.array(
                [0.0129145602077450, 0.0033861280946483, 0.0134531907495737]
            ),
            "dt(Shift)": np.array(
                [0.0000000000000000, 0.0000000000000000, 0.0000000000000000]
            ),
            "deriv(Shift)": np.array(
                [
                    [
                        0.2592362314285327,
                        -0.0014313993410471,
                        -0.0056869934732700,
                    ],
                    [
                        -0.0014313993410471,
                        0.2643202273757710,
                        -0.0014910990435719,
                    ],
                    [
                        -0.0056869934732700,
                        -0.0014910990435719,
                        0.2587713503477229,
                    ],
                ]
            ),
            "SpatialMetric": np.array(
                [
                    [
                        1389.2313517296529426,
                        0.0000000000000000,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        1389.2313517296529426,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        0.0000000000000000,
                        1389.2313517296529426,
                    ],
                ]
            ),
            "dt(SpatialMetric)": np.array(
                [
                    [
                        0.0000000000000000,
                        0.0000000000000000,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        0.0000000000000000,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        0.0000000000000000,
                        0.0000000000000000,
                    ],
                ]
            ),
            "deriv(SpatialMetric)": np.array(
                [
                    [
                        [
                            -25962.1361093563755276,
                            -0.0000000000000000,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -25962.1361093563755276,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -0.0000000000000000,
                            -25962.1361093563755276,
                        ],
                    ],
                    [
                        [
                            -6807.1321874555487739,
                            -0.0000000000000000,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -6807.1321874555487739,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -0.0000000000000000,
                            -6807.1321874555487739,
                        ],
                    ],
                    [
                        [
                            -27044.9449092429713346,
                            -0.0000000000000000,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -27044.9449092429713346,
                            -0.0000000000000000,
                        ],
                        [
                            -0.0000000000000000,
                            -0.0000000000000000,
                            -27044.9449092429713346,
                        ],
                    ],
                ]
            ),
            "SqrtDetSpatialMetric": 51779.9782473547966219,
            "ExtrinsicCurvature": np.array(
                [
                    [
                        -52.7199338403740967,
                        -110.2379588223377738,
                        -437.9787906501542807,
                    ],
                    [
                        -110.2379588223377738,
                        338.8195337372775953,
                        -114.8356788016019578,
                    ],
                    [
                        -437.9787906501542807,
                        -114.8356788016019578,
                        -88.5223405999799411,
                    ],
                ]
            ),
            "InverseSpatialMetric": np.array(
                [
                    [
                        0.0007198225110274,
                        0.0000000000000000,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        0.0007198225110274,
                        0.0000000000000000,
                    ],
                    [
                        0.0000000000000000,
                        0.0000000000000000,
                        0.0007198225110274,
                    ],
                ]
            ),
        }

    # for random pts check, we simply calculate the spacetime quantities
    # using the python functions above
    else:
        integrate = importlib.import_module("scipy.integrate")
        tanhsinh = getattr(integrate, "tanhsinh")
        optimize = importlib.import_module("scipy.optimize")
        root_scalar = getattr(optimize, "root_scalar")

        threshold_lapse = 0.1
        max_isotropic_r = 5000 * mass

        def crit_lapse(n):
            return np.sqrt(
                (np.sqrt(4 + 9 * n * n) - 3 * n)
                / (np.sqrt(4 + 9 * n * n) + 3 * n)
            )

        crit_lapse_ = crit_lapse(n)

        def c_squared(n):
            numerator = 3 * n + np.sqrt(4 + 9 * n**2)
            return (
                numerator**3
                / (128 * n**3)
                * np.exp(-2 * crit_lapse_ / n)
                * mass**4
            )

        c_squared_ = c_squared(n)

        def crit_schwarzschild_r(n, mass):
            return (
                3 * n * n * mass
                + np.sqrt(4 * n * n * mass * mass + 9 * n**4 * mass * mass)
            ) / (4 * n * n)

        crit_schwarzschild_r_ = crit_schwarzschild_r(n, mass)

        def func_of_lapse_and_schwarzschild_r(schwarzschild_r, n, mass, lapse):
            return (
                (1 - lapse * lapse) * schwarzschild_r**4
                - 2 * mass * schwarzschild_r**3
                + c_squared_ * np.exp(2 * lapse / n)
            )

        # This solver may fail when lapse is near crit_lapse,
        # where two branches of solution cross.
        # Not a problem in SpECTRE as we ensure the solver succeeds
        # when initializing the source grid. Then we
        # simply interpolate to the user requested grid pts.

        def schwarzschild_r_from_lapse(n, mass, lapse):
            lapse_arr = np.asarray(lapse)
            sol = np.empty_like(lapse_arr, dtype=float)
            for idx, l in np.ndenumerate(lapse_arr):
                if l == crit_lapse_:
                    return crit_schwarzschild_r_
                elif l > crit_lapse_:
                    lower_bound = crit_schwarzschild_r_
                    upper_bound = max(max_isotropic_r, 4 * mass / (1 - l * l))
                else:
                    lower_bound = 0.0
                    upper_bound = crit_schwarzschild_r_

                sol[idx] = root_scalar(
                    func_of_lapse_and_schwarzschild_r,
                    args=(n, mass, l),
                    bracket=[lower_bound, upper_bound],
                    xtol=1e-15,
                    rtol=1e-15,
                    maxiter=100,
                ).root
            return sol

        def first_integrand_above_threshold(lapse, mass, n):
            return np.log(schwarzschild_r_from_lapse(n, mass, lapse)) / (
                lapse * lapse
            )

        def first_integral_above_threshold(mass, lapse_upper_bound, n):
            res = tanhsinh(
                lambda lapse: first_integrand_above_threshold(lapse, mass, n),
                threshold_lapse,
                lapse_upper_bound,
            )

            return res.integral

        def c_0(mass, n):
            return first_integral_above_threshold(mass, 1.0, n)

        c_0_ = c_0(mass, n)

        def _raw_d_lapse_d_schwarzschild_r(n, mass, lapse, schwarzschild_r):
            return (
                -n
                * (
                    3 * mass
                    - 2 * schwarzschild_r
                    + 2 * schwarzschild_r * lapse * lapse
                )
                / (
                    schwarzschild_r
                    * (
                        schwarzschild_r
                        - 2 * mass
                        + n * schwarzschild_r * lapse
                        - schwarzschild_r * lapse * lapse
                    )
                )
            )

        # This function is undefined near the critical point corresponding to
        # critical lapse or critical schwarzschild r, where two solution
        # branches of func_of_lapse_and_schwarzschild_r cross. However, our
        # chosen branch should be continuously differentiable. Here, we fix
        # this issue by linearly interpolating from two neighboring pts.
        # In SpECTRE, we demand no source grid point is nearly critical before
        # interpolating values to user-requested grid pts (which can be nearly
        # critical). The unit test may fail if one deliberately requests a
        # nearly critical grid point, and it merely reflects that this function
        # here no longer is exact.
        # This subtlety should bear little practical significance.
        def get_d_lapse_d_schwarzschild_r(
            n, mass, lapse, schwarzschild_r, tol=1e-6
        ):
            too_close_to_crit = np.abs(lapse - crit_lapse_) < tol
            too_close_to_crit &= (
                np.abs(schwarzschild_r - crit_schwarzschild_r_) < tol
            )

            if not np.any(too_close_to_crit):
                return _raw_d_lapse_d_schwarzschild_r(
                    n, mass, lapse, schwarzschild_r
                )
            else:
                temp_lapse = np.asarray(lapse, dtype=float)
                temp_schwarzschild_r = np.asarray(schwarzschild_r, dtype=float)

                # compute the full derivative array
                deriv = _raw_d_lapse_d_schwarzschild_r(
                    n, mass, temp_lapse, temp_schwarzschild_r
                )

                # average at the "too close" entries from neighbors
                temp_lapse_p = temp_lapse[too_close_to_crit] + tol
                temp_lapse_m = temp_lapse[too_close_to_crit] - tol
                temp_schwarzschild_r_p = schwarzschild_r_from_lapse(
                    n, mass, temp_lapse_p
                )
                temp_schwarzschild_r_m = schwarzschild_r_from_lapse(
                    n, mass, temp_lapse_m
                )

                deriv_p = _raw_d_lapse_d_schwarzschild_r(
                    n, mass, temp_lapse_p, temp_schwarzschild_r_p
                )
                deriv_m = _raw_d_lapse_d_schwarzschild_r(
                    n, mass, temp_lapse_m, temp_schwarzschild_r_m
                )

                deriv[too_close_to_crit] = 0.5 * (deriv_p + deriv_m)

                return deriv

        def first_integrand_below_threshold(lapse, mass, n):
            schwarzschild_r = schwarzschild_r_from_lapse(n, mass, lapse)
            return (
                -1.0
                / get_d_lapse_d_schwarzschild_r(n, mass, lapse, schwarzschild_r)
                / lapse
                / schwarzschild_r
            )

        def first_integral_below_threshold(mass, lapse_lower_bound, n):
            res = tanhsinh(
                lambda lapse: first_integrand_below_threshold(lapse, mass, n),
                lapse_lower_bound,
                threshold_lapse,
            )

            return res.integral

        def isotropic_r_from_lapse(lapse, mass, n):
            if lapse == 0:
                return 0.0
            elif lapse > threshold_lapse:
                r_val = schwarzschild_r_from_lapse(n, mass, lapse) ** (
                    1.0 / lapse
                )

                integral = first_integral_above_threshold(mass, lapse, n)
            else:
                r_val = schwarzschild_r_from_lapse(
                    n, mass, threshold_lapse
                ) ** (1.0 / threshold_lapse)
                integral = first_integral_below_threshold(mass, lapse, n)

            return r_val * np.exp(integral - c_0_)

        def isotropic_r_from_lapse_minus_target(
            lapse, mass, n, target_isotropic_r
        ):
            return isotropic_r_from_lapse(lapse, mass, n) - target_isotropic_r

        def lapse_from_isotropic_r(target_isotropic_r, mass, n):
            max_lapse = np.sqrt(1.0 - mass / max_isotropic_r)

            if target_isotropic_r == 0:
                return 0.0
            else:
                # define the function whose root we seek
                return root_scalar(
                    isotropic_r_from_lapse_minus_target,
                    method="toms748",
                    bracket=[0.0, max_lapse],
                    args=(mass, n, target_isotropic_r),
                    xtol=1e-15,
                    rtol=1e-15,
                    maxiter=100,
                ).root

        isotropic_r = np.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
        lapse = lapse_from_isotropic_r(isotropic_r, mass, n)
        schwarzschild_r = schwarzschild_r_from_lapse(n, mass, lapse)
        d_lapse_d_schwarzschild_r = get_d_lapse_d_schwarzschild_r(
            n, mass, lapse, schwarzschild_r
        )
        f = 1 - 2 * mass / schwarzschild_r
        sqrt_of_lapse_squared_minus_f = np.sqrt(lapse * lapse - f)

        def trumpet_schwarzschild_dt_lapse():
            return 0.0

        def trumpet_schwarzschild_d_lapse():
            return (
                lapse
                * schwarzschild_r
                * d_lapse_d_schwarzschild_r
                / (isotropic_r * isotropic_r)
                * x
            )

        def trumpet_schwarzschild_shift():
            return np.sqrt(lapse * lapse - f) / schwarzschild_r * x

        def trumpet_schwarzschild_dt_shift():
            return np.zeros_like(x)

        def trumpet_schwarzschild_d_shift():
            isotropic_r_squared = isotropic_r * isotropic_r
            common_factor = (
                lapse
                * lapse
                * d_lapse_d_schwarzschild_r
                / (isotropic_r_squared * sqrt_of_lapse_squared_minus_f)
                - lapse
                * sqrt_of_lapse_squared_minus_f
                / (isotropic_r_squared * schwarzschild_r)
                - lapse
                * mass
                / (
                    isotropic_r_squared
                    * schwarzschild_r
                    * schwarzschild_r
                    * sqrt_of_lapse_squared_minus_f
                )
            )
            diagonal_terms = (
                np.identity(3) * sqrt_of_lapse_squared_minus_f / schwarzschild_r
            )

            return common_factor * np.outer(x, x) + diagonal_terms

        def trumpet_schwarzschild_spatial_metric():
            return (schwarzschild_r / isotropic_r) ** 2 * np.identity(3)

        def trumpet_schwarzschild_dt_spatial_metric():
            return np.zeros((len(x), len(x)))

        def trumpet_schwarzschild_d_spatial_metric():
            d_spatial_metric = np.outer(
                2.0
                * schwarzschild_r
                * schwarzschild_r
                * (lapse - 1)
                / isotropic_r**4
                * x,
                np.identity(3),
            )

            return d_spatial_metric.reshape(3, 3, 3)

        def trumpet_schwarzschild_sqrt_det_spatial_metric():
            return (schwarzschild_r / isotropic_r) ** 3

        def trumpet_schwarzschild_extrinsic_curvature():
            common_factor = (
                schwarzschild_r
                * (
                    (
                        schwarzschild_r * lapse * d_lapse_d_schwarzschild_r
                        - mass / schwarzschild_r
                    )
                    - (
                        sqrt_of_lapse_squared_minus_f
                        * sqrt_of_lapse_squared_minus_f
                    )
                )
                / sqrt_of_lapse_squared_minus_f
                / isotropic_r**4
            )

            return schwarzschild_r * sqrt_of_lapse_squared_minus_f / (
                isotropic_r * isotropic_r
            ) * np.identity(3) + common_factor * np.outer(x, x)

        def trumpet_schwarzschild_inverse_spatial_metric():
            return (isotropic_r / schwarzschild_r) ** 2 * np.identity(3)

        return {
            "Lapse": lapse,
            "dt(Lapse)": trumpet_schwarzschild_dt_lapse(),
            "deriv(Lapse)": trumpet_schwarzschild_d_lapse(),
            "Shift": trumpet_schwarzschild_shift(),
            "dt(Shift)": trumpet_schwarzschild_dt_shift(),
            "deriv(Shift)": trumpet_schwarzschild_d_shift(),
            "SpatialMetric": trumpet_schwarzschild_spatial_metric(),
            "dt(SpatialMetric)": trumpet_schwarzschild_dt_spatial_metric(),
            "deriv(SpatialMetric)": trumpet_schwarzschild_d_spatial_metric(),
            "SqrtDetSpatialMetric": (
                trumpet_schwarzschild_sqrt_det_spatial_metric()
            ),
            "ExtrinsicCurvature": trumpet_schwarzschild_extrinsic_curvature(),
            "InverseSpatialMetric": (
                trumpet_schwarzschild_inverse_spatial_metric()
            ),
        }
