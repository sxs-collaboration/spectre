# Distributed under the MIT License.
# See LICENSE.txt for details.

"""Utilities shared between volume CCE tests.
"""

import os
import re
import unittest

import h5py
import numpy as np
import yaml


def load_cce_volume_field(vol_f, obsid, name):
    """Load a field from a CCE volume file.

    Parameters
    ----------
    vol_f : h5py.File
        An open CCE volume file
    obsid : str | int
        The observation id from which to load the field.
    name : str
        The name of the field to load (e.g. "J", "NewmanPenroseGamma", "Psi2"

    Returns
    -------
    field : np.ndarray
        An array with dtype=np.complex128, and of shape
        (int(number_of_radial_shells), int((l_max + 1) ** 2)). Mode indexing
        always starts at (ℓ=0,m=0), with m indexing fastest, i.e. in the order
        (0,0), (1,-1), (1,0), (1,1), (2,-2), ...
    """

    obsid_path = f"/CceVolumeData/VolumeData.vol/ObservationId{obsid}"
    (number_of_radial_shells, l_max, _) = vol_f[f"{obsid_path}/total_extents"][
        ()
    ]
    field = vol_f[f"{obsid_path}/{name}"][()]
    field = field.view(np.complex128).reshape(
        [int(number_of_radial_shells), int((l_max + 1) ** 2)]
    )
    return field


class CheckCceVolumeBase(unittest.TestCase):
    """A base class for CCE volume unit tests in the Schwarzschild background.

    This is used in both
    AnalyticTestVolumeNewmanPenroseSpinCoefficientsRhoGammaMu.py and
    AnalyticTestVolumeWeylPsi2.py. The common code is collected here to avoid
    repeating ourselves.

    Attributes
    ----------
    input_filename : str
        The path of the yaml file for the test
    vol_f_path : str
        The path to the h5 volume file containing the data to be tested
    vol_f : h5py.File
        The opened volume file object
    one_over_r : h5py.Dataset
        An array (like an ndarray) containing the radial collocation points in
        terms of 1/r
    vol_obsids : list[str]
        A list of the observation id strings found in the volume file
    setUp_completed : bool
        A sentinel tracking if the setUp() function completed successfully. This
        attribute may not be present, so its presence should be detected via
        hasattr()
    """

    def setUp(self):
        if hasattr(self, "setUp_completed") and self.setUp_completed:
            return

        # Parse the yaml file to know what we're looking for
        with open(self.input_filename, "r") as open_input_file:
            parsed_yaml = list(yaml.safe_load_all(open_input_file))

        # parsed_yaml[0] is the testing part of the stream;
        # parsed_yaml[1] is the input file for the Cce executable

        extraction_R = parsed_yaml[1]["Cce"]["ExtractionRadius"]
        self.vol_f_path = os.path.join(
            self.run_directory,
            parsed_yaml[1]["Observers"]["VolumeFileName"] + ".h5",
        )
        self.vol_f = h5py.File(self.vol_f_path, "r")

        obsid_pat = re.compile(r"ObservationId(?P<obsid>.+)")
        one_minus_y_obsids = [
            obsid_pat.match(k).group("obsid")
            for k in self.vol_f["/CceVolumeData/OneMinusY.vol"].keys()
        ]

        one_minus_y = self.vol_f[
            "/CceVolumeData/OneMinusY.vol/ObservationId"
            + one_minus_y_obsids[0]
            + "/OneMinusY"
        ][()]
        R_over_r = 0.5 * one_minus_y
        self.one_over_r = R_over_r / extraction_R

        self.vol_obsids = [
            obsid_pat.match(k).group("obsid")
            for k in self.vol_f["/CceVolumeData/VolumeData.vol"].keys()
        ]

        self.setUp_completed = True
