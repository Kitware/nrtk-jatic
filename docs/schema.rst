###########
JSON Schema
###########


Required Parameters
===================

-----------------
Header Parameters
-----------------
- **id** (string)
  The unique identifier of the JSON message
- **name** (string)
  The human interpretable name of the invocation of nrtk

------------------
Dataset Parameters
------------------
- **dataset_dir** (str)
  A filepath to the top folder of a COCO dataset.
- **label_file** (str)
  A filepath to the annotations file for the COCO dataset.
- **output_dir** (str)
  A filepath to the directory where augmented datasets should be saved.
- **gsds** (List[float])
  A list of gsds (pixel/m) where the length of the list is the same as the number of images in the dataset.

---------------
NRTK Parameters
---------------
- **theta_keys** (List[str])
  The PyBSM parameters to be perturbed
- **thetas** (List[List[float]])
  The values for each of the parameters specified in "thetas"

The following parameters will be used to generate a PyBSM configuration to
recreate the input data. These should be as close as possible to the parameters of
sensor and scenario used when capturing the images. If these optional parameters
are not included, nrtk will use default values.

Optional Sensor Parameters
==========================
- **D** (float)
  Effective aperture diameter (m). Defualt value of 0.005.

- **f** (float)
  Focal length (m). Defualt value of 0.014.

- **px** (float)
  Detector center-to-center spacings (pitch) in the x and y directions (m). Defualt value of 0.0000074.

- **optTransWavelengths** (numpy array)
  Spectral bandpass of the camera (m). At minimum, start and end wavelengths should be specified. Defualt value of [3.8e-7, 7.0e-7].

- **opticsTransmission** (float)
  Full system in-band optical transmission (unitless). Loss due to any telescope obscuration should *not* be included in this optical transmission array.

- **eta** (float)
  Relative linear obscuration (unitless).

- **wx** and **wy** (float)
  Detector width in the x and y directions (m).

- **qe** (function of wavelength)
  Quantum efficiency as a function of wavelength (e-/photon).

- **qewavelengths** (numpy array)
  Wavelengths corresponding to the quantum efficiency array (m).

- **darkCurrent** (float)
  Detector dark current (e-/s).

- **maxN** (int)
  Detector electron well capacity (e-).

- **maxWellFill** (float)
  Desired well fill, i.e., maximum well size × desired fill fraction.

- **bitdepth** (int)
  Resolution of the detector ADC in bits (unitless).

- **sx** and **sy** (float)
  Root-mean-squared jitter amplitudes in the x and y directions, respectively (rad).

- **dax** and **day** (float)
  Line-of-sight angular drift rate during one integration time in the x and y directions, respectively (rad/s).

- **otherNoise** (float)
  A catch-all for noise terms that are not explicitly included elsewhere (read noise, photon noise, dark current, quantization noise, etc.).

Optional Scenario Parameters
============================
- **ihaze** (integer)
  MODTRAN code for visibility. Valid options are:
    - 1: Rural extinction with 23 km visibility
    - 2: Rural extinction with 5 km visibility
  Default value of 2.

- **altitude** (integer)
  Sensor height above ground level in meters. The database includes the following altitude options:
    - 2, 32.55, 75, 150, 225, 500 meters
    - 1000 to 12000 in 1000 meter steps
    - 14000 to 20000 in 2000 meter steps
    - 24500
  Default value of 75.

- **groundRange** (integer)
  Distance on the ground between the target and sensor in meters. The following ground ranges are included in the database at each altitude until the ground range exceeds the distance to the spherical earth horizon:
    - 0, 100, 500 meters
    - 1000 to 20000 in 1000 meter steps
    - 22000 to 80000 in 2000 meter steps
    - 85000 to 300000 in 5000 meter steps
  Default value of 0.

- **aircraftSpeed** (float)
  Ground speed of the aircraft in meters per second (m/s).

- **targetReflectance** (float)
  Object reflectance (unitless).

- **targetTemperature** (float)
  Object temperature in Kelvin.

- **backgroundReflectance** (float)
  Background reflectance (unitless).

- **backgroundTemperature** (float)
  Background temperature in Kelvin.

- **haWindspeed** (float)
  High altitude windspeed in meters per second (m/s). Used to calculate the turbulence profile.

- **cn2at1m** (float)
  Refractive index structure parameter "near the ground" (e.g., at h = 1 m). Used to calculate the turbulence profile.
