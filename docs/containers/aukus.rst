AUKUS Container
===============

To support users that require tools for an ML T&E workflow, we define a container that would accept an input dataset
and apply perturbations to the entire dataset. The perturbed images should be saved to disk and then the container will
shut down. In order to support this workflow, the AUKUS container was created.

Given a COCO dataset and a configuration file, the AUKUS container is able to generate perturbed images for each image
in the dataset. Each perturbed image will be saved to a given output directory as a new COCO dataset. Once all
perturbed images are saved, the container will terminate.

How to Use
----------
To run the AUKUS container, use the following command:
``docker run -v /path/to/input:/root/input/:ro -v /path/to/output:/root/output/ nrtk-jatic``
This will mount the inputs to the correct locations and use the ``nrtk-perturber`` CLI script with the default args.

Default Behavior
^^^^^^^^^^^^^^^^
- COCO dataset loaded from the ``dataset`` directory. Must be in the directory mounted to ``/root/input/``. The
  annotations file is expected to be named ``annotations.json`` at the root of the ``dataset`` directory. Optionally,
  ``image_metadata.json``, also at the root of the ``dataset`` directory, containing a list of per-image dictionaries
  of metadata may be provided. If a perturber requires a specific piece of metadata, it should be specified via this
  file.
- Configuration file loaded from ``nrtk_config.json``. Must be in the directory mounted to ``/root/input/``. The
  configuration file specifies the ``PerturbImageFactory`` parameters for image perturbations and should have the
  following structure::

    {
      "type": "one-of-the-keys-below",
      "PerturbImageFactoryImpl1": {
        "param1": "val1",
        "param2": "val2"
      },
      "PerturbImageFactoryImpl2": {
        "p1": 4.5,
        "p2": null
      }
    }

  The "type" key is considered a special key that should always be present and it specifies one of the other keys
  within the same dictionary. Each other key in the dictionary should be the name of a ``Configurable`` inheriting
  class type. In this case, the "type" key specifies which ``PerturbImageFactory`` configuration to use. Each
  subdictionary represents the configuration for that specific ``PerturbImageFactory`` implementation.
- For each perturber in the perturber factory provided by the configuration file, a sub-directory with a new COCO
  dataset is created with the corresponding perturbed images and detections in ``/root/output``.

Customizing Behavior
^^^^^^^^^^^^^^^^^^^^
If the user wants to use different input/output paths, the container expects the following arguments:

   * ``dataset_dir`` : Input COCO dataset
   * ``output_dir``  : Directory to store the perturbed dataset
   * ``config_file`` : JSON configuration file path

Note: The values for ``dataset_dir`` and ``config_file`` should be written from the perspective of the container (i.e.
``/path/on/container/dataset_dir/`` instead of ``/path/on/local/machine/dataset_dir/``)

Limitations
-----------

Currently, the AUKUS container supports the loading of only COCO datasets. Any existing dataset must be converted to a
COCO dataset before using the AUKUS container. Please see
`KWCOCO documentation <https://kwcoco.readthedocs.io/en/main/>`_ for more information on COCO datasets.
