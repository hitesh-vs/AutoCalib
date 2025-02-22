### RBE 549 HW1 - AutoCalib ###

* By default, the directory containing the images used for calibration is set to `Calibration_Imgs/Calibration_Imgs`. To change the directory path, modify the following line in the `Wrapper.py` file:

  ```python
  imgs_path = "Calibration_Imgs/Calibration_Imgs"  # Path to the directory containing the calibration images

* The code calculates the world points from the reference image `src_checker.png`, which is an image of how the checkboard looks like in the world without the distortions. 