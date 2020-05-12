# 3D-object-reconstruction-from-a-single-image

1. Download the dataset from the ???.

2. Install the requirements.

   1. Install the chamfer distance module.

      ```bash
      python ./chamfer/setup.py install
      ```

   2. Install the requirements.

      ```bash
      pip install -r .requirements.txt 
      ```

3. train

   ```bash
   python read_obj.py
   python train_pic2point.py
   ```

   