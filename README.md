# Fire for MODIS
This is a **System** designed for fire detection in **MODIS Level1B** products.
## Step 1: Classification
+ Use Threshold in Channel 21 and Channel 22 to filter the negative sub images
+ Use SimpleViT to complete the Classification Tasks
+ ## Step 2: Segmentation
+ Use UNet to complete the Segmentation Tasks
+ ## Step 3: System
+ backend: Flask
+ frontend: Vue
