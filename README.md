# beach_trenches_and_crests_detection
Detecting trenches and crests using edge detection using two methods - Canny and zero crossing of gradient

# Capabilities:
- The code can handle both xyz files of bathymetry (depth in every point) and regular image formates (such as png, jpeg...).
- It creates a mask of edges pinpointing where are the trenches and crests (maximums and minimums of gradients) in the image. If input is XYZ file, it can then output the mask
  as an XYZ file or a shapefile to be uploaded into most GIS softwares i.e. ArcGIS Pro, QGIS and so on. If image then can't output either since no geo-information is available, only regular image formates.
- "coastline_trenches_edge_detection" - Uses the Canny edge detection methode.
- "zero_crossing_edge_detection" - Uses zero crossing with two possible kernels for convolution - LoG and DoG (Laplace of Gaussian and Difference of Gaussian).

# How to run
Using the main file:
```
python main.py -i <input_path> -m <method: canny, zero-dog, zero-log> -f <output_format: shp, xyz>
```


