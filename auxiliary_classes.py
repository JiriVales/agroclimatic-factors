import numpy as np
import datetime
import datetime
from osgeo import gdal, gdalnumeric, ogr, osr
from datetime import timedelta
import numpy as np
from PIL import ImageDraw

def convert_time(time_since_1900):
 d = datetime.datetime(1900, 1, 1)
 return (str(d+timedelta(hours=time_since_1900)))

def convert_time_reverse(date_time):
 d1 = datetime.datetime(1900, 1, 1)
 d2 = date_time
 d3 = d2-d1
 return ((d3.days*24)+(d3.seconds/3600))

def kelvin_to_celsius(temperature):
 return temperature-273.15

kelvin_to_celsius_vector = np.vectorize(kelvin_to_celsius)

class Grid:
 '''
 Data describes grid class.
 The most important attributes are WGS-84 coordinates of the grid origin.
 Then stepsize which is a tuple of x and y stepsize.
 And griddata which is data matrix which needs to be fit into the grid.
 '''
 def __init__ (self, grid_origin, grid_stepsize, grid_size=None, grid_data=None):
  '''
  Initialization of the grid object that takes in the described 3 parameters.
  '''
  self._grid_origin = grid_origin
  self._grid_stepsize = grid_stepsize
  self._grid_size=grid_size
  self._grid_data=grid_data
  
 def get_gridorigin(self):
  '''
  Returns x,y coordinates of the grid origin.
  '''
  return self._grid_origin
  
 def get_gridstepsize(self):
  '''
  Returns tuple of x,y grid step size.
  '''
  return self._grid_stepsize
 
 def get_gridsize(self):
  '''
  Returns the number of rows and columns that are supposed to be inside the grid.
  '''
  return self._grid_size
  
 def get_griddata(self):
  '''
  Returns the ndarray with the data that is fit into the grid.
  '''
  return self._grid_data
  
 def get_affinetransformation(self):
  '''
  Returns gdal affine transformation matrix.
  '''
  return (self._grid_origin[0],self._grid_stepsize[0],0,self._grid_origin[1],0,self._grid_stepsize[1])

 def iterate_sm_grids(self, step):
  start=self._grid_origin
  stop=start+np.multiply(self._grid_size, step)
  i=start[0]
  while (i!=stop[0]):
    j=start[1]
    while (j!=stop[1]):
     yield [i,j]
     j+=step[1]
    i+=step[0]

 
 def find_index(self,coordinate):
  '''
  Finds index of the point coordinate if it is within the grid.
  Otherwise returns error.
  '''
  xOffset = math.floor(round(coordinate[0]  - self._grid_origin[0], 2)/self._grid_stepsize[0])
  yOffset = math.ceil(round(coordinate[1] - self._grid_origin[1], 2)/self._grid_stepsize[1])
  return(xOffset,yOffset)



class Image:
 """Multi-array image. Typically netCDF data format."""
 def __init__ (self, data=None, metadata=None):
  """Initialize the image object.
  Typically should have attributes such as data and metadata.
  Data is a netCDF4 dataset object and metadata is a dictionary object."""
  self._data = data
  self._metadata = metadata

 def get_dimensions(self):
  return list(self._data.dimensions.keys())[::-1]

 def get_variables(self):
  return list(self._data.variables)

 def get_data(self):
  return self._data
  
 def set_data(self, data):
  self._data=data

 def find_index(self, dictionary):
  """Find slice indices given the dictionary with slice dimension name and its' range.
  For instance, dictionary {'latitude':[40,50]} would return index to make appropriate slice of dataframe"""
  variable=list(dictionary.keys())[0]
  if len(dictionary[variable])==2:
   return np.where(np.logical_and(self._data.variables[variable][:]>=np.sort(dictionary[variable])[0], self._data.variables[variable][:]<=np.sort(dictionary[variable])[1]))[0]
  elif len(dictionary[variable])==1:
   return np.array(np.where(self._data.variables[variable][:]==dictionary[variable][0])).flatten()
  else:
   raise ValueError('The dictionary should contain variable name with its value or range. ')

 def slice (self, attribute, dictionary):
  """Create subImage in a way of slicing original Image by dictionary of attributes"""
  dimensions=self._data.variables[attribute].dimensions
  indices=[]
  for dim in dimensions:
   if dim in list(dictionary.keys()):
    indices.append(self.find_index({dim:dictionary[dim]}))
   else:
    indices.append(slice(None))
  return self._data.variables[attribute][tuple(indices)].data

 def get_statistics(self, attribute, dictionary, kind):
  dimensions=self._data.variables[attribute].dimensions
  indices=[]
  for dim in dimensions:
   if dim in list(dictionary.keys()):
    indices.append(self.find_index({dim:dictionary[dim]}))
   else:
    indices.append(slice(None))
  if 'longitude' in list(dictionary.keys()):
   min_longitude=np.min(dictionary['longitude'])
  else:
   min_longitude=np.min(self._data.variables['longitude'])
  if 'latitude' in list(dictionary.keys()):
   max_latitude=np.max(dictionary['latitude'])
  else:
   max_latitude=np.max(self._data.variables['latitude'])
  if kind=='min':
   data=self._data.variables[attribute][indices].min(axis=0)
  elif kind=='max':
   data=self._data.variables[attribute][indices].max(axis=0)
  elif kind=='mean':
   data=self._data.variables[attribute][indices].mean(axis=0)
  elif kind=='sum':
   data=self._data.variables[attribute][indices].sum(axis=0)
  elif kind=='less_then_0_count':
   def less_then_zero(a):
    return (a<273.15).astype(int)
   pre_data=np.apply_along_axis(less_then_zero,0, self._data.variables[attribute][indices])
   data=pre_data.sum(axis=0)
  else:
   print('This kind of statistical measurement is not yet available. ')
  return subImage(data,{'affine_transformation':(min_longitude,abs(self._data.variables['longitude'][1]-self._data.variables['longitude'][0]),0,max_latitude,0,-abs(self._data.variables['longitude'][1]-self._data.variables['longitude'][0]))})
  
 def export_as(self, folder,  filename, format):
  if format=='h5':
   create_folder_if_not_exists(folder)
   h5file = tables.open_file(folder+filename+'.'+format, "w")
   h5file.create_array(h5file.root, 'data', self._data, title='data')
   h5file.close()
   return (folder+filename)
  else:
   return ('export to this file format not supported')

class subImage:
 '''
 Data is a double(x,y)-array image.
 Metadata is a dictionary object. One of the metadata keys should be 'affine_transformation'.
 It holds affine transformation parameters from ogr.gdal.GetGeoTransform() function.
 Typically represented by gdal array data type.
 Another recommended metadata key in the dictionary is 'nodata' key referring to which value should be neglected.
 '''
 def __init__ (self, dataarray=None, metadata=None):
  '''
  Initialize the Imagee object.
  It is needed to provide numpy array (values in 2D space) as well as metadata,
  where 'affine_transformation' and 'nodata' keys are important.
  '''
  self._data = dataarray
  self._metadata = metadata
  
 def get_metadata(self):
  '''
  Returns metadata dictionary.
  '''
  return self._metadata

 def set_metadata(self, dictionary):
  '''
  Sets subImage metadata by dictionary.
  '''
  self._metadata=dictionary
  
 def get_data(self):
  '''
  Returns 2D matrix of values.
  '''
  return self._data

 def set_data(self,data_matrix):
  '''
  Sets self data by provided 2D matrix of values.
  '''
  self._data=data_matrix
  
  
 def export_as_tif(self,filename):
  '''
  Export self data as GeoTiff 1-band image. 
  Output filename should be provided as a parameter.
  '''
  nrows,ncols=self._data.shape
  geotransform = self._metadata['affine_transformation']
  output_raster = gdal.GetDriverByName('GTiff').Create(filename, ncols, nrows, 1, gdal.GDT_Float32)
  output_raster.SetGeoTransform(geotransform)
  srs = osr.SpatialReference()
  srs.ImportFromEPSG(4326)
  #srs.ImportFromWkt('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]')
  output_raster.SetProjection(srs.ExportToWkt())
  output_raster.GetRasterBand(1).WriteArray(self._data)
  output_raster.GetRasterBand(1).SetNoDataValue(-32767)
  output_raster.FlushCache()
  del output_raster
  
 def clip_by_shape(self, geom_wkt, nodata=-32767):
  '''
  Clip an Imagee by wkt geometry.
  '''
  rast = self._data
  gt=self._metadata['affine_transformation']

  poly=ogr.CreateGeometryFromWkt(geom_wkt)

  # Convert the layer extent to image pixel coordinates
  minX, maxX, minY, maxY = poly.GetEnvelope()
  ulX, ulY = world_to_pixel(gt, minX, maxY)
  lrX, lrY = world_to_pixel(gt, maxX, minY)

  # Calculate the pixel size of the new image
  pxWidth = int(lrX - ulX)
  pxHeight = int(lrY - ulY)

  # If the clipping features extend out-of-bounds and ABOVE the raster...
  if gt[3] < maxY:
   # In such a case... ulY ends up being negative--can't have that!
   iY = ulY
   ulY = 0

  clip = rast[ulY:lrY, ulX:lrX]

  # Create a new geomatrix for the image
  gt2 = list(gt)
  gt2[0] = minX
  gt2[3] = maxY

  # Map points to pixels for drawing the boundary on a blank 8-bit,
  #   black and white, mask image.
 
  raster_poly = Image.new('L', (pxWidth, pxHeight), 1)
  rasterize = ImageDraw.Draw(raster_poly)
  
  def rec(poly_geom):
   '''
   Recursive drawing of parts of multipolygons over initialized PIL Image object using ImageDraw.Draw method.
   '''
   if poly_geom.GetGeometryCount()==0:
    points=[]
    pixels=[]
    for p in range(poly_geom.GetPointCount()):
     points.append((poly_geom.GetX(p), poly_geom.GetY(p)))
    for p in points:
     pixels.append(world_to_pixel(gt2, p[0], p[1]))
    rasterize.polygon(pixels, 0)
   if poly_geom.GetGeometryCount()>=1:
    for j in range(poly_geom.GetGeometryCount()):
     rec(poly_geom.GetGeometryRef(j))

  rec(poly)

  mask = image_to_array(raster_poly)

  # Clip the image using the mask
  try:
   clip = gdalnumeric.choose(mask, (clip, nodata))

  # If the clipping features extend out-of-bounds and BELOW the raster...
  except ValueError:
   # We have to cut the clipping features to the raster!
   rshp = list(mask.shape)
   if mask.shape[-2] != clip.shape[-2]:
    rshp[0] = clip.shape[-2]

   if mask.shape[-1] != clip.shape[-1]:
    rshp[1] = clip.shape[-1]

   mask.resize(*rshp, refcheck=False)

   clip = gdalnumeric.choose(mask, (clip, nodata))
  
  d={}
  d['affine_transformation'],d['ul_x'],d['ul_y'],d['nodata']=gt2,ulX,ulY,-32767  
  return (clip, d)
  
 def clip_by_shape_bb_buffer(self, envelope, buffer=0):
 
  '''
  Clip an Imagee by bounding box of wkt geometry. Add buffer in pixels optionally.
  '''
 
  rast = self._data
  gt=self._metadata['affine_transformation']

  # Convert the layer extent to image pixel coordinates
  minX = custom_floor(envelope[0],gt[1],precision_and_scale(gt[1])[1])
  maxX = custom_ceiling(envelope[1],gt[1],precision_and_scale(gt[1])[1])
  minY = custom_floor(envelope[2],gt[1],precision_and_scale(gt[1])[1])
  maxY = custom_ceiling(envelope[3],gt[1],precision_and_scale(gt[1])[1])
  
  minX-=(buffer*gt[1])
  maxX+=(buffer*gt[1])
  minY+=(buffer*gt[5])
  maxY-=(buffer*gt[5])
  
  ulX, ulY = world_to_pixel(gt, minX, maxY)
  lrX, lrY = world_to_pixel(gt, maxX, minY)
  

  # Calculate the pixel size of the new image
  pxWidth = int(lrX - ulX)
  pxHeight = int(lrY - ulY)

  clip = rast[ulY:lrY, ulX:lrX]

  # Create a new geomatrix for the image
  gt2 = list(gt)
  gt2[0] = minX
  gt2[3] = maxY
  
  d={}
  d['affine_transformation'],d['ul_x'],d['ul_y']=gt2,ulX,ulY
  
  return (clip, d)
  
 def calculate_slope(self):
  '''
  Calculate slope from self data of DEM image.
  '''
  x, y = np.gradient(self._data)
  slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
  return (slope,self._metadata)
  
 def calculate_azimuth(self):
  '''
  Calculate azimuth from self data of DEM image.
  '''
  x, y = np.gradient(self._data)
  aspect = (np.arctan2(-x, y))*180/np.pi
  return (aspect,self._metadata)
  
 def get_min_value(self):
  '''
  Get self min value excluding self nodata value.
  '''
  return np.min(self._data[np.where(self._data!=self._metadata['nodata'])])
  
 def get_max_value(self):
  '''
  Get self max value excluding self nodata value.
  '''
  return np.max(self._data[np.where(self._data!=self._metadata['nodata'])])
 
 def get_mean_value(self):
  '''
  Get self mean value excluding self nodata values.
  '''
  return np.mean(self._data[np.where(self._data!=self._metadata['nodata'])])
  
 def get_median_value(self):
  '''
  Get self median value excluding self nodata values.
  '''
  return np.median(self._data[np.where(self._data!=self._metadata['nodata'])])

