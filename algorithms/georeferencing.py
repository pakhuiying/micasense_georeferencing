import cameratransform as ct
import numpy as np
import os
import rasterio
from PIL import Image
import algorithms.flight_attributes as FlightAttributes

def directGeoreferencing_rasterio(imagePath,image_name,focal,lat,lon,alt,flight_angle,original_image_size=(1280,960),cropped_dimensions=(4, 6, 1240, 920)):
    """ 
    :param imagePath (str): directory where raw images are stored
    :param image_name (str): e.g. IMG_0079_1.tif
    :param focal (float): local length of camera
    :param original_image_size (tuple): original image_size = ImageWidth, ImageHeight
    :param lat (float): center coordinates of image
    :param lon (float): center coordinates of image
    :param alt (float): flight altitude
    :param flight_angle (float): in degrees, UAV's yaw converted to degrees
    :param cropped_dimensions (tuple of int): left, top, w, h (this is the cropped dimensions of dual camera system, it may be different for RedEdge cropped dimensions)
    this function georeferences one image
    modified from georeferencing code - UAV-Water-mosaicking-code
    im_cropped = img[top:top+h, left:left+w]
    """
    (left, top, w, h) = cropped_dimensions
    croppedImageWidth, croppedImageHeight = w-1, h-1
    rightExtent = croppedImageWidth + left
    bottomExtent = croppedImageHeight + top
    # heading param is wrt to east, so we add 90 degrees
    cam = ct.Camera(ct.RectilinearProjection(focallength_mm=focal,
                                            image=original_image_size),
                    ct.SpatialOrientation(elevation_m=alt,
                                        tilt_deg=0,
                                        roll_deg=0,
                                    heading_deg=flight_angle+90)) # This parameter should be set as 0º (when the UAV is heading East), 90º (when the UAV is heading North), 180º (when the UAV is heading West), or 270º (when the UAV is heading South). 

    # Latitude and Longitude of the GPS
    cam.setGPSpos(lat, lon, alt)

    # Image corners coordinates
    # UL (upper left), UR (upper right), LR (lower right), LL (lower left)
    coords = np.array([cam.gpsFromImage([left , top]), \
        cam.gpsFromImage([rightExtent-1 , top]), \
        cam.gpsFromImage([rightExtent-1, bottomExtent-1]), \
        cam.gpsFromImage([left , bottomExtent-1])])

    gcp1 = rasterio.control.GroundControlPoint(row=0, col=croppedImageHeight-1, x=coords[0,1], y=coords[0,0], z=coords[0,2], id=None, info=None)
    gcp2 = rasterio.control.GroundControlPoint(row=croppedImageWidth-1, col=croppedImageHeight-1, x=coords[1,1], y=coords[1,0], z=coords[1,2], id=None, info=None)
    gcp3 = rasterio.control.GroundControlPoint(row=croppedImageWidth-1, col=0, x=coords[2,1], y=coords[2,0], z=coords[2,2], id=None, info=None)
    gcp4 = rasterio.control.GroundControlPoint(row=0, col=0, x=coords[3,1], y=coords[3,0], z=coords[3,2], id=None, info=None)

    # Opening the original Image and generating a profile based on flight_stacks file generated before
    with rasterio.open(os.path.join(imagePath,'stacks',image_name), 'r') as src:
        profile = src.profile

        # Transformation
        tsfm = rasterio.transform.from_gcps([gcp1,gcp2,gcp3,gcp4])
        crs = rasterio.crs.CRS({"init": "epsg:4326"})
        profile.update(dtype=rasterio.uint16, transform = tsfm, crs=crs)

        georeferenced_stack_dir = os.path.join(imagePath, 'georeferenced_stacks')
        if not os.path.exists(georeferenced_stack_dir):
            os.mkdir(georeferenced_stack_dir)
            
        with rasterio.open(os.path.join(georeferenced_stack_dir, image_name), 'w', **profile) as dst:
            print(src.read().shape)
            dst.write(src.read().astype(rasterio.uint16)) # We write the coordinates in the image with this line.

def directGeoreferencing_gdal(imagePath,image_name,lat,lon,alt,flight_angle,dirname=""):
    """ 
    :param imagePath (str): directory where raw images are stored
    :param image_name (str): e.g. IMG_0079_1.tif
    :param lat (float): center coordinates of image
    :param lon (float): center coordinates of image
    :param alt (float): flight altitude
    :param flight_angle (float): in degrees, UAV's yaw converted to degrees
    :param dirname (str): directory name
    """
    georeferenced_thumbnails_directory = os.path.join(imagePath,f'georeferenced_thumbnails_{dirname}')
    if not os.path.exists(georeferenced_thumbnails_directory):
        os.mkdir(georeferenced_thumbnails_directory)

    # open image
    image_name = os.path.splitext(image_name)[0]
    try:
        im = np.asarray(Image.open(os.path.join(imagePath,'thumbnails',f'{image_name}.jpg')))
    except:
        im = None
        print(f"Image not found: {image_name}")

    image_fn = os.path.join(georeferenced_thumbnails_directory,image_name)
    GI = FlightAttributes.GeotransformImage(im,lat,lon,
                                altitude = alt,
                                angle = flight_angle)
    GI.georegister(image_fn)