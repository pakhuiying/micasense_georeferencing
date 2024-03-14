import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import os
import PIL.Image as Image
import math

import algorithms.flight_attributes as FlightAttributes
from algorithms.alignment_error import correlation_coefficient

def fancy_coord(coord, pos, neg):
    """
    Stringifies the point in a more fancy way than __repr__, e.g.
    "44°35'27.6"N 100°21'53.1"W", i.e. with minutes and seconds.
    lat = fancy_coord(lat, "N", "S")
    lon = fancy_coord(lon, "E", "W")
    """
    coord_dir = pos if coord > 0 else neg
    coord_tmp = abs(coord)
    coord_deg = math.floor(coord_tmp)
    coord_tmp = (coord_tmp - math.floor(coord_tmp)) * 60
    coord_min = math.floor(coord_tmp)
    coord_sec = round((coord_tmp - math.floor(coord_tmp)) * 600) / 10
    coord = f"{coord_deg}°{coord_min}'{coord_sec}\"{coord_dir}"
    return coord

class PlotMap:
    def __init__(self,ne,sw,img):
        """
        :param (img): basemap obtained from google map tiles
            +---+ ne
            |   |
        sw  +---+
        """
        self.ne = ne
        self.sw = sw
        self.img = img
        self.image_height = img.shape[0]
        self.image_width = img.shape[1]
        self.lat_res_per_pixel = (self.ne[0] - self.sw[0])/self.image_height
        self.lon_res_per_pixel = (self.ne[1] - self.sw[1])/self.image_width
        # TILE_SIZE = 256  # in pixels
        # EARTH_CIRCUMFERENCE = 40075.016686 * 1000  # in meters, at the equator
        # zoom = 20
        # meters_per_pixel_at_zoom_0 = ((EARTH_CIRCUMFERENCE / TILE_SIZE) * math.cos(math.radians(self.sw[0])))
        # # pixel resolution from google tile at zoom=20
        # meters_per_pixel = meters_per_pixel_at_zoom_0 / (2 ** zoom)
        self.metres_per_degree = 111177.1472578764 
        # pixel resolution for the image
        self.meters_per_pixel = (self.ne[1] - self.sw[1])*self.metres_per_degree/self.image_width

    def get_ticks(self,breaks = 6):
        y_breaks = x_breaks = breaks
        # position of ticks
        yticks = np.arange(0,self.image_height, int(self.image_height/y_breaks))
        xticks = np.arange(0,self.image_width, int(self.image_width/x_breaks))
        # ytick labels
        lat_ticks = [self.ne[0] - y*self.lat_res_per_pixel for y in yticks]
        lat_labels = [fancy_coord(l, "N", "S") for l in lat_ticks]
        # xtick labels
        lon_ticks = [self.sw[1] + x*self.lon_res_per_pixel for x in xticks]
        lon_labels = [fancy_coord(l, "E", "W") for l in lon_ticks]
        return ((yticks,lat_labels),(xticks,lon_labels))
    
    def buffer(self,linewidth):
        return [pe.withStroke(linewidth=linewidth, foreground="white")]
    
    def get_compass_location(self):
        # location of north compass
        yArrow = int(0.05*self.image_height)
        yN = yArrow + yArrow
        xArrow = int(0.05*self.image_width)
        xN = xArrow - int(0.01*self.image_width)
        return ((yN,xN),(yArrow,xArrow))

    def get_scale_bar_location(self,scale_unit_distance=10):
        pad = int(0.05*self.image_height)
        # length of scale bar, pixels to represent 10 metres
        pixelsScale = int(scale_unit_distance/self.meters_per_pixel)
        # location of scale bar
        yScale = int(0.85*self.image_height)
        xScale = int(0.85*self.image_width)
        xScaleEnd = xScale + pixelsScale
        # location of scale number
        yScaleNumber = yScale - pad//2
        xScaleNumber = xScale
        return ((xScale,xScaleEnd,yScale),(xScaleNumber,yScaleNumber))
    
    def add_ticks(self, ax, add_yticks=True, add_xticks=True,tick_breaks=6,fontsize=12):
        ((yticks,lat_labels),(xticks,lon_labels)) = self.get_ticks(tick_breaks)
        # set y ticks
        if add_yticks is True:
            ax.set_yticks(yticks)
            ax.set_yticklabels(lat_labels)
            ax.tick_params(axis="y",labelsize=fontsize)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])
        # set x ticks
        if add_xticks is True:
            ax.set_xticks(xticks)
            ax.set_xticklabels(lon_labels)
            ax.tick_params(axis="x",labelsize=fontsize)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])
        return
    
    def add_compass(self,ax,fontsize_N=17,fontsize_triangle=11):
        ((yN,xN),(yArrow,xArrow)) = self.get_compass_location()
        # add N
        ax.text(xN,yN,'N', fontsize=fontsize_N,fontweight='bold',color='black',
                path_effects=self.buffer(4))
        # add triangle
        ax.plot(xArrow,yArrow,color='black',marker="^",markersize=fontsize_triangle,
                path_effects=self.buffer(4))
        return
    
    def add_scale_bar(self,ax,scale_unit_distance=10,fontsize=10,lw=4):
        ((xScale,xScaleEnd,yScale),(xScaleNumber,yScaleNumber)) = self.get_scale_bar_location(scale_unit_distance)
        # add scale
        ax.text(xScaleNumber,yScaleNumber,'10m', fontsize=fontsize,fontweight='bold',color='black',
                path_effects=self.buffer(3))
        ax.plot([xScale,xScaleEnd],[yScale,yScale],lw=lw,color='black',
                    path_effects=self.buffer(6))
        return
    
    def add_wql_points(self,ax, wql_dict, font_size=16, cbar_tick_font_size=10, cbar_label_font_size=12,
                       bbox_to_anchor=(0.5,-0.2),wql_label="Turbidity (NTU)",**kwargs):
        """
        overlay points over canvas given coordinates and corresponding wql concentration
        :param wql_dict (dict): where keys are: lat, lon, measurements; values are list of float
        """
        tss_lat = wql_dict['lat']
        tss_lon = wql_dict['lon']
        tss_measurements = wql_dict['measurements']
        rows_idx = []
        cols_idx = []
        tss_idx = []
        for i in range(len(tss_lat)):
            lat = tss_lat[i]
            lon = tss_lon[i]
            if lat > self.ne[0] or lat < self.sw[0]:
                continue
            if lon > self.ne[1] or lon < self.sw[1]:
                continue
            row_idx = int((self.ne[0] - lat)/self.lat_res_per_pixel)
            col_idx = int((lon - self.sw[1])/self.lon_res_per_pixel)
            rows_idx.append(row_idx)
            cols_idx.append(col_idx)
            tss_idx.append(tss_measurements[i])
        
        im = ax.scatter(cols_idx,rows_idx,c=tss_idx,alpha=0.5,label='in-situ sampling',**kwargs)
        ax.legend(loc='lower center',bbox_to_anchor=bbox_to_anchor,prop={'size': font_size})
        axcb = plt.colorbar(im,ax=ax)
        axcb.ax.tick_params(labelsize=cbar_tick_font_size)
        axcb.set_label(wql_label,size=cbar_label_font_size)
        return
    
    def plot(self,ax=None,add_ticks=True, add_compass=True, add_scale_bar=True):

        # plot
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(12,10))
        # plot image
        ax.imshow(self.img)
        if add_ticks is True:
            self.add_ticks(ax)
        if add_compass is True:
            self.add_compass(ax)
        if add_scale_bar is True:
            self.add_scale_bar(ax)
        
        if ax is None:
            plt.show()
        return
        

