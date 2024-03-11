import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

def readSelectedGPS(imagePath):
    """
    select GPS points from select_GPS.py
    returns flight_points which corresponds to the image index
    """
    fp = os.path.join(os.path.dirname(imagePath),'flight_attributes','gps_index.txt')
    with open(fp, "r") as output:
        idx_list = output.readlines()
    intList = sorted([int(i.replace('\n','')) for i in idx_list])
    print(intList)
    assert len(intList)%2 == 0, "number of GPS points must be even as one flight light has start and end point"
    n = len(intList)//2
    listIndex = [np.arange(intList[i*2],intList[i*2+1]+1).tolist() for i in range(n)]
    return [i for l in listIndex for i in l]

class DetectLines:
    def __init__(self,df, n = 3, thresh=0.99, plot = True):
        """ 
        identifies the flight lines automatically, and obtain the start and stop lines 
        :param df (pd.DataFrame): dataframe of the saved flight attributes in the flight attributes folder
        :param n (int): calculate normal vector across n points
        :param thresh (float): dot product between 2 vectors
        :param plot (bool): to plot the flight points
        """
        self.df = df
        self.n = n
        self.thresh = thresh
        self.plot = plot

    def get_filtered_points(self):
        """ 
        identify points in lines that are parallel to each other
        returns a list of dict with observations containing index, lat, lon
        """
        N = len(self.df.index)
        lon = self.df.longitude.to_numpy()
        lat = self.df.latitude.to_numpy()
        # print(lon.shape)
        # print(lat.shape)
        idx_list = []
        # i = n
        # while (i < N-n):
        for i in range(self.n,N-self.n):
        
            a1 = np.array([lon[i-self.n],lat[i-self.n]])
            a2 = np.array([lon[i],lat[i]])
            vec1 = a2 - a1
            norm_vec1 = vec1/np.linalg.norm(vec1)

            b2 = np.array([lon[i+self.n],lat[i+self.n]])
            vec2 = b2 - a2
            norm_vec2 = vec2/np.linalg.norm(vec2)

            # theta = np.arccos(np.dot(norm_vec2,norm_vec1))/np.pi*180
            north_vec = np.array([0,1])
            theta = int(np.arccos(np.dot(norm_vec2,north_vec))/np.pi*180)
            dot_product = np.dot(norm_vec2,norm_vec1)
            # print(dot_product)
            if dot_product > self.thresh:
                idx_list.append({'index':i,'lat':lat[i],'lon':lon[i],'dot_product':dot_product, 'theta':theta})
        
        theta_unique, theta_count = np.unique([i['theta'] for i in idx_list], return_counts = True)
        idx_max_count = np.argmax(theta_count)
        theta_selected1 = theta_unique[idx_max_count]
        theta_selected2 = 180 - theta_selected1
        
        filtered_idx_list = []
        for i in idx_list:
            if (i['theta'] >= theta_selected1 -self.n and i['theta'] <= theta_selected1+self.n) or (i['theta'] >= theta_selected2 -self.n and i['theta'] <= theta_selected2+self.n):
                filtered_idx_list.append(i)
        return filtered_idx_list
    
    def get_points(self, flight_points = None):
        """ 
        identify points in lines that are parallel to each other
        returns a list of indices
        """
        if flight_points is None:
            filtered_idx_list = self.get_filtered_points()
        else:
            filtered_idx_list = flight_points
        
        if self.plot is True:
            lon = self.df.longitude.to_numpy()
            lat = self.df.latitude.to_numpy()
            plt.figure(figsize=(10,10))
            plt.plot(lon,lat)
            if flight_points is None:
                plt.plot([i['lon'] for i in filtered_idx_list],[i['lat'] for i in filtered_idx_list],'r.',label='detected points')
            else:
                plt.plot([lon[i] for i in filtered_idx_list],[lat[i] for i in filtered_idx_list],'r.',label='detected points')
            plt.legend()
            plt.xlabel('Longitude')
            plt.xlabel('Latitude')
            plt.show()
        if flight_points is None:
            return [i['index'] for i in filtered_idx_list]
        else:
            return filtered_idx_list
    
    def get_start_stop_indices(self):
        """ 
        obtain the start and stop indices for each image line
        """
        filtered_idx_list = self.get_filtered_points()
        lon = self.df.longitude.to_numpy()
        lat = self.df.latitude.to_numpy()

        lines_idx = np.abs(np.diff([i['theta'] for i in filtered_idx_list], prepend = filtered_idx_list[0]['theta']))
        assert len(lines_idx) == len(filtered_idx_list)
        # print(filtered_idx_list)
        n_idx = len(lines_idx)
        breaks_idx = np.argwhere(lines_idx >= self.n).flatten() # identify breaks in between lines
        # print(breaks_idx)
        # print(lines_idx[breaks_idx])
        start_stop_idx = [(breaks_idx[i],breaks_idx[i+1]-1) for i in range(len(breaks_idx)-1)]
        if breaks_idx[0] > 0:
            start_stop_idx = [(0,breaks_idx[0]-1)] + start_stop_idx
        if breaks_idx[-1] < n_idx:
            start_stop_idx = start_stop_idx + [(breaks_idx[-1],n_idx-1)]
        # print(start_stop_idx)
        start_stop_idx = [(filtered_idx_list[start]['index'],filtered_idx_list[stop]['index']) for start,stop in start_stop_idx]
        # print(start_stop_idx)
        # print([i['index'] for i in filtered_idx_list])
        # print([lat[i['index']] for i in filtered_idx_list])
        if self.plot is True:
            plt.figure(figsize=(10,10))
            plt.plot(lon,lat)
            plt.plot([i['lon'] for i in filtered_idx_list],[i['lat'] for i in filtered_idx_list],'r.',label='detected points')
            for start, stop in start_stop_idx:
                plt.plot(lon[start],lat[start],'k.')
                plt.plot(lon[stop],lat[stop],'k.')
            # plt.plot([lon[i['index']] for i in filtered_idx_list],[lat[i['index']] for i in filtered_idx_list],'k.')
            plt.legend()
            plt.show()
        return start_stop_idx
    
def draw_plot(image_file_path):
    '''
    Plot GPS coordinates using matplotlib GUI interface
    '''
    gps_df,parent_dir = import_gps(image_file_path)#
    fig, ax = plt.subplots()
    ax.set_title('click on points to select start and end points')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    #add 
    global coords
    coords = []
    global indices
    indices = []

    ax.plot(gps_df['longitude'],gps_df['latitude'])
    line, = ax.plot(gps_df['longitude'],gps_df['latitude'],'o', #will plot a scatter plot when 'o' is used
                    picker=True, pickradius=5)  # 5 points tolerance


    def onpick(event):
        thisline = event.artist

        # global xdata, ydata
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()

        # global ind
        ind = event.ind

        coords.append((xdata[ind[0]], ydata[ind[0]]))

        indices.append(ind[0])

        print("Coords: {}\nIndices: {}".format(coords,indices))
        ax.plot(xdata[ind[0]], ydata[ind[0]],'ro')
        fig.canvas.draw()

        
        gps_fp = os.path.join(parent_dir,"gps_index.txt")
        with open(gps_fp, "w") as output:
            for i in indices:
                output.write(str(i)+'\n') #write list of gps indices to .txt file
        return coords, indices


    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()
    return indices

def import_gps(image_file_path):
    parent_dir = os.path.join(os.path.dirname(image_file_path),'flight_attributes')
    fp = os.path.join(parent_dir,'flight_attributes.csv')
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        return df,parent_dir
    
if __name__ == "__main__":
    draw_plot(sys.argv[1])