import pandas as pd
import matplotlib.pyplot as plt
import os

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
    draw_plot(r"D:\EPMC_flight\pandanRes\45angle_35H_70overlap\flight_attributes")