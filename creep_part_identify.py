import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import datetime as dt
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

class ZoomPan:
    "Scrolling for zoom on interactive plot from: https://stackoverflow.com/users/1629298/seadoodude"
    def __init__(self):
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.xpress = None
        self.ypress = None


    def zoom_factory(self, ax, base_scale = 2.):
        def zoom(event):
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xdata = event.xdata # get event x location
            ydata = event.ydata # get event y location

            if event.button == 'down':
                # deal with zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'up':
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1
                print(event.button)

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
            ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest
        fig.canvas.mpl_connect('scroll_event', zoom)

        return zoom

    def pan_factory(self, ax):
        def onPress(event):
            if event.inaxes != ax: return
            self.cur_xlim = ax.get_xlim()
            self.cur_ylim = ax.get_ylim()
            self.press = self.x0, self.y0, event.xdata, event.ydata
            self.x0, self.y0, self.xpress, self.ypress = self.press

        def onRelease(event):
            self.press = None
            ax.figure.canvas.draw()

        def onMotion(event):
            if self.press is None: return
            if event.inaxes != ax: return
            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress
            self.cur_xlim -= dx
            self.cur_ylim -= dy
            ax.set_xlim(self.cur_xlim)
            ax.set_ylim(self.cur_ylim)

            ax.figure.canvas.draw()

        fig = ax.get_figure() # get the figure of interest

        # attach the call back
        fig.canvas.mpl_connect('button_press_event',onPress)
        fig.canvas.mpl_connect('button_release_event',onRelease)
        fig.canvas.mpl_connect('motion_notify_event',onMotion)

        #return the function
        return onMotion

def creep_event_dataframe(dataframe,duration, start, creep_data,creepmeter):
    dataframes={}
    creep_index = np.array(dataframe.og_index)
    for j in range(len(dataframe)):
        Creep_event_time = ()
        Creep_event_slip = ()
        Creep_event_slip_rate = ()
        #boolarr = (START.iloc[j].replace(tzinfo=None)<= data.Time) & (data.Time <= END.iloc[j].replace(tzinfo=None))
        if creepmeter == 'XHR' and j == 4:
            boolarr = (start.iloc[j] - dt.timedelta(hours=2)  <= creep_data.Time) & (creep_data.Time <= start.iloc[j] + dt.timedelta(hours=duration[j]) - dt.timedelta(hours=4)) #assumption made here
        else: 
            boolarr = (start.iloc[j] - dt.timedelta(hours=2) <= creep_data.Time) & (creep_data.Time <= start.iloc[j] + dt.timedelta(hours=duration[j])- dt.timedelta(minutes=10)) #assumption made here
        Creep_event_time = creep_data.Time[boolarr]
        Creep_event_time = (Creep_event_time - Creep_event_time.iloc[0])/dt.timedelta(hours=1) #set start time to 0
        Creep_event_slip = creep_data.Creep[boolarr]
        #Creep_event_slip = creep_data.rolling_mean[boolarr] #extract slip
        Creep_event_slip = Creep_event_slip - Creep_event_slip.iloc[0] #set initial slip to 0
        Velocity = creep_data.vel[boolarr]
        Acceleration = creep_data.acc[boolarr]
        dataframes[creep_index[j]] = pd.DataFrame({'Time':Creep_event_time,'Slip':Creep_event_slip,'Velocity':Velocity,'Acceleration':Acceleration})
        dataframes[creep_index[j]].reset_index(inplace=True)
    return dataframes, creep_index    
    

def creep_separator(dataframe_data,dataframe_events,creepmeter):
    for i in range(len(dataframe_events)):
        print(i)
        plt.close('all')
        fig=plt.figure()
        ax = fig.add_subplot(111, xlim=(0,1), ylim=(0,1), autoscale_on=False)
        ax.set_title('Scroll to zoom')
        scale = 1.1
        zp = ZoomPan()
        figZoom = zp.zoom_factory(ax, base_scale = scale)
        figPan = zp.pan_factory(ax)
        s = 1


        ax.scatter(dataframe_data[i].Time,dataframe_data[i].Slip, s=1, c = 'black')
        plt.xlim([dataframe_data[i].Time.iloc[0]-1,dataframe_data[i].Time.iloc[-1]+1])
        plt.ylim(-1,1.2*max(dataframe_data[i].Slip))
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        pts = plt.ginput(5,timeout = 25)
        plt.show()
        
        if len(pts) ==4:
            pts.append((np.nan,np.nan))
        if len(pts) ==3:
            pts.append((np.nan,np.nan))
            pts.append((np.nan,np.nan))
        if len(pts) == 2:
            pts.append((np.nan,np.nan))
            pts.append((np.nan,np.nan))
            pts.append((np.nan,np.nan))
        if len(pts) == 1:
            pts.append((np.nan,np.nan))
            pts.append((np.nan,np.nan))
            pts.append((np.nan,np.nan))
            pts.append((np.nan,np.nan))
        if len(pts) == 0:
            pts.append((np.nan,np.nan))
            pts.append((np.nan,np.nan))
            pts.append((np.nan,np.nan))
            pts.append((np.nan,np.nan))
            pts.append((np.nan,np.nan))
            
        Coords_creep=[]
        Creep_identified=[]
        for sublist in pts:
            for item in sublist:
                Coords_creep.append(item)
        Creep_identified = [Coords_creep]
        #print(Creep_identified)
        if i < 1:
            TEST_DF = pd.DataFrame(Creep_identified,columns=['Ts','Ds','T01', 'D01', 'T02','D02','T03','D03','T04','D04'])
        else:
            df_2 = pd.DataFrame(Creep_identified,columns=['Ts','Ds','T01', 'D01', 'T02','D02','T03','D03','T04','D04'])
            TEST_DF.loc[i] = df_2.loc[0]
        #print(TEST_DF)
        TEST_DF.to_csv("../../Rheology/{k}/Creep_phases_{k}_C.csv".format(k=creepmeter)) 
    plt.close('all')    
    return TEST_DF
