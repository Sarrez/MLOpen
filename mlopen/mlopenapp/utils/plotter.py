import plotly.express as px
import plotly.offline as opy
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def pie_plot_from_lists(labels=None, values=None, title=None):
    """
        Update the plot data in plotly format.

        :param labels: The labels of the pie chart.
        :param values: The values of the pie chart.
        :param plot_title: The title of the plotly plot.
        :return: A string containing the graph in html form.
        """

    fig = px.pie(values=values, names=labels,
                 title=title)
    div = opy.plot(fig, auto_open=False, output_type='div', include_plotlyjs=False)
    return div

def plotlify_pie_js(xy=None, description=""):
    """
    Update the plot data in plotly format.

    :param xy: x and y in a single structure.
    :param description: The description of the plotly plot.
    :param plot_type: The type of the plotly plot.
    :return: A dictionary with the data in plotly format.
    """

    ret = {
        'data': [],
        'layout': {
            'paper_bgcolor': 'rgba(243, 243, 243, 1)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'title': {
                'text': description,
            }
        }
    }

    ret['data'].append(
        {
            'values': [v for k, v in xy.items()],
            'labels': [str(k) for k, v in xy.items()],
            'type': 'pie',
        }
    )
    return ret


def plotlify_scatter_js(xy=None, x=None, y=None, xtag=None, ytag=None, description=""):
    """
    Update the plot data in plotly format.

    :param xy: x and y in a single structure.
    :param description: The description of the plotly plot.
    :param plot_type: The type of the plotly plot.
    :return: A dictionary with the data in plotly format.
    """

    ret = {
        'data': [],
        'layout': {
            'paper_bgcolor': 'rgba(243, 243, 243, 1)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'title': {
                'text': description,
            }
        }
    }
    if xy:
        ret['data'].append(
            {
                'x': [v for k, v in xy.items()],
                'y': [k for k, v in xy.items()],
                'type': 'scatter',
            }
        )
    elif x and y:
        ret['data'].append(
            {
                'x': x,
                'y': y,
                'type': 'scatter',
            }
        )
    return ret

def bar(df,x_name,y_name):
    fig = px.bar(df, x=x_name, y=y_name)
    div = opy.plot(fig, auto_open=False, output_type='div', include_plotlyjs=False)

    return div
def detect_bbox(img,xmin,ymin,xmax,ymax):
    
    ''' Stacks bounding boxes of the a class on an initially empty image.
        :param img: 2D numpy array '''
    
    for i in range(img.shape[0]):
        height=i
        for j in range(img.shape[1]):
            width=j
            if(width<xmin or width>xmax):
                img[i][j]+=1
            elif(height<ymin or height>ymax):
                img[i][j]+=1
    return img

def collect_imgs(frames):

    '''  Creates heatmaps for all classes.
        :param frames: list of dataframes, one for each class detected.
        :return: dictionary with all heatmaps'''

    imglist=[]
    total_confidence=0
    heatmaps = {"Heatmap":[]}
    for i in range (0, len(frames)):   
        #initialize empty image as numpy array 
        final_img = np.zeros((150,150))
        imglist=[]
        frame = frames[i]
        for i, row in frame.iterrows():
            final_img = detect_bbox(final_img,row.xmin,row.ymin,row.xmax,row.ymax)
            total_confidence += row.confidence
            if row.imgname not in imglist:
                imglist.append(row.imgname) 
        final_img = final_img/len(frame)
        heatmaps["Heatmap"].append(final_img)
        
    return heatmaps

def custom_heatmap(frames, class_names):

    ''' Creates a plot with heatmaps as subplots.'''
    
    results = collect_imgs(frames)
    heatmaps = results['Heatmap']
    width = int(len(heatmaps)/4)
    height = 2*int(len(heatmaps)%4)
    fig = make_subplots(7, 4, subplot_titles=class_names)
    i=0
    while i < len(heatmaps):
        for j in range(1, 8):
            for k in range(1,5):
                fig.add_trace(go.Heatmap(z= heatmaps[i]), j, k)
                i = i + 1
                if(i==len(heatmaps)):
                    break
            if(i==len(heatmaps)):
                break
    
    fig.update_layout(width = 1500, height = 2800)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_traces(showscale=False)
    fig.update_yaxes(autorange="reversed")
    div = opy.plot(fig, auto_open=False, output_type='div', include_plotlyjs=False)
    return div
