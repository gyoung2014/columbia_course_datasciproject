
# coding: utf-8

# In[1]:


import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd


# In[2]:


plotly.tools.set_credentials_file(username='yg2499',api_key='el7rt983crGV1cp4vHvr')
df1 = pd.read_csv('/Users/gyang/Desktop/ProjectDataScience/final/test code/Gross domestic product (GDP) by state (millions of current dollars).csv')


# In[13]:


state=list(df1['Area'])

first = list(df1['8'])
second = list(df1['9'])
third = list(df1['10'])
fourth = list(df1['11'])
fifth = list(df1['12'])
sixth = list(df1['13'])
seventh = list(df1['14'])
eighth = list(df1['15'])

trace1 = go.Bar(
    y=state,
    x=first,
    name='2008',
    orientation = 'h',
    marker = dict(
        color = 'rgba(236, 155, 206)',
        line = dict(
            color = 'rgba(255,255,255)',
            width = 0)
    )
)
trace2 = go.Bar(
    y=state,
    x=second,
    name='2009',
    orientation = 'h',
    marker = dict(
        color = 'rgba(243, 178, 243)',
        line = dict(
            color = 'rgba(255,255,255)',
            width = 0)
    )
)
trace3 = go.Bar(
    y=state,
    x=third,
    name='2010',
    orientation = 'h',
    marker = dict(
        color = 'rgba(217, 178, 243)',
        line = dict(
            color = 'rgba(255,255,255)',
            width = 0)
    )
)
trace4 = go.Bar(
    y=state,
    x=fourth,
    name='2011',
    orientation = 'h',
    marker = dict(
        color = 'rgba(198, 213, 247)',
        line = dict(
            color = 'rgba(255,255,255)',
            width = 0)
    )
)

trace5 = go.Bar(
    y=state,
    x=fifth,
    name='2012',
    orientation = 'h',
    marker = dict(
        color = 'rgba(205, 242, 243)',
        line = dict(
            color = 'rgba(255,255,255)',
            width = 0)
    )
)
trace6 = go.Bar(
    y=state,
    x=sixth,
    name='2013',
    orientation = 'h',
    marker = dict(
        color = 'rgba(179, 247, 204)',
        line = dict(
            color = 'rgba(255,255,255)',
            width = 0)
    )
)

trace7 = go.Bar(
    y=state,
    x=seventh,
    name='2014',
    orientation = 'h',
    marker = dict(
        color = 'rgba(236, 131, 131)',
        line = dict(
            color = 'rgba(255,255,255)',
            width = 0)
    )
)
trace8 = go.Bar(
    y=state,
    x=eighth,
    name='2015',
    orientation = 'h',
    marker = dict(
        color = 'rgba(251, 255, 102)',
        line = dict(
            color = 'rgba(255,255,255)',
            width = 0)
    )
)



data = [trace1, trace2,trace3,trace4,trace5,trace6,trace7,trace8]
layout = go.Layout(
    barmode='relative',
    title ='2007~2015 GDP Change Rate by state'
)


fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='2007~2015 GDP Change Rate by state')


# In[14]:


df2=(pd.read_csv('/Users/gyang/Desktop/ProjectDataScience/final/test code/PCE_ALL_try.csv'))


# In[15]:


state=list(df2['GeoName'])



first = list(df2['8'])
second = list(df2['9'])
third = list(df2['10'])
fourth = list(df2['11'])
fifth = list(df2['12'])
sixth = list(df2['13'])
seventh = list(df2['14'])
eighth = list(df2['15'])

trace1 = go.Bar(
    y=state,
    x=first,
    name='2008',
    orientation = 'h',
    marker = dict(
        color = 'rgba(236, 155, 206)',
        line = dict(
            color = 'rgba(255,255,255)',
            width = 0)
    )
)
trace2 = go.Bar(
    y=state,
    x=second,
    name='2009',
    orientation = 'h',
    marker = dict(
        color = 'rgba(243, 178, 243)',
        line = dict(
            color = 'rgba(255,255,255)',
            width = 0)
    )
)
trace3 = go.Bar(
    y=state,
    x=third,
    name='2010',
    orientation = 'h',
    marker = dict(
        color = 'rgba(217, 178, 243)',
        line = dict(
            color = 'rgba(255,255,255)',
            width = 0)
    )
)
trace4 = go.Bar(
    y=state,
    x=fourth,
    name='2011',
    orientation = 'h',
    marker = dict(
        color = 'rgba(198, 213, 247)',
        line = dict(
            color = 'rgba(255,255,255)',
            width = 0)
    )
)

trace5 = go.Bar(
    y=state,
    x=fifth,
    name='2012',
    orientation = 'h',
    marker = dict(
        color = 'rgba(205, 242, 243)',
        line = dict(
            color = 'rgba(255,255,255)',
            width = 0)
    )
)
trace6 = go.Bar(
    y=state,
    x=sixth,
    name='2013',
    orientation = 'h',
    marker = dict(
        color = 'rgba(179, 247, 204)',
        line = dict(
            color = 'rgba(255,255,255)',
            width = 0)
    )
)

trace7 = go.Bar(
    y=state,
    x=seventh,
    name='2014',
    orientation = 'h',
    marker = dict(
        color = 'rgba(236, 131, 131)',
        line = dict(
            color = 'rgba(255,255,255)',
            width = 0)
    )
)
trace8 = go.Bar(
    y=state,
    x=eighth,
    name='2015',
    orientation = 'h',
    marker = dict(
        color = 'rgba(251, 255, 102)',
        line = dict(
            color = 'rgba(255,255,255)',
            width = 0)
    )
)




data = [trace1, trace2,trace3,trace4,trace5,trace6,trace7,trace8]
layout = go.Layout(
    barmode='relative',
    title ='2007~2015 US PCE by State'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='2007~2015 US PCE by State')

