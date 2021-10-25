from pathlib import Path
from datetime import datetime


today = datetime.today()
sales_file = (
    Path.cwd()
    / "data"
    / "raw"
    / "Jun22_2020"
    / "Potato_Fertilizer_Othello_Jun22_M10_transparent_reflectance_blue-444.tif"
)
# summary_file = Path.cwd() / "data" / "processed" / f"summary_{today:%b-%d-%Y}.pkl"
print(sales_file)


# load packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np# prepare some data

######
np.random.seed(42)
data1 = np.random.randn(100)
data2 = np.random.randn(100)
data3 = np.random.randn(100)
fig,ax = plt.subplots()
bp = ax.boxplot(x=[data1,data2,data3],  # sequence of arrays
positions=[1,5,7],   # where to put these arrays
patch_artist=True).  # allow filling the box with colors


fig,ax = plt.subplots()
ax.scatter(x=[1,2,3],y=[1,2,3],s=[100,200,300],c=['r','g','b'])

# font type
plt.rc('font', family='serif')
fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(1, 1, 1)
plt.plot([1, 2, 3, 4])
ax.set_xlabel('The x values')
ax.set_ylabel('The y values')

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')


# font sizes
fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(1, 1, 1)
plt.plot([1, 2, 3, 4])
ax.set_xlabel('The x values')
ax.set_ylabel('The y values')

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(1, 1, 1)


#colored lines
x = np.linspace(1., 8., 30)
ax.plot(x, x ** 1.5, color='k', ls='solid')
ax.plot(x, 20/x, color='0.50', ls='dashed')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Temperature (K)')

# subplots
fig = plt.figure()  # create a figure object
ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure

# inset plots
fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax2 = fig.add_axes([0.72, 0.72, 0.16, 0.16])

# two y axes
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax2 = ax1.twinx()
t = np.linspace(0., 10., 100)
ax1.plot(t, t ** 2, 'b-')
ax2.plot(t, 1000 / (t + 1), 'r-')
ax1.set_ylabel('Density (cgs)', color='red')
ax2.set_ylabel('Temperature (K)', color='blue')
ax1.set_xlabel('Time (s)')

# adding legends
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x = np.linspace(1., 8., 30)
ax.plot(x, x ** 1.5, 'ro', label='density')
ax.plot(x, 20/x, 'bx', label='temperature')
ax.legend()
#plt.rc('legend', fontsize='small')

# adding a colorbar
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
image = np.random.poisson(10., (100, 80))
i = ax.imshow(image, interpolation='nearest')
fig.colorbar(i)  # note that colorbar is a method of the figure, not the axes

# can control the colorbar position as so:
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.6,0.8])
image = np.random.poisson(10., (100, 80))
i = ax.imshow(image, interpolation='nearest')
colorbar_ax = fig.add_axes([0.7, 0.1, 0.05, 0.8])
fig.colorbar(i, cax=colorbar_ax)

# colorbar with a scatterplot
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])
x = np.random.random(400)
y = np.random.random(400)
c = np.random.poisson(10., 400)
s = ax.scatter(x, y, c=c, edgecolor='none')
ax.set_xlim(0., 1.)
ax.set_ylim(0., 1.)
colorbar_ax = fig.add_axes([0.7, 0.1, 0.05, 0.8])
fig.colorbar(s, cax=colorbar_ax)

# custom ticks and labels
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xticks([0.1, 0.5, 0.7])
ax.set_yticks([0.2, 0.4, 0.8])

# can hide ticks and labels
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xticks([])

# when saving an image, make the bounding box tight around the edges of the figures
fig.savefig('myplot.eps', bbox_inches='tight')
fig.savefig('Stylized Plots.png', dpi=300, bbox_inches='tight', transparent=True)


# helper functions
def custom_lineplot(ax, x, y, error, xlims, ylims, color='red'):
    """Customized line plot with error bars."""
    
    ax.errorbar(x, y, yerr=error, color=color, ls='--', marker='o', capsize=5, capthick=1, ecolor='black')
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    return ax
    
def custom_scatterplot(ax, x, y, error, xlims, ylims, color='green', markerscale=100):
    """Customized scatter plot where marker size is proportional to error measure."""
    
    markersize = error * markerscale
    
    ax.scatter(x, y, color=color, marker='o', s=markersize, alpha=0.5)
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    return ax
    
def custom_barchart(ax, x, y, error, xlims, ylims, error_kw, color='lightblue', width=0.75):
    """Customized bar chart with positive error bars only."""
    
    error = [np.zeros(len(error)), error]
    
    ax.bar(x, y, color=color, width=width, yerr=error, error_kw=error_kw, align='center')
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    return ax
    
def custom_boxplot(ax, x, y, error, xlims, ylims, mediancolor='magenta'):
    """Customized boxplot with solid black lines for box, whiskers, caps, and outliers."""
    
    medianprops = {'color': mediancolor, 'linewidth': 2}
    boxprops = {'color': 'black', 'linestyle': '-'}
    whiskerprops = {'color': 'black', 'linestyle': '-'}
    capprops = {'color': 'black', 'linestyle': '-'}
    flierprops = {'color': 'black', 'marker': 'x'}
    
    ax.boxplot(y,
               positions=x,
               medianprops=medianprops,
               boxprops=boxprops,
               whiskerprops=whiskerprops,
               capprops=capprops,
               flierprops=flierprops)
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    return ax

def stylize_axes(ax, title, xlabel, ylabel, xticks, yticks, xticklabels, yticklabels):
    """Customize axes spines, title, labels, ticks, and ticklabels."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.set_tick_params(top='off', direction='out', width=1)
    ax.yaxis.set_tick_params(right='off', direction='out', width=1)
    
    ax.set_title(title)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)

# example of a 2x2 grid of plots

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,6))

y = data.mean()
y_all = data.values
x = np.arange(len(means))
error = data.std()

xlims = (-1, 5)
ylims = (-5, 15)
bar_ylims = (0, 15)

custom_lineplot(ax[0][0], x, y, error, xlims, ylims)
custom_scatterplot(ax[0][1], x, y, error, xlims, ylims)
custom_barchart(ax[1][0], x, y, error, xlims, bar_ylims, error_kw)
custom_boxplot(ax[1][1], x, y_all, error, xlims, ylims)

titles = ['Line Plot', 'Scatter Plot', 'Bar Chart', 'Box Plot']
xlabel = 'Group'
ylabel = 'Value ($units^2$)'
xticks = x
xticklabels = range(1,6)

for i, axes in enumerate(ax.flat):
    # Customize y ticks on a per-axes basis
    yticks = np.linspace(axes.get_ylim()[0], axes.get_ylim()[1], 5)
    yticklabels = yticks
    stylize_axes(axes, titles[i], xlabel, ylabel, xticks, yticks, xticklabels, yticklabels)
    
fig.tight_layout()

