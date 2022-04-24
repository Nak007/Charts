'''
Available methods are the followings:
[1] create_schedule
[2] gantt_plot

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 21-04-2022

'''
import pandas as pd, numpy as np, sys
from pandas import Timestamp
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import(FixedLocator, 
                              FixedFormatter, 
                              StrMethodFormatter,
                              FuncFormatter)
from matplotlib.patches import Patch
from datetime import date

# Adding fonts
from matplotlib import font_manager
paths = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
for font_path in paths:
    if font_path.find("Hiragino Sans GB W3")>-1: 
        try:
            font_manager.fontManager.addfont(font_path)
            prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = prop.get_name()
            plt.rcParams.update({'font.family':'sans-serif'})
            plt.rcParams.update({'font.sans-serif':prop.get_name()})
            plt.rc('axes', unicode_minus=False)
            break
        except:pass

__all__ = ["create_schedule", "gantt_plot"]

def create_schedule(schedule, ref_date=None):
    
    '''
    Create schedule DataFrame.
    
    Parameters
    ----------
    schedule : pd.DataFrame object
        A Dataframe with the following columns:
        
        Column      Dtype         
        ------      -----         
        task        object        
        start       datetime64[ns]
        end         datetime64[ns]
        completion  float64    
    
    ref_date : str, default=None
        The reference date to be benchmarked against all tasks. If
        None, it defaults to today's date.
        
    Returns
    -------
    X : pd.DataFrame object
        A Dataframe with the following columns:
   
        Column      Dtype           Description
        ------      -----           -----------
        task        object          Task name
        start       datetime64[ns]  Starting date
        end         datetime64[ns]  Ending date
        completion  float64         Percent completion (actual)
        start_num   int64           Starting index
        end_num     int64           Ending index
        duration    int64           Duration in day(s)
        prog_day    float64         Actual progress in day(s)
        exp_day     int64           Expected progress in day(s)
        plan        float64         Percent completion (expected)
        diff_pct    float64         Difference in percent completion
        diff_day    float64         Difference in days
        status      object          Task status
    
    '''
    
    # ===============================================================
    X = pd.DataFrame(schedule).copy()
    # Default of ref_date is today().
    if ref_date is None: ref_date = Timestamp(date.today())
    ref_date = Timestamp(ref_date)
    # ---------------------------------------------------------------
    # Starting and ending dates
    start_date = X["start"].min()
    end_date = X["end"].max()
    # ---------------------------------------------------------------
    # Number of days from project start to start of tasks
    X['start_num'] = (X["start"] - start_date).dt.days
    # Number of days from project start to end of tasks
    X['end_num'] = (X["end"] - start_date).dt.days
    # Number of days between start and end of each task
    X['duration'] = X["end_num"] - X["start_num"]
    # ---------------------------------------------------------------
    # Working days between start and current progression of each task
    X['prog_day'] = (X["duration"] * X["completion"])
    exp_day = np.fmin(np.fmax((ref_date-X["start"]).dt.days,0),
                      X["duration"])
    X["exp_day"] = exp_day
    X["plan"] = exp_day/np.fmax(X["duration"],1)
    # ---------------------------------------------------------------
    # Difference between actual and plan
    # (-) : behind schedule
    # (+) : on-time or ahead of schedule
    X["diff_pct"] = X["completion"] - X['plan'] # <-- % diff
    X["diff_day"] = X["prog_day"] - X["exp_day"] # <-- day diff
    # ---------------------------------------------------------------
    # Task status
    status = np.where(X["plan"] > X["completion"], "delay", 
                      np.where(X["completion"]==1,
                               "complete","on-time"))
    X["status"] = np.where(X["duration"]==0,"event",status)
    # ===============================================================

    return X.copy()

def split_char(string, length, n_lines=2, suffix=" ..."):
    
    '''Private Function: split string of words into lines'''
    lines = []
    split2w = np.r_[string.lstrip().rstrip().split(" ")]
    while len(split2w)>0:
        index = np.r_[[len(word) for word in split2w]]
        index = np.cumsum(index)<=length
        lines += [" ".join(split2w[index])]
        split2w = split2w[~index]
        if len(lines)==n_lines: 
            if len(split2w)>0: lines[-1] += suffix
            break
    return lines

def gantt_plot(schedule, ax=None, ref_date=None ,colors=None, 
               intv_day=3, char_length=20, tight_layout=True, 
               holidays=None, show_delta=False):
    
    '''
    Gantt Chart
    
    Parameters
    ----------
    schedule : pd.DataFrame object
        An output from `create_schedule` function.
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, it uses default size, 
        figsize=(max(len(dates)*0.3 + 1, 6), max(n_tasks*0.45, 4.5)), 
        where `len(dates)` is a number of dates, and `n_tasks` is a
        number of tasks.
        
    ref_date : str, default=None
        The reference date to be benchmarked against all tasks. If
        None, it defaults to today's date.
        
    colors : array-like of shape (n_labels,), default=None
        List of color-hex must be arranged in correspond to following 
        labels i.e. ["complete", "on-time", "delay", "event", 
        "holidays"]. If None, it uses default colors from Matplotlib.
        
    intv_day : int, default=3
        Interval of dates to be displayed. 
        
    char_length : int, default=20
        A length of characters (task) to be displayed with in 2 lines.

    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots i.e. 
        plt.tight_layout().
        
    holidays : list of str, default=None
        A list of public or special holiday dates in `str` format i.e.
        ["2022-01-01"].
        
    show_delta : boolean, default=None
        If True, it displays differences between acutal and expected
        progress of tasks, whose status is either "delay" or "on-time".

    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    # Initialize default parameters
    # ===============================================================
    X = schedule.copy()
    n_tasks = len(X)
    start_date, end_date = X["start"].min(), X["end"].max()
    dates = pd.date_range(start_date, end=end_date)
    ax = plt.subplots(figsize=(max(len(dates)*0.3 + 1,6),
                               max(n_tasks*0.45, 4.5)))[1]
    # ---------------------------------------------------------------
    # Default of ref_date is today().
    if ref_date is None: ref_date = Timestamp(date.today())
    ref_date = Timestamp(ref_date)
    if colors is None:
        colors = ["#3498db","#2ecc71","#e74c3c","#2C3A47","#25CCF7"]
    intv_day = max(1, intv_day)
    # ---------------------------------------------------------------
    if holidays is not None:
        # Convert string date to datetime64.
        holidays = pd.Series([Timestamp(t) for t in holidays])
    # ===============================================================

    # Gantt chart
    # ===============================================================
    y, patches, labels, = np.arange(n_tasks), [], []
    status = ["complete", "on-time", "delay", "event"]
    for i,s in enumerate(status):
        index = X["status"]==s
        if (sum(index)>0) & (s!="event"):
            # Expected number of days.
            args = (y[index], X["exp_day"][index])
            kwds = dict(left=X["start_num"][index], height=0.7, 
                        color=colors[i], zorder=1, alpha=0.4)
            ax.barh(*args, **kwds)
            # Actual progress
            args = (y[index], X["prog_day"][index])
            kwds = dict(left=X["start_num"][index], height=0.7, 
                        color=colors[i], zorder=2)
            ax.barh(*args, **kwds)
        elif  (sum(index)>0) & (s=="event"):
            kwds = dict(c=colors[i], zorder=2, marker="D", s=50)
            ax.scatter(X["start_num"][index], y[index], **kwds)
            legend_kwds = dict(marker="D", markersize=5, 
                               markerfacecolor=colors[i], 
                               markeredgecolor="none",
                               color='none')
            sc = mpl.lines.Line2D([0],[0], **legend_kwds)
    # ---------------------------------------------------------------          
        labels += [s[0].upper() + s[1:] + 
                   " ({:,d})".format(sum(index))]
        if s=="event": patches += [sc]
        else: patches += [Patch(facecolor=colors[i])]
    # ---------------------------------------------------------------
    # Facecolor
    index = X["status"]!="event"
    ax.barh(y[index], X["duration"][index], left=X["start_num"][index], 
            alpha=0.5, facecolor="grey", height=0.7, edgecolor="none",
            zorder=1)
    # ---------------------------------------------------------------
    kwds = dict(facecolor="none", height=0.7, zorder=3, lw=1)
    delay = [(X["status"]!="delay") & (X["status"]!="event"),
             (X["status"]=="delay")]
    for index,ec in zip(delay, ["#636e72","#e74c3c"]):
        if sum(index)>0:
            args = (y[index], X["duration"][index])
            kwds.update({"left":X["start_num"][index],"edgecolor":ec})
            ax.barh(*args, **kwds)
    # ===============================================================

    # Task, and % progress
    # ===============================================================
    kwds = dict(textcoords='offset points', va="center", fontsize=12, 
                color="k", bbox=dict(boxstyle='round', facecolor="w", 
                                     pad=0.1, edgecolor="none"))
    r_text = {**kwds, **dict(ha="left" , xytext=(+3,0))}
    l_text = {**kwds, **dict(ha="right", xytext=(-3,0))}
    # ---------------------------------------------------------------            
    for n in range(n_tasks):
        if X["status"][n]!="event":
            s = "{:.0%}".format( X["completion"][n])
            if (X["status"][n]!="complete") & show_delta:
                diff = X["diff_pct"][n]
                if diff!=0:s += " ({:+,.0%})".format(diff)
            ax.annotate(s,(X["end_num"][n],n), **r_text)
            s = split_char(X["task"][n], char_length)
            ax.annotate("\n".join(s), (X["start_num"][n],n),**l_text)
        else: 
            s = split_char(X["task"][n], char_length)
            ax.annotate("\n".join(s), (X["start_num"][n],n),
                          **{**l_text,**dict(xytext=(-10,0))})
    # ===============================================================    

    # Set other attributes
    # ===============================================================
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylim(-0.5,max(y)+0.5)
    # ---------------------------------------------------------------
    # X-ticks
    xticks = np.arange(0, len(dates), intv_day)
    xticks_minor = np.arange(0, len(dates), 1)
    # ---------------------------------------------------------------
    ax.set_xticks(xticks)
    ax.set_xticks(xticks_minor, minor=True)
    ax.set_xticklabels(dates.strftime("%d/%m")[::intv_day], color='k')
    ax.set_xlim(-1, max(xticks_minor)+1)
    # ===============================================================

    # Vertival spans & reference date
    # ===============================================================
    # Weekends
    for n in xticks_minor[dates.dayofweek>=5]:
        ax.axvspan(n, n+1, zorder=-1, facecolor="grey", 
                   edgecolor="none", alpha=0.1)
    # ---------------------------------------------------------------
    # Holidays
    ref_x = xticks_minor[np.isin(dates, holidays)]
    if len(ref_x)>0:
        for n in ref_x:
            ax.axvspan(n, n+1, zorder=-1, alpha=0.2, 
                       facecolor=colors[4], edgecolor="none")
    labels += ["Weekends", 'Holidays']
    patches += [Patch(facecolor='grey', alpha=0.1),
                Patch(facecolor=colors[4], alpha=0.2)]
    # ---------------------------------------------------------------
    # Reference date.
    kwds = dict(textcoords='offset points', va="bottom", ha="center", 
                fontsize=13, xytext=(0,3), color="k", fontweight=600)
    ref_x = xticks_minor[np.isin(dates, ref_date)]
    if len(ref_x)>0:
        line = ax.axvline(ref_x, zorder=-1, lw=1, c="grey", ls="--")
        ax.annotate(Timestamp(ref_date).strftime("%d/%m")
                    ,(ref_x, n_tasks-0.5), **kwds)
        labels += ["Reference date"]
        patches += [line]
    # ==============================================================

    # Legends
    # ==============================================================
    legend = ax.legend(patches, labels, edgecolor="none", ncol=1,
                       borderaxespad=0.25, markerscale=1.5, 
                       columnspacing=0.3, labelspacing=0.7, 
                       handletextpad=0.5, prop=dict(size=12), 
                       loc='center left') 
    legend.set_bbox_to_anchor([1.01, 0.5], transform = ax.transAxes)
    if tight_layout: plt.tight_layout()
    # ===============================================================

    return ax