'''
Available methods are the followings:
[1] create_schedule
[2] gantt_plot
[3] workingdays
[4] get_workload
[5] PrintSchedules

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 05-06-2023

'''
import pandas as pd, numpy as np, sys, os
from openpyxl import load_workbook
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import(FixedLocator, 
                              FixedFormatter, 
                              StrMethodFormatter,
                              FuncFormatter)
from matplotlib.patches import Patch
from datetime import date
import datetime

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

__all__ = ["create_schedule", 
           "gantt_plot", 
           "workingdays", 
           "get_workload", 
           "PrintSchedules"]

def create_schedule(schedule, ref_date=None, holidays=None):
    
    '''
    Create schedule DataFrame.
    
    Parameters
    ----------
    schedule : pd.DataFrame object
        A DataFrame with the following columns:
        
        Column      Dtype           Description         
        ------      -----           -----------    
        task        object          Task name
        start       datetime64[ns]  Starting date
        end         datetime64[ns]  Ending date
        completion  float64         Percent completion (actual)
    
    ref_date : str, default=None
        The reference date to be benchmarked against all tasks. If 
        None, it defaults to today's date.
        
    holidays : array of datetime64[D], default=None
        An array of dates to consider as invalid dates. They may be 
        specified in any order, and NaT (not-a-time) dates are ignored.
        
    Returns
    -------
    X : pd.DataFrame object
        A DataFrame with the following columns:
   
        Column      Dtype           Description
        ------      -----           -----------
        task        object          Task name
        start       datetime64[ns]  Starting date
        end         datetime64[ns]  Ending date
        completion  float64         Percent completion (actual)
        start_num   int64           Starting index
        end_num     int64           Ending index
        duration    int64           Duration in day(s)
        busday      int64           Number of business days
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
    if ref_date is None: ref_date = pd.Timestamp(date.today())
    ref_date = pd.Timestamp(ref_date)
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
    # Number of business days
    X['busday'] = [len(workingdays(X["start"][n], 
                                   X["end"][n], holidays)) 
                   for n in range(len(X))]
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
               major_locator=3, minor_locator=1, char_length=20, 
               tight_layout=True, holidays=None, show_delta=False, 
               show_vline=True, show_hline=True, show_days=True, 
               start_date=None, end_date=None):
    
    '''
    Gantt Chart
    
    Parameters
    ----------
    schedule : pd.DataFrame object
        An output from `create_schedule` function.
    
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, it uses default size.
        
    ref_date : str, default=None
        The reference date to be benchmarked against all tasks. If
        None, it defaults to today's date.
        
    colors : array-like of shape (n_labels,), default=None
        List of color-hex must be arranged in correspond to following 
        labels i.e. ["complete", "on-time", "delay", "event", 
        "holidays"]. If None, it uses default colors from Matplotlib.
        
    major_locator : int, default=3
        A number of date intervals (major ticks) to be displayed on 
        x-axis. 
        
    minor_locator : int, default=1
        A number of intervals with minor ticks on x-axis.
        
    char_length : int, default=20
        A length of characters (task) to be displayed with in 2 lines.

    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots i.e. 
        plt.tight_layout().
        
    holidays : list of str or array of datetime64[D], default=None
        A list of public or special holiday dates in `str` format i.e.
        ["2022-01-01"].
        
    show_delta : boolean, default=None
        If True, it displays differences between acutal and expected
        progress of tasks, whose status is either "delay" or "on-time".
        
    show_vline : bool, default=True
        If True, it displays minor gridlines on x-axis.
    
    show_hline : bool, default=True
        If True, it display minor gridlines on y-axis.
    
    show_days : bool, default=True
        If True, it displays a number of business days and duration
        underneath percent completion e.g. (8/10).
    
    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    # Initialize default parameters
    # ===============================================================
    X = schedule.reset_index(drop=True).copy()
    n_tasks = len(X)
    # ---------------------------------------------------------------
    # Convert to datetime64
    X[["start","end"]] = X[["start", "end"]].apply(pd.to_datetime)
    min_date, max_date = X["start"].min(), X["end"].max()
    # ---------------------------------------------------------------
    # Starting date
    if start_date is None: start_date = X["start"].min()
    start_date = max(pd.Timestamp(start_date), min_date)
    # --------------------------------------------------------------
    # Ending date
    if end_date is None: end_date = X["end"].max()
    end_date = min(pd.Timestamp(end_date), max_date)
    # --------------------------------------------------------------
    # Validate: end_date > start_date
    if end_date <= start_date:
        raise ValueError(f"`end_date` must be greater than" 
                         f"`start_date`. Got `end_date` "
                         f"({end_date.date()}) <= `start_date " 
                         f"({start_date.date()}).")
    # --------------------------------------------------------------
    # Number of periods and date range.
    dates = pd.date_range(min_date, max_date+np.timedelta64(1,'D'))
    # Determine x_min, and x_max
    sta_end = np.array([start_date, end_date],dtype='datetime64[D]')
    indices = np.isin(dates, sta_end)
    x_min, x_max = np.arange(len(dates))[indices]
    # ---------------------------------------------------------------
    # Default axis
    if ax is None:
        height = max(n_tasks*0.62, 4.5)
        width  = max((x_max - x_min + 2)*0.25, 12)
        ax = plt.subplots(figsize=(width, height))[1]
    # ---------------------------------------------------------------
    # Default of ref_date is today().
    if ref_date is None: ref_date = pd.Timestamp(date.today())
    ref_date = pd.Timestamp(ref_date)
    if colors is None:
        colors = ["#1B9CFC","#55E6C1","#FC427B","#82589F","#FEA47F"]
    major_locator = max(1, major_locator)
    minor_locator = max(1, minor_locator)
    # ---------------------------------------------------------------
    if holidays is not None:
        # Convert string date to datetime64.
        holidays = pd.Series([pd.Timestamp(t) for t in holidays])
    else: holidays = np.array(holidays, dtype="datetime64[D]")
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
    # ---------------------------------------------------------------          
        labels += [s[0].upper() + s[1:] + 
                   " ({:,d})".format(sum(index))]
        if s=="event": 
            legend_kwds = dict(marker="D", markersize=5, 
                               markerfacecolor=colors[-2], 
                               markeredgecolor="none",
                               color='none') 
            patches += [mpl.lines.Line2D([0],[0], **legend_kwds)]
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
    kwds = dict(textcoords='offset points', va="center", fontsize=10.5, 
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
            if show_days:
                a = (X["busday"][n], X["duration"][n])
                s = "\n".join((s,"({:,d}/{:,d})".format(*a))) 
            ax.annotate(s,(X["end_num"][n],n), **r_text)
            s = "\n".join(split_char(X["task"][n], char_length))
            i = min(max(X["start_num"][n], x_min), x_max)
            ax.annotate(s, (i,n),**l_text)
        else: 
            s = split_char(X["task"][n], char_length)
            i = min(max(X["start_num"][n], x_min), x_max)
            ax.annotate("\n".join(s), (i,n),
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
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylim(-0.5,max(y)+0.5)
    # ---------------------------------------------------------------
    # X-ticks
    xticks = np.arange(0, len(dates), major_locator)
    xticks_minor = np.arange(0, len(dates), minor_locator)
    # ---------------------------------------------------------------
    ax.set_xticks(xticks)
    ax.set_xticks(xticks_minor, minor=True)
    ax.set_xticklabels(dates.strftime("%d/%m")[::major_locator], 
                       color='k')
    ax.set_xlim(x_min, x_max)
    # ===============================================================

    # Vertival spans & reference date
    # ===============================================================
    # Weekends
    xticks = np.arange(0, len(dates), 1)
    for n in xticks[dates.dayofweek>=5]:
        ax.axvspan(n, n+1, zorder=-1, facecolor="grey", 
                   edgecolor="none", alpha=0.1)
    # ---------------------------------------------------------------
    # Holidays
    ref_x = xticks[np.isin(dates, holidays)]
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
    ref_x = xticks[np.isin(dates.astype(str),
                           ref_date.strftime("%Y-%m-%d"))]
    
    if len(ref_x)>0:
        line = ax.axvline(ref_x, zorder=-1, lw=1, c="grey", ls="--")
        ax.annotate(pd.Timestamp(ref_date).strftime("%d/%m")
                    ,(ref_x, n_tasks-0.5), **kwds)
        labels += ["Reference date"]
        patches += [line]
    # ---------------------------------------------------------------
    # Show vertical lines
    if show_vline:
        for n in np.arange(*ax.get_xlim(), minor_locator):
            ax.axvline(n, color="#bbbbbb", linestyle="--", lw=0.5, 
                       zorder=-1)
    ax.set_xlim(x_min-0.5, x_max+0.5)
    # ---------------------------------------------------------------
    # Show horizontal lines
    if show_hline:
        for n in np.arange(*ax.get_ylim())+1:
            ax.axhline(n, color="#bbbbbb", linestyle="--", lw=0.5, 
                       zorder=-1)
    # ===============================================================

    # Legends
    # ===============================================================
    legend = ax.legend(patches, labels, edgecolor="none", ncol=7,
                       borderaxespad=0.25, markerscale=1.5, 
                       columnspacing=0.3, labelspacing=0.5, 
                       handletextpad=0.5, prop=dict(size=12), 
                       loc='upper center', facecolor="w") 
    legend.set_bbox_to_anchor([0.5, -0.10], transform = ax.transAxes)
    if tight_layout: plt.tight_layout()
    # ===============================================================

    return ax

def workingdays(start, end, holidays=None):
    
    '''
    Calculates which of the given dates are valid business days, and 
    which are not.
    
    Parameters
    ----------
    start, end : datetime64[D] or str
        Starting and ending dates e.g. "1999-12-25" (25th of December 
        1999).
    
    holidays : array of datetime64[D], default=None
        An array of dates to consider as invalid dates. They may be 
        specified in any order, and NaT (not-a-time) dates are ignored. 
    
    Returns
    -------
    busday : array_like of datetime64[D]
        An array of valid business days.
    
    '''
    dates = np.arange(start, end, dtype="datetime64[D]")
    if holidays is None: holidays = np.r_[np.datetime64('nat')]
    holidays = np.array(holidays, dtype="datetime64[D]")
    busday = dates[np.is_busday(dates, holidays=holidays)]
    return busday

def get_workload(X, resources, holidays=None, hours=8):
    
    '''
    Parameters
    ----------
    X : pd.DataFrame object
        A DataFrame with the following columns:

        Column      Dtype           Description
        ------      -----           -----------    
        task        object          Task name
        start       datetime64[ns]  Starting date
        end         datetime64[ns]  Ending date
        n_hours     float64         Working hours/person
        resources   int64           Responsible persons
        
    resources : list of str
        List of numerical variables within `X` to be used as a 
        resource matrix.
        
    holidays : array of datetime64[D], default=None
        An array of dates to consider as invalid dates. They may be 
        specified in any order, and NaT (not-a-time) dates are ignored.
        
    hours : float or 1D-array of shape (n_task,), default=8
        If float, it is used as a fixed number of working hours per day,
        otherwise array of working hours (float) must be provided.
    
    Returns
    -------
    workloads : dict
        A dict of {"resource" : {"task": "number of working hours"}}

    works : dict
        A dict of {"resource": "number of works"}
    
    workhours : dict
        A dict of {"resource": "Total number of working hours"}
    
    busdays : array of datetime64[D]
        An array of business-day dates.
    
    '''
    # Valid business days
    busdays = workingdays(X["start"].min(), 
                          X["end"].max(), holidays)
    
    # Initialize parameters
    dtype = "datetime64[D]"
    start = np.array(X["start"], dtype=dtype)
    end   = np.array(X["end"]  , dtype=dtype)
    tasks = X["task"].values.copy()
    
    # Responsibility matrix
    resps = X[resources].values.copy() 
    rescs = np.r_(resources).ravel()
    
    # Validate `hours`
    if isinstance(hours, (int,float)):
        hours = hours / X[resources].values.sum(axis=1)
    elif not hasattr(hours, "__array__"):
        raise ValueError("`hours` must be either a number "
                         "or 1D-array")
    else: hours = np.array(hours).ravel()
        
    workloads = dict([(r,{}) for r in rescs])
    for n in range(len(tasks)):
        
        # Working hours given task
        task = workingdays(start[n], end[n], holidays)
        work = hours[n] * np.isin(busdays, task) 
        
        # Assign working hours to responsible person(s)
        workforces = rescs[resps[n,:]==1]
        if len(workforces) > 0:
            for w in workforces:
                workloads[w].update({tasks[n]:work})
    
    # Determine work amount & hour (`workloads`>0)
    works, workhours = dict(), dict()
    for key in workloads.keys():
        v = np.r_[list(workloads[key].values())]
        workhours.update({key:v.sum(0)})
        works.update({key:(v>0).sum(0)})
    
    return workloads, works, workhours, busdays

def read_workschedule(xls, completion=1., tasks=["Project"]):
    
    '''
    Read schedule from excel file.
    
    Parameters
    ----------
    xls : Excel file (*.xls, *.xlsm) object
        For any schedule sheet, it must contain the following columns:
        
        Column      Dtype           Description         
        ------      -----           -----------    
        task        object          Task name
        start       datetime64[ns]  Starting date
        end         datetime64[ns]  Ending date
        completion  float64         Percent completion
        type        object          Type of tasks ("schedule")
    
    completion : float, default=1.
        Select task whose percent completion is less than "completion".
        
    tasks : str, or list of str, default=["Project"]
        Select task whose task type matches "tasks".
        
    Returns
    -------
    data : dictionary
    
    tasks : dictionary
        {'PROJECT01': 'Description 1',
         'PROJECT02': 'Description 2'}
         
    members : dictionary
        {'PROJECT01': 'Member1, Member2',
         'PROJECT02': 'Member3', 
         'PROJECT03': 'Member2, Member4'}
    
    '''
    data = dict()
    schedule, holidays, team = "Schedule", "PublicHolidays", "Team"
    tasks = [tasks] if not isinstance(tasks, list) else tasks
    
    # Sheet name : 'schedule'
    df = pd.read_excel(xls, sheet_name=schedule) 
    df = df.drop(columns=["start_date", "days"]).fillna(0)
    cond = df["type"].isin(tasks) & (df["completion"]<completion)
    data[schedule] = df.loc[cond].reset_index(drop=True).to_dict()
    
    # Sheet name : 'PublicHolidays'
    df = pd.read_excel(xls, sheet_name=holidays)
    data[holidays] = df["Date"].values.astype("datetime64[D]")
    
    # Sheet name : 'Team'
    df = pd.read_excel(xls, sheet_name=team)
    data[team] = df["nickname"].values.astype("str")
    
    # Other sheets
    for sh_name in data[schedule]["ref_id"].values():
        try:
            df = pd.read_excel(xls, sheet_name=sh_name)
            data[sh_name] = df.to_dict()
        except: pass
    
    # Tasks' decription
    tasks = dict([(a,b) for a,b in 
                  zip(data[schedule]["ref_id"].values(),
                      data[schedule]["task"].values())])
    
    # Project owners (responsible personnel)
    df = pd.DataFrame(data[schedule])
    members = list(set(data[team]).intersection(df.columns))
    owners = np.full(df[members].shape, members)
    owners = np.where(df[members]==1, owners, "")
    members = dict([(a,", ".join(b[b!=""])) for a,b in 
                    zip(data[schedule]["ref_id"].values(),owners)])
     
    return data, tasks, members

def get_subfolder(path=None):
    
    '''
    Create folder (namely "Progress") and its subfolder, whose name 
    is the current date in "%Y-%m-%d" format e.g. 2023-03-01.
    
    Parameters
    ----------
    path : str, default=None
        A path-like object representing a file system path. If None,
        it uses a current working directory i.e. os.getcwd().
    
    Returns
    -------
    path : Folder path
    
    '''
    if path is None or os.path.isdir(path)==False:
        path = os.getcwd() + "\\Progress"
        if os.path.isdir(path)==False: os.mkdir(path)
    subfolder = datetime.datetime.now().strftime("%Y-%m-%d")
    path = f"{path}\\{subfolder}"
    if os.path.isdir(path)==False: os.mkdir(path)
    return path, subfolder

def PrintSchedules(xls, completion=1., tasks=["Project"], n_days=15, 
                   path=None, dpi=200, kwargs=None):
    
    '''
    Print all schedules in `xls`. All images will be saved under
    "\\Progress\\%Y-%m-%d" or "{path}\\%Y-%m-%d", where "%Y-%m-%d" 
    represents a current date format e.g. 2023-03-01.
    
    Parameters
    ----------
    xls : Excel file (*.xls, *.xlsm) object
        For any schedule sheet, it must contain the following columns:
        
        Column      Dtype           Description         
        ------      -----           -----------    
        task        object          Task name
        start       datetime64[ns]  Starting date
        end         datetime64[ns]  Ending date
        completion  float64         Percent completion
        type        object          Type of tasks ("schedule")
    
    completion : float, default=1.
        Select task whose percent completion is less than "completion".
        
    tasks : str, or list of str, default=["Project"]
        Select task whose task type matches "tasks".
        
    n_days : int, default=15
        Number of days
        
    path : str, default=None
        A path-like object representing a file system path. If None,
        it uses a current working directory i.e. os.getcwd(). 
    
    dpi : float, default=200
        The resolution in dots per inch.
        
    kwargs : dict, default=None
        All keyword parameters supported by `gantt_plot`. Some
        parameters will be changed and ignored.
    
    Returns
    -------
    path : str
        A path that stores all saved images.
        
    '''
    # Read data and create repository (folder)
    data, tasks, members = read_workschedule(xls)
    path, suffix = get_subfolder(path)
    suffix = suffix.replace("-","")
    kwargs = dict() if not isinstance(kwargs, dict) else kwargs

    df = pd.DataFrame(data["Schedule"])
    holidays = data["PublicHolidays"]
    if len(df)==0: raise ValueError("No project found.")
    df["task"] = df[["ref_id","task"]].apply(lambda x: ": ".join(x), axis=1)

    ref_date = kwargs.get("ref_date", None)
    ref_date = pd.Timestamp(date.today()) if ref_date is None else ref_date
    S0 = ref_date - np.timedelta64(n_days, "D")
    E0 = ref_date + np.timedelta64(n_days, "D")
    S1 = df["start"].min()
    E1 = df["end"].max()

    # Determine starting and ending dates 
    overlap = min((E0-S1).days, (E1-S0).days)<=0
    start_date = min(S0, S1) if overlap else S0
    end_date = max(E0, E1) if overlap else E0
    kwargs.update({"ref_date" : ref_date, 
                   "holidays" : holidays, 
                   "start_date" : start_date, 
                   "end_date" : end_date})
    
    # Create master schedule (input)
    plt.ioff()
    args = (df, ref_date, holidays)
    X = create_schedule(*args).sort_values(by="start",ascending=False)
    X.reset_index(drop=True, inplace=True)
    ax = gantt_plot(X, **kwargs)
    plt.gcf().suptitle("Master Schedule", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{path}\\masterschedule_{suffix}.png", dpi=dpi)
    
    # Other projects
    _ = kwargs.pop("start_date")
    _ = kwargs.pop("end_date")
    projects = set(data.keys()).intersection(tasks.keys())
    if len(projects) > 0:
        for project in projects:
            X = pd.DataFrame(data[project])   
            X = create_schedule(X, ref_date, holidays)
            X = X.sort_index(ascending=False).reset_index(drop=True)
            ax = gantt_plot(X, **kwargs)
            plt.gcf().suptitle(f"{project}: {tasks[project]}\n"
                               f"Owners : {members[project]}", 
                               fontsize=15)
            plt.tight_layout()
            plt.savefig(f"{path}\\{project}_{suffix}.png", dpi=dpi)
    plt.ion()
    
    return path