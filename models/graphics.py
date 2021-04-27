import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn

def visualise_board(log, title):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12,12))

    fig.suptitle(title, fontsize=16)
    ax1.plot(log["epochs"],log["T Loss"], c="tab:blue")
    ax1.xaxis.grid(True,linestyle=":",color='black')
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Train. Loss")

    ax2.plot(log["epochs"],log["T Accuracy"], c="tab:red")
    ax2.xaxis.grid(True,linestyle=":",color='black')
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Train. Accuracy")

    ax3.plot(log["epochs"],log["V Loss"], c="tab:green")
    ax3.xaxis.grid(True,linestyle=":",color='black')
    ax3.set_xlabel("Iterations")
    ax3.set_ylabel("Val. Loss")

    ax4.plot(log["epochs"],log["V Accuracy"], c="tab:orange")
    ax4.xaxis.grid(True,linestyle=":",color='black')
    ax4.set_xlabel("Iterations")
    ax4.set_ylabel("Val. Accuracy")

def pairPlot(data):
    seaborn.pairplot(data,hue="income",plot_kws={"s": 3},dropna=True)

def corrMap(data):
    fig,ax = pyplot.subplots(figsize=(15,8))
    seaborn.heatmap(ax=ax,data=data.corr().round(2),annot=True,cmap=seaborn.diverging_palette(220,20),linewidth=2)

def progress_plot(ax,model,X,x):
    minX = min(X)
    maxX = max(X)
    ax.plot([minX,maxX], [model.yhat(minX),model.yhat(maxX)], c="tab:red", alpha=0.35, linewidth=0.1)
    ax.scatter(x,model.yhat(x), c="tab:red", alpha=0.35, linewidth=0.1)
