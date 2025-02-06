#!/usr/bin/env python
# coding: utf-8

# # Complex Rotator Pipeline
# 
# To run the pipeline, run the whole notebook and then execute the `main` function with a list of TICs. If the parameter `after_56_only` of the `main` function is `True`, the pipeline will ignore any sectors before 56. Sectors before 56 do not always (but sometimes do) have 200 second cadence. 
# 
# **Pipeline Outputs**:
# - `build/main.pdf`: a PDF of each sector with 3 different graphs, organized into complex and non-complex stars
# - `output/complex.csv` and `output/not_complex.csv`: lists of complex and non-complex stars.
# - `output/[TIC]/`: a directory containing scatter plots, river plots, and periodograms for each TIC.

# In[ ]:

import os
import time
import shutil

import pandas as pd
import lightkurve as lk 
import matplotlib.pyplot as plt

from datetime import datetime
from multiprocessing import cpu_count, Pool
from pylatex.section import Chapter, Subsection
from pylatex import Document, Section, Figure, NoEscape, Command

from pipeline import *

plt.ioff()

import warnings
warnings.filterwarnings("ignore")

OUTPUT_PATH = f"{os.getcwd()}/output"


# In[29]:


def get_sectors(tic):
    """Returns a list of available sectors given an TIC."""
    sectors = []
    for result in lk.search_tesscut(str(tic)):
        sectors.append(int(result.mission[0].split(" ")[2]))
        
    return {"tic": tic, "sectors": sectors}

def get_targets(tics):
    """Constructs a dictionary of TICs and available sectors from a list of TICs"""
    targets = []

    for tic in tics:
        targets.append(get_sectors(tic))

    return targets


# In[30]:


def make_dataframes(l):
    """Convert the pipeline output into a CSV"""
    df = pd.DataFrame()
    
    tics = []
    sectors = []
    for tic in list(l.keys()):
        for sector in l[tic]:
            tics.append(tic)
            sectors.append(sector)
    
    df["TIC"] = tics
    df["Sectors"] = sectors

    return df


# In[32]:


def make_chapter(title, data, doc):
    """Used by `make_pdf_report` to generate each chapter of the PDF."""
    with doc.create(Chapter(title)):
        for tic in data.keys():
            doc.append(NoEscape(r'\newpage'))
            with doc.create(Section(f"{tic}")):
                for i, sector in enumerate(data[tic]):
                    if i > 0:
                        doc.append(NoEscape(r'\newpage'))
                    with doc.create(Subsection(f"Sector {sector}", label=f"{tic}_{sector}")):
                        lc = load_lc(f"{tic}", sector)

                        failed = []
                        for plot_type in ["plot", "river", "periodogram"]:
                            if not os.path.exists(f"{OUTPUT_PATH}/{tic}/{sector}_{plot_type}.png"):
                                failed.append(plot_type)
                                continue
                            with doc.create(Figure(position="H")) as plot:
                                doc.append(NoEscape(r'\begin{center}'))
                                plot.add_image(f"{OUTPUT_PATH}/{tic}/{sector}_{plot_type}.png", width=NoEscape(r'0.5\textwidth'))
                                doc.append(NoEscape(r'\end{center}'))

                        if failed != []:
                            doc.append("Failed to generate " + ", ".join(failed) + f" for {tic}.")

def make_pdf_report(complex, not_complex):
    """Generate a PDF with the pipeline results"""
    try:
        shutil.rmtree("./build")
    except:
        pass
    os.mkdir("./build")

    doc = Document(documentclass="report", lmodern=False, geometry_options={"margin": "0.5in", "top": "0.5in", "bottom": "0.5in"})

    doc.preamble.append(Command("title", "JMAG Complex Rotator Pipeline Results"))
    doc.preamble.append(Command("date", f"Generated on {datetime.now().strftime('%x %X')}"))
    doc.preamble.append(NoEscape(r'\usepackage{float}'))
    doc.preamble.append(NoEscape(r'\usepackage{hyperref}'))
    doc.preamble.append(NoEscape(r'\hypersetup{colorlinks=true}'))


    doc.append(NoEscape(r'\maketitle'))
    doc.append(NoEscape(r'\tableofcontents'))
    
    make_chapter("Complex Rotators", complex, doc)
    make_chapter("Non-Complex Rotators", not_complex, doc)
    doc.generate_tex("./build/main")
    
    os.popen("(cd build && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex) > /dev/null").read()


# In[33]:


def main(tics, after_56_only = False):
    start = time.time()
    print("Generating list of sectors")
    targets = get_targets(tics)
    
    complex = {}
    not_complex = {}

    print("Identifying complex rotators")
    for target in targets:
        for sector in target["sectors"]: 
            print(f"\rProcessing {target['tic']} sector {sector}", end="")
            if sector <= 56 and after_56_only:
                continue
            lc = load_lc(f"{target['tic']}", sector)
            
            # Determine complexity
            if is_complex(lc):
                if target["tic"] not in complex.keys():
                    complex[target["tic"]] = []
                complex[target["tic"]].append(sector)
            else:
                if target["tic"] not in not_complex.keys():
                    not_complex[target["tic"]] = []
                not_complex[target["tic"]].append(sector)
            if not os.path.isdir(f"{OUTPUT_PATH}/{target['tic']}"):
                os.mkdir(f"{OUTPUT_PATH}/{target['tic']}")
    
            try:
                # Plot river and save
                plt.figure(river_plot(lc).number) # this is necessary to make the output current and save the correct figure.
                plt.savefig(f"{OUTPUT_PATH}/{target['tic']}/{sector}_river.png")
                plt.close()
        
                # Plot lc and save
                plt.figure(graph_lc(lc).number) 
                plt.savefig(f"{OUTPUT_PATH}/{target['tic']}/{sector}_plot.png")
                plt.close()
        
                # Make periodogram and save
                plt.figure(lombscargle(lc).number) 
                plt.savefig(f"{OUTPUT_PATH}/{target['tic']}/{sector}_periodogram.png")
                plt.close()
            except:
                print(f"Failed to generate one or more plots for {tic}:{sector}")

    print("\nSaving results to csv")
    make_dataframes(complex).to_csv(f"{OUTPUT_PATH}/complex.csv")
    make_dataframes(not_complex).to_csv(f"{OUTPUT_PATH}/not_complex.csv")

    print("Saving results to pdf")
    make_pdf_report(complex, not_complex)

    print(f"Completed")
    print(f"Processed {len(tics)} stars and {sum([len(target['sectors']) for target in targets])} sectors in {int(time.time()-start)} seconds.")
    print(f"Identified {len(list(complex.keys()))} complex rotators")


# In[35]:


main(["TIC 404144841", "TIC 363963079"])

