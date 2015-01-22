# clip_shp_to_shp.py
import os
import glob

def clip_shp_to_shp(directory, shpclippath, pref="", suf="_clip"):
    # List shp file in a directory (not recursive)
    listResults = glob.glob(os.path.join(directory, '*.shp'))
    # call ogr2ogr to clip with shpclip var
    import subprocess
    for source in listResults:
        subprocess.call(["ogr2ogr", "-f", "ESRI Shapefile", "-clipsrc",
shpclip, os.path.basename(source) + "_clip.shp", source])

# set clipping shp path
shpclip = "/Users/fonnescj/Dropbox/Adaptive Disease Surveillance/Sao Paulo/sp/35MEE250GC_SIR.shp"
# set dir where searching shp to clip
directory = os.getcwd()
clip_shp_to_shp(directory, shpclip)