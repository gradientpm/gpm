# Python classical imports
import optparse
import os
import math
import shutil
import sys
import datetime

# Python for reading XML files
import xml.etree.ElementTree as ET

# Other images !!!!
import showResults
import generateFigures

# To read the images
import rgbe.io
import rgbe.utils
try:
    import Image
except ImportError:
    from PIL import Image

##########################################
##########################################
## HARD CODE HTML
##########################################
##########################################
PRODMOD = True
REMOVE_TMP = False
ABOVE_CURVES = "<p></p>"
DEFAULTREF = "%%REF%%"
htmlHead = """
<html xmlns="http://www.w3.org/1999/xhtml">

<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>%%%TITLEPAGE%%%</title>
<meta http-equiv="Content-Language" content="English">

<script type="text/javascript" src="./js/jquery.min.js"></script>
<script type="text/javascript" src="./js/image-compare.js"></script>
<script src="./js/jquery.flot.js" type="text/javascript" language="javascript"></script>
<script src="./js/jquery.flot.axislabels.js" type="text/javascript" language="javascript"></script>
<script src="./js/jquery.flot.dashes.js" type="text/javascript" language="javascript"></script>

<script>
    $(window).resize(function()
    {
        $('.className').css({
            position:'absolute',
            left: ($(window).width() - $('.className').outerWidth())/2,
            top: ($(window).height() - $('.className').outerHeight())/2
        });

    });

    // To initially run the function:
    $(window).resize();
</script>

<script type="text/javascript">
    $(function () {
      $('.vert_compare').qvertcompare();
      $('.three_compare').qthreecompare();
      $('.cross_compare').qcrosscompare();
    });
    </script>
<style type="text/css">
.ba-layer_tl {position:absolute; top:0; left:0; z-index:3; border-right:3px solid #333; border-bottom:3px solid #333;}
.ba-layer_tr {position:absolute; top:0; left:0; z-index:2; border-bottom:3px solid #333;}
.ba-layer_bl {position:absolute; top:0; left:0; z-index:1; border-right:3px solid #333;}
.ba-layer_br {position:absolute; top:0; left:0; z-index:0;}

.ba-caption_tl {position:absolute; bottom:10px; right:10px; z-index:120; color:#fff; text-align:center; padding:5px; font-size:12px; font-family:arial; display:none;}
.ba-caption_tr {position:absolute; bottom:10px; left: 10px; z-index:120; color:#fff; text-align:center; padding:5px; font-size:12px; font-family:arial; display:none;}
.ba-caption_bl {position:absolute; top:10px;    right:10px; z-index:120; color:#fff; text-align:center; padding:5px; font-size:12px; font-family:arial; display:none;}
.ba-caption_br {position:absolute; top:10px;    left: 10px; z-index:120; color:#fff; text-align:center; padding:5px; font-size:12px; font-family:arial; display:none;}

#container
{
    width: 800px;
    max-width: 800px;
    position: relative;
    margin: 0 auto;
}

div.title
{
    font-family: Verdana;
    font-weight: bold;
    font-size: 14pt;
    padding: 40px 0 10px 0;
}

div.top
{
    width: 100%;
    text-align: right;
    padding-top: 3px;
}

div.top a
{
    text-decoration: none;
}

div.top a:hover
{
    text-decoration: underline;
}

div.caption
{
    background-color: #d8d8d8;
    padding: 15px;
    width: %%%CAPTIONWIDTH%%%px;
    margin: 0 0 70px 0;
    font-family:"Ubuntu", "Calibri", "Lucida Grande", lucida, helvetica, verdana, sans-serif;
    font-size: 11pt;
}

hr
{
    margin: 10px 0 30px 0;
    border: 1px solid black;
}

h2
{
    margin: 90px 0 30px 0;
    /*border-bottom: 1px solid black;*/
}

h3
{
    margin: 5px 0 50px 0;
    font-size: 12pt !important;
    font-weight: normal !important;
}

th {
    text-align: center;
    font-size: 120%;
}

/**************** Image comparison styles ****************/

.ba-layer_tl {position:absolute; top:0; left:0; z-index:3; border-right:2px solid #333; border-bottom:2px solid #333;}
.ba-layer_tr {position:absolute; top:0; left:0; z-index:2; border-bottom:2px solid #333;}
.ba-layer_bl {position:absolute; top:0; left:0; z-index:1; border-right:2px solid #333;}
.ba-layer_br {position:absolute; top:0; left:0; z-index:0;}
.ba-layer_b  {position:absolute; top:0; left:0; z-index:0;}

.ba-caption_tl {position:absolute; bottom:10px; right:10px; z-index:120; color:#fff; text-align:center; padding:5px; font-size:11pt; font-family:"Ubuntu", "Calibri", "Lucida Grande", lucida, helvetica, verdana, sans-serif; display:none; text-shadow: 0 0 0.3em black, 0 0 0.3em black, 0 0 0.3em black;}
.ba-caption_tr {position:absolute; bottom:10px; left: 10px; z-index:120; color:#fff; text-align:center; padding:5px; font-size:11pt; font-family:"Ubuntu", "Calibri", "Lucida Grande", lucida, helvetica, verdana, sans-serif; display:none; text-shadow: 0 0 0.3em black, 0 0 0.3em black, 0 0 0.3em black;}
.ba-caption_bl {position:absolute; top:10px;    right:10px; z-index:120; color:#fff; text-align:center; padding:5px; font-size:11pt; font-family:"Ubuntu", "Calibri", "Lucida Grande", lucida, helvetica, verdana, sans-serif; display:none; text-shadow: 0 0 0.3em black, 0 0 0.3em black, 0 0 0.3em black;}
.ba-caption_br {position:absolute; top:10px;    left: 10px; z-index:120; color:#fff; text-align:center; padding:5px; font-size:11pt; font-family:"Ubuntu", "Calibri", "Lucida Grande", lucida, helvetica, verdana, sans-serif; display:none; text-shadow: 0 0 0.3em black, 0 0 0.3em black, 0 0 0.3em black;}
.ba-caption_b  {position:absolute; top:10px;    left:  0px; z-index:120; color:#fff; text-align:center; padding:5px; font-size:11pt; font-family:"Ubuntu", "Calibri", "Lucida Grande", lucida, helvetica, verdana, sans-serif; display:none; text-shadow: 0 0 0.3em black, 0 0 0.3em black, 0 0 0.3em black;}

.ba-layer_l {position:absolute; top:0; left:0; z-index:1; border-right:2px solid #333;}
.ba-layer_r {position:absolute; top:0; left:0; z-index:0;}

.ba-caption_l {position:absolute; top:10px; right:10px; z-index:120; color:#fff; text-align:center; padding:5px; font-size:11pt; font-family:"Ubuntu", "Calibri", "Lucida Grande", lucida, helvetica, verdana, sans-serif; display:none; text-shadow: 0 0 0.3em black, 0 0 0.3em black, 0 0 0.3em black;}
.ba-caption_r {position:absolute; top:10px; left: 10px; z-index:120; color:#fff; text-align:center; padding:5px; font-size:11pt; font-family:"Ubuntu", "Calibri", "Lucida Grande", lucida, helvetica, verdana, sans-serif; display:none; text-shadow: 0 0 0.3em black, 0 0 0.3em black, 0 0 0.3em black;}

</style>
"""

html4Way = """
<div class="cross_compare" style="width: %%WIDTH%%px; height: %%HEIGHT%%px; cursor: crosshair; overflow: hidden; position: relative;">
    <img src="%%F1%%" alt="%%F1DESC%%" width="%%WIDTH%%" height="%%HEIGHT%%" style="display: none;">
    <img src="%%F2%%" alt="%%F2DESC%%" width="%%WIDTH%%" height="%%HEIGHT%%" style="display: none;">
    <img src="%%F3%%" alt="%%F3DESC%%" width="%%WIDTH%%" height="%%HEIGHT%%" style="display: none;">
    <img src="%%F4%%" alt="%%F4DESC%%" width="%%WIDTH%%" height="%%HEIGHT%%" style="display: none;">
</div>
"""
htmlCaption = """
<div class="caption"><b>%%TITLE%%</b>%%DESC%%</div>"""

def comparison4Way(techniques, imgAlias, output, w, h):
    """
    This method will fill the HTML pattern to generate meanfull GHMLT
    :param descripts: Array of description of the technqiues
    :param files: Array of the images files
    :param w: width of the images
    :param h: height of the images
    :return: the filled HTML code (str)
    """
    rawHtml = html4Way
    rawHtml = rawHtml.replace("%%WIDTH%%", str(w))
    rawHtml = rawHtml.replace("%%HEIGHT%%", str(h))
    rawHtml = rawHtml.replace("%%F1%%", techniques[0].images[imgAlias].replace(output, "./"))
    rawHtml = rawHtml.replace("%%F1DESC%%", techniques[0].name)
    rawHtml = rawHtml.replace("%%F2%%", techniques[1].images[imgAlias].replace(output, "./"))
    rawHtml = rawHtml.replace("%%F2DESC%%", techniques[1].name)
    rawHtml = rawHtml.replace("%%F3%%", techniques[2].images[imgAlias].replace(output, "./"))
    rawHtml = rawHtml.replace("%%F3DESC%%", techniques[2].name)
    rawHtml = rawHtml.replace("%%F4%%", techniques[3].images[imgAlias].replace(output, "./"))
    rawHtml = rawHtml.replace("%%F4DESC%%", techniques[3].name)
    return rawHtml

class HTMLRow():
    """
    Class to generate HTML row by adding tables to organize all the images
    """
    def __init__(self):
        self.el = []
        self.titles = []
    
    def add(self, e, title):
        self.el.append(e)
        self.titles.append(title)
    
    def generateHTML(self):
        if len(self.el) == 0:
            print("WARN: Empty row, normal ?")
        elif len(self.el) == 1:
            return self.el[0]
        else:
            code = "<table><tr>"
            for title in self.titles:
                code += "<th>"+title+"</th>"
            code += "</tr><tr>"
            for e in self.el:
                code += "<td>" + e + "</td>"
            code += "</tr></table>"
            return code

def readOptional(e, name, default=""):
    v = default
    if(name in e.attrib):
        v = e.attrib[name]
    return v

class Technique:
    def __init__(self, name, filename,
                reference, prefix): # Optional informations
        self.name = name
        self.filename = filename

        # Each technique can have different reference
        # because some time techniques do not compute the same light transport
        self.reference = reference

        # Prefix: Some techniques can be a variation from another techniquje
        # So we need to make a distinction between them
        self.prefix = prefix

        # Associated images
        self.images = {}
    @staticmethod
    def parseXMLEntry(e):
        # Read optional informations
        reference = readOptional(e, "reference", "Reference")
        prefix = readOptional(e, "prefix")
        return Technique(e.attrib["name"], e.attrib["filename"],
                         reference, prefix)

    def isRef(self):
        return self.filename == "Ref" or self.filename == "Reference"

    def filenameTime(self, time):
        """Return filename"""
        # Special case if ref
        if(self.isRef()):
            return self.filename

        if(self.prefix == ""):
            return self.filename+"_"+time
        else:
            return self.filename+"_"+self.prefix+"_"+time

class Compare:
    def __init__(self, title, description):
        """
        Create empty compare entry
        """
        self.title = title
        self.desc = description
        self.techniques = []

    @staticmethod
    def parseXMLEntry(e, techDict):
        # Create empty compare entry
        c = Compare(e.attrib["title"], readOptional(e, "desc"))

        # Append to the compare entry all the technique associated to him
        for el in e.iter('Element'):
            if(el.attrib["name"] in techDict):
                c.techniques.append(techDict[el.attrib["name"]])
            else:
                print("ERROR in compare: " + el.attrib["name"]  + " (not exist)")
                sys.exit(-1)
        return c

def readXMLComparisons(file):
    """
    This method will read the XML files to extract which
    :param file: The configuration file
    :return: techniques dict and array of comparison
    """
    tree = ET.parse(file)
    root = tree.getroot()
    
    techDict = {}
    comps = []

    # Read all techniques and create a dictionary
    for tech in root.iter('Technique'):
        newTech = Technique.parseXMLEntry(tech)
        techDict[newTech.name] = newTech

    # Here it can be little tricky. Indeed, we need to associated
    # at each image the reference to use it for bias and other computations
    for tech in techDict.keys():
        objTech = techDict[tech]
        objTech.reference = techDict[objTech.reference]

    # --- Load comparison table
    for compEntry in root.iter('Compare'):
        comps.append(Compare.parseXMLEntry(compEntry, techDict))

    return techDict, comps

class SectionCurve:
    
    def __init__(self, title, description):
        self.title = title
        self.description = description
        self.curves = []
        
    def addCurves(self, curve):
        self.curves.append(curve)

    @staticmethod
    def parseXMLEntry(e):
        description = ""
        if("desc" in e.attrib):
            description = e.attrib["desc"]
        return SectionCurve(e.attrib["name"], description)

    def HTMLcode(self, rep, entries, output, clampTime, step):
        # Create HTML code
        s = "<h2>"+self.title+"</h2>"
        if(self.description != ""):
            s += "<p>"+self.description+"</p>"
        
        # Generate JS
        for c in self.curves:
            textJS = c.generateJS(rep, entries, clampTime, step)
            
            # Write file
            file = open(output+os.path.sep+c.getName()+".js", "w")
            file.write(textJS)
            file.close()
            
            s += '<script src="'+c.getName()+".js"+'" type="text/javascript" language="javascript"></script>'
            s += '<div id="'+c.getName()+'" style="width:960px;height:540px"></div>'
            if ABOVE_CURVES != "":
                s += ABOVE_CURVES
        return s
    
class Curve:
    def __init__(self, name, xlabel, ylabel, log):
        self.csv = name
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.log = log
        
    @staticmethod
    def parseXMLEntry(e):
        return Curve(e.attrib["csv"],
                     e.attrib["xlabel"],
                     e.attrib["ylabel"],
                     e.attrib["log"] == "true")
    
    def getName(self):
        if self.log:
            return "log"+self.csv
        else:
            return self.csv
        
    def generateJS(self, rep, entries, clampTime, step):
        # Create the list of techniques filenames
        listCSVFiles = []
        for e in entries:
            if(e.time == ""):
                listCSVFiles.append(e.technique)
            else:
                listCSVFiles.append(e.technique+","+e.time)
        print(listCSVFiles)
        
        # Read all associated CSV
        useLog = self.log
        if self.csv == "time":
            # Special case when we need to display rendering time
            useLog = False

        techniques = showResults.readAllTechniques(listCSVFiles, rep, step, useLog,  basey='_'+self.csv+'.csv')
        
        # Generate JS file
        totalText = '$(function() { $.plot("#'+self.getName()+'",['
        for i in range(len(techniques)):
            t = techniques[i]
            if(clampTime != -1):
                if not self.log:
                    t.clampTime(clampTime)
                else:    
                    t.clampTime(math.log10(float(clampTime)))
                
            print(entries[i].name, t.name)
            # Data row
            
            totalText += "{data: "
            
            if self.csv == "time":
                if self.log:
                    totalText += str(t.generateConstantDataXLog())
                else:
                    totalText += str(t.generateConstantDataX())
            else:
                totalText += str(t.generatePairData())
            totalText += ', label: "' + entries[i].name + '"'
            if(entries[i].dashed):
                totalText += ', dashes: { show: true }'
            totalText += '}'
            
            if(t != techniques[-1]):
                totalText += ",\n"
        totalText += "]"
        
        # Color dicts
        totalText += ",{colors: ["
        for e in entries:
            totalText += '"' + e.color + '", '
        totalText += ']'
        
        totalText += """, xaxis: {
            axisLabel: '"""+self.xlabel+"""',
            axisLabelUseCanvas: true,
            axisLabelFontSizePixels: 14,
            axisLabelFontFamily: 'Ubuntu, Calibri, Lucida Grande',
            axisLabelPadding: 10,
        }, 
        yaxis: {
            ticks: 10,
            tickDecimals: 3,
            axisLabel: '"""+self.ylabel+"""',
            axisLabelUseCanvas: true,
            axisLabelFontSizePixels: 14,
            axisLabelFontFamily: 'Ubuntu, Calibri, Lucida Grande',
            axisLabelPadding: 10,
        }});\n"""
        
        # Close the end JS
        totalText += '});' 
        
        return totalText

class CurveEntry:
    def __init__(self, name, color, technique, dashed, time):
        self.name = name
        self.color = color
        self.technique = technique
        self.dashed = dashed
        self.time = time
    @staticmethod
    def parseXMLEntry(e):
        time = ""
        if("time" in e.attrib):
            time = e.attrib["time"]
        return CurveEntry(e.attrib["name"], 
                          e.attrib["color"],
                          e.attrib["technique"],
                          e.attrib["dashed"] == "true",
                          time)

def readXMLCurves(file):
    """
    Function to read curves config
    :param file:
    :return: all the entries (techniques we want to plot) and sections (what metric we want to use)
    """
    tree = ET.parse(file)
    root = tree.getroot()

    entries = []
    sections = []
    
    for e in root.findall('./Curves/Entries/Entry'):
        entries.append(CurveEntry.parseXMLEntry(e))
    
    for e in root.findall('./Curves/Section'):
        sections.append(SectionCurve.parseXMLEntry(e))
        for f in e.iter('Curve'):
            sections[-1].addCurves(Curve.parseXMLEntry(f))

    return entries, sections

if __name__ == "__main__":
    #tracker = SummaryTracker()
    
    # --- Read all params
    parser = optparse.OptionParser()

    # Input/Output options
    parser.add_option('-i','--input', help='input')
    parser.add_option('-o','--output', help='output')
    parser.add_option('-n','--name', help='scene name', default="NO_NAME")
    parser.add_option('-r','--reference',help='Reference image')

    # Informations
    parser.add_option('-t','--time', help='time content (sec)')
    parser.add_option('-e','--exposure', help='time content', default=0)
    parser.add_option('-c','--compare', help='compare layout xml file', default="")
    parser.add_option('-j', '--jsdir', help='JS dir scripts', default="./data/js")

    # If we want to skip the image generation (usefull some time for HTML changes)
    parser.add_option('-S','--skip', help='skip image generation', default=False, action="store_true")
    parser.add_option('-s','--step', help='steps during the image generation', default=0)

    # Options for curves and related stuff
    parser.add_option('-m','--metric', help='Show RMSE plots', default=False, action="store_true")
    parser.add_option('-C','--clampTime', help='Clamp time into curves', default="-1")
    copyHDR = True
    
    (opts, args) = parser.parse_args()
    exposure = int(opts.exposure)
    clampTime = int(opts.clampTime)
    step = int(opts.step) # Only usefull for curve generation
    
    if(not os.path.exists(opts.input)):
        print("Input path: "+opts.input+" not exists, QUIT")
        sys.exit(3)
    
    refPath = opts.input + os.path.sep + "Ref.hdr"
    (widthRef,heightRef),pRef = rgbe.io.read(refPath)
    wTot,hTot = widthRef,heightRef
    
    if(opts.time == None):
        print("Need to specified the time, QUIT")
        sys.exit(1)
    
    if(opts.compare == ""):
        print("Need the compare layout, QUIT")
        sys.exit(1)

    # Read all the configurations...
    techDict, comp = readXMLComparisons(opts.compare)
    curvesEntries, curvesSections = readXMLCurves(opts.compare)

    print("Read all images files: "+opts.input)
    
    # --- Create directory if not exist
    if(not os.path.exists(opts.output)):
        os.makedirs(opts.output)
    
    # In case of js directory redelete it
    # to be sure to always to be update
    pathJS = opts.output+"/js"
    if(os.path.exists(pathJS)):
        shutil.rmtree(pathJS)
    shutil.copytree(opts.jsdir, pathJS)
    
    # === Convert file name
    otherImages = [("dx", "Abs X gradient", exposure+3, "dxAbs"),
                   ("dy", "Abs Y gradient", exposure+3, "dyAbs")]

    black_image = opts.output+os.path.sep+"black.png"
    for t in techDict.keys():
        # Take the technique and generate all the file paths
        tech = techDict[t]
        filenameTime = tech.filenameTime(str(opts.time))
        filenameRefHDR = opts.input+os.path.sep+tech.reference.filenameTime(str(opts.time))+".hdr"

        # Tone map
        filenameHDR = opts.input+os.path.sep+filenameTime+".hdr"
        filenameIMG = opts.output+os.path.sep+filenameTime+".png"
        # If the image is not found, do a special treatment
        if(not os.path.exists(filenameHDR)):
            print("NO HDR FILE FOUND FOR: "+filenameHDR)
            # Use all black images
            tech.images["tonemap"] = black_image
            tech.images["bias"] = black_image
            for other in otherImages:
                tech.images[other[0]] = black_image

            continue

        if(not opts.skip):
            # Sometime, we do not want to generates the images...
            # So we may want skip it !
            (width,height),p = rgbe.io.read(filenameHDR)
            rgbe.utils.applyExposureGamma(p, exposure, 2.2)

            im = Image.new("RGB", (width,height))
            rgbe.utils.copyPixeltoPIL(width, height, p, im)
            print("Save "+filenameIMG)
            im.save(filenameIMG, optimize=True)
        tech.images["tonemap"] = filenameIMG

        # Generate bias images
        filenameIMG_BIAS = opts.output+os.path.sep+filenameTime+"_bias.png"
        if(not opts.skip):
            if(tech.isRef()):
                # No bias image need to be computed here
                filenameIMG_BIAS = black_image
            else:
                generateFigures.saveNPImageRef(filenameHDR,filenameRefHDR,filenameIMG_BIAS)
        tech.images["bias"] = filenameIMG_BIAS

        # Generate other images
        for other in otherImages:
            alias, desc, newExp, newPrefix = other
            if(tech.isRef()):
                # No bias image need to be computed here
                tech.images[alias] = black_image
                continue

            # Generate new file names
            filename = tech.filename+"_"+newPrefix+"_"+str(opts.time)
            filenameHDR = opts.input+os.path.sep+filename+".hdr"
            filenameIMG = opts.output+os.path.sep+filename+".png"

            # If there is no HDR, associate a black image
            if(not os.path.exists(filenameHDR)):
                print("NO HDR FILE FOUND FOR: "+filenameHDR)
                tech.images[alias] = black_image
                continue

            if(not opts.skip):
                # Sometime, we do not want to generates the images...
                # So we may want skip it !
                (width,height),p = rgbe.io.read(filenameHDR)
                rgbe.utils.applyExposureGamma(p, newExp, 2.2)

                im = Image.new("RGB", (width,height))
                rgbe.utils.copyPixeltoPIL(width, height, p, im)
                print("Save "+filenameIMG)
                im.save(filenameIMG, optimize=True)

            tech.images[alias] = filenameIMG


    ################################################
    ################################################
    ## From this point all the images have been
    ## generated, now we need to generate HTML
    ## and JS files
    ################################################
    ################################################

    print("Generate HTML ...")
    htmlCode = htmlHead
    htmlCode = htmlCode.replace("%%%TITLEPAGE%%%", opts.name + " [" + str(opts.time) + "sec]")
    htmlCode = htmlCode.replace("%%%CAPTIONWIDTH%%%", str(int(wTot)*3 - 20))
    
    htmlCode += "</head><body>"
    if(not PRODMOD):
        htmlCode += "Page generated: "+datetime.datetime.now().strftime("%A %d %Y, %Hpm")+"<br/>"
    
    htmlCode += "<h2>HTML Results</h2>"
    htmlCode += '<p style="120%">'
    htmlCode += 'Scene name: ' + str(opts.name) + '<br/>'
    htmlCode += 'Comparison time: '+ str(opts.time) + " seconds"
    htmlCode += '</p>'

    # Generate all the image sections
    for c in comp:
        # Tone mapped images
        row = HTMLRow()
        row.add(comparison4Way(c.techniques,"tonemap", opts.output,
                               wTot, hTot), "Rendered images")

        # Bias images
        row.add(comparison4Way(c.techniques,"bias", opts.output,
                               wTot, hTot), "Bias (N/P false color)")

        # Other images ...
        for imgSetup in otherImages:
            row.add(comparison4Way(c.techniques,imgSetup[0], opts.output,
                                   wTot, hTot), imgSetup[1])

        htmlCode += row.generateHTML()
        
        # === Inject description caption
        htmlCaptionNew = htmlCaption.replace("%%TITLE%%", c.title)
        if(c.desc == ""):
            htmlCaptionNew = htmlCaptionNew.replace("%%DESC%%", "")
        else:
            htmlCaptionNew = htmlCaptionNew.replace("%%DESC%%", "<br/>"+c.desc)
        
        htmlCode += htmlCaptionNew
        htmlCode += "\n<br>\n"

    # Generate all the curves sections (if we want to)
    if(opts.metric):
        for section in curvesSections:
            htmlCode += section.HTMLcode(opts.input, curvesEntries, opts.output, clampTime, step)
    else:
        print("No curve generation requested... skip this part !")
            
    htmlCode += "\n</body>\n</html>"
    f = open(opts.output+os.path.sep+"index.html", "w")
    f.write(htmlCode)
    f.flush()
    f.close()

    # Black img
    im = Image.new("RGB", (widthRef,heightRef))
    im.save(black_image)
