"""Generate Figure S9A and assemble Figure S9B from the saved source package."""
from pathlib import Path
import sys
import numpy as np, zlib
from pypdf.generic import NameObject, NumberObject
import copy, io, json
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import RectangleObject, ContentStream, FloatObject
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color
from reportlab.pdfbase.pdfmetrics import stringWidth, registerFont
from reportlab.pdfbase.ttfonts import TTFont
import argparse
import matplotlib

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

# Use the font files supplied by matplotlib in every supported runtime.
fonts = Path(matplotlib.get_data_path()) / "fonts/ttf"
registerFont(TTFont("S9Sans", str(fonts / "DejaVuSans.ttf")))
registerFont(TTFont("S9Sans-Bold", str(fonts / "DejaVuSans-Bold.ttf")))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("folder", type=Path, help="Extracted S9 package containing S9A/ and S9B/")
folder = parser.parse_args().folder.expanduser().resolve()
summary=json.loads((folder/'S9B/Figure_S9B_examples_summary.json').read_text())
assert len(summary)==5

import pandas as pd
from xenium_pipeline.supplementary_s9 import plot_composition
composition = pd.read_csv(folder/'provenance/Figure_S9A_sample_composition.csv', float_precision='round_trip')
cohort = pd.read_csv(CODE_ROOT/'tables/cohort_samples.csv')
assert len(cohort) == 35
plot_composition(composition, cohort, folder/'S9A', smooth_muscle_as_uncertain=True)
sources=[]
# Source broad labels are kept on disk; only the embedded polygon display is pooled.
recolor_audit=[]
def pool_b_polygon_color(page):
    image=next(im for im in page.images if im.name=='I2.png')
    pixels=np.asarray(image.image.convert('RGB')).copy()
    gold=np.array([213,172,48]);purple=np.array([147,104,172])
    alpha=(255-pixels[:,:,2].astype(float))/(255-gold[2])
    expected=255-alpha[:,:,None]*(255-gold)
    mask=(alpha>.02)&(np.max(np.abs(expected-pixels),axis=2)<=1.5)
    pixels[mask]=np.rint(255-alpha[mask,None]*(255-purple)).clip(0,255).astype(np.uint8)
    obj=image.indirect_reference.get_object()
    obj._data=zlib.compress(pixels.tobytes())
    obj[NameObject('/Filter')]=NameObject('/FlateDecode')
    obj[NameObject('/ColorSpace')]=NameObject('/DeviceRGB')
    obj[NameObject('/BitsPerComponent')]=NumberObject(8)
    obj.pop(NameObject('/DecodeParms'),None)
    return int(mask.sum())

def bold_embedded_text(page):
    # Stroke only copied text in its original color; preserve source crop geometry.
    # Type-3 glyphs paint their own paths, so text rendering mode alone is insufficient.
    for font in page['/Resources']['/Font'].values():
        font = font.get_object()
        if font.get('/Subtype') == '/Type3':
            for glyph in font['/CharProcs'].values():
                glyph = glyph.get_object()
                glyph_stream = ContentStream(glyph, page.pdf)
                glyph_ops = []
                for args, op in glyph_stream.operations:
                    if op in (b'f', b'f*'):
                        glyph_ops.extend([([NumberObject(28)], b'w'), ([NumberObject(1)], b'j')])
                        op = b'B' if op == b'f' else b'B*'
                    glyph_ops.append((args, op))
                glyph_stream.operations = glyph_ops
                glyph.set_data(glyph_stream.get_data())
    stream = ContentStream(page.get_contents(), page.pdf)
    result, stack = [], []
    fill = ([NumberObject(0)], b'g')
    for args, op in stream.operations:
        if op == b'q':
            stack.append(fill)
        elif op == b'Q':
            fill = stack.pop()
        elif op in (b'g', b'rg', b'k'):
            fill = (args, op)
        if op == b'BT':
            color_args, color_op = fill
            result.extend([([], b'q'), (color_args, color_op.upper()),
                           ([FloatObject(.22)], b'w'), ([NumberObject(2)], b'Tr')])
        result.append((args, op))
        if op == b'ET':
            result.append(([], b'Q'))
    stream.operations = result
    page[NameObject('/Contents')] = stream

target_order = ['Tumor Cells', 'Normal Epithelial Cells', 'T Cells', 'pan-APC + B', 'Stromal Cells']
assert {example['featured_target'] for example in summary} == set(target_order)
for source_index, example in sorted(enumerate(summary,1), key=lambda item: target_order.index(item[1]['featured_target'])):
    paths=list((folder/'S9B').glob(f'{source_index:02d}_*/*/Figure_S9B.pdf'))
    assert len(paths)==1,paths
    reader=PdfReader(paths[0]); assert len(reader.pages)==1
    recolor_audit.append(dict(dataset=example['dataset'],sample=example['sample'],recolored_polygon_pixels=pool_b_polygon_color(reader.pages[0])))
    bold_embedded_text(reader.pages[0])
    sources.append((reader,example))

W,H=1235,1024
left=24; square=132; gap=10; step=square+gap
first_top=109; row_step=174
buf=io.BytesIO(); c=canvas.Canvas(buf,pagesize=(W,H))
c.setTitle('Figure S9B - Five Breast Xenium Examples')
c.setAuthor('STpath')
c.setFont('S9Sans-Bold',16)
c.drawString(left+20,H-28,'Representative Breast Xenium Regions')
c.setFont('S9Sans-Bold',10)
headers=[['Registered H&E'],['Annotated cell','polygons'],['Tumor','Cells'],['Normal Epithelial','Cells'],['T Cells'],['pan-APC'],['Stromal Cells'],['Uncertain']]
for j,lines in enumerate(headers):
    c.setFont('S9Sans-Bold',11)
    center=left+j*step+square/2
    for k,line in enumerate(lines):
        c.drawCentredString(center,H-68-k*13,line)
for i,(_,e) in enumerate(sources):
    top=first_top+i*row_step
    y=H-top+10
    ds=e['dataset'].replace('_',' ')
    label=f"{ds}  |  {e['sample']}"
    right=f"{e['featured_target'].replace('pan-APC + B','pan-APC')}: {e['featured_fraction']:.1%}"
    # The original region reference fraction is kept, calculated over retained cells.
    c.setFont('S9Sans-Bold',10.5)
    assert stringWidth(label,'S9Sans-Bold',10.5)+stringWidth(right,'S9Sans-Bold',10.5)+25 < 8*square+7*gap
    c.drawString(left,y,label)
    c.drawRightString(left+7*step+square,y,right)
    if i:
        c.setStrokeColor(HexColor('#dddddd'));c.setLineWidth(.45)
        c.line(left,y+13,left+7*step+square,y+13)
legend=[('Tumor Cells','#C64A50'),('Normal Epithelial Cells','#4B8DBD'),('T Cells','#459B69'),('pan-APC','#9368AC'),('Stromal Cells','#49A8A0'),('Uncertain','#B4B4B4')]
legend_sizes=[14+6+stringWidth(label,'S9Sans-Bold',10)+28 for label,_ in legend]
x=left+(8*square+7*gap-sum(legend_sizes))/2
for (label,color),width in zip(legend,legend_sizes):
    c.setFillColor(HexColor(color));c.rect(x,41,14,7,stroke=0,fill=1)
    c.setFillColor(Color(0,0,0));c.setFont('S9Sans-Bold',10);c.drawString(x+20,40,label)
    x+=width
c.setFont('S9Sans-Bold',9)
c.showPage();c.save()
base=PdfReader(io.BytesIO(buf.getvalue())).pages[0]

# Reposition source PDF content, preserving paths, images, and scale bars.
# Each original export has eight axes arranged in two rows of four.
centers_x=[95.3073684211,315.48,535.6526315789,755.8252631579]
tops=[62.56,261.36]; native=159.12

def place_crop(page,rect,x,y,width,height):
    crop=copy.copy(page)
    crop.cropbox=RectangleObject(rect)
    sx,sy=width/(rect[2]-rect[0]),height/(rect[3]-rect[1])
    matrix=Transformation((sx,0,0,sy,x-rect[0]*sx,y-rect[1]*sy))
    base.merge_transformed_page(crop,matrix,over=True,expand=False)

for i,(reader,e) in enumerate(sources):
    page=reader.pages[0]; ph=float(page.mediabox.height)
    assert abs(float(page.mediabox.width)-938.3295)<.01 and abs(ph-503.82)<.01
    for j in range(8):
        x0=centers_x[j%4]-native/2
        top=tops[j//4]
        rect=[x0,ph-top-native,x0+native,ph-top]
        place_crop(page,rect,left+j*step,H-(first_top+i*row_step)-square,square,square)
# Keep the original continuous scale with its labels, once for all maps.
page=sources[0][0].pages[0];ph=float(page.mediabox.height)
bar=[870.0,ph-387.0,936.0,ph-126.0]
place_crop(page,bar,1165,380,66,261)
writer=PdfWriter();writer.add_page(base)
writer.add_metadata({'/Title':'Figure S9B - Five Breast Xenium Examples','/Subject':'Merged five-row layout using unchanged source PDF panels'})
out=folder/'S9B/Figure_S9B.pdf'
with out.open('wb') as f:writer.write(f)
r=PdfReader(out);assert len(r.pages)==1
text=r.pages[0].extract_text()
for _,e in sources:assert e['sample'] in text,e['sample']
print(out)
print('Single-page PDF, eight columns and five rows; source scale bars and all source panels retained.')

(folder/'S9B/Figure_S9B_display_metadata.json').write_text(json.dumps(dict(figure_display_mapping={'B Cells':'pan-APCs','Smooth Muscle Cells':'Uncertain'},pan_apc_display_label='pan-APC',reference_fractions_changed=False,all_text_bold=True,panel_letter_removed=True,source_pdfs_preserved=True,polygon_recoloring=recolor_audit),indent=2))

# Place the two accepted panels on one page in manuscript order.
panel_a = PdfReader(folder/'S9A/Figure_S9A.pdf').pages[0]
panel_b = PdfReader(out).pages[0]
width = float(panel_b.mediabox.width)
scale = width / float(panel_a.mediabox.width)
height_a = float(panel_a.mediabox.height) * scale
height_b = float(panel_b.mediabox.height)
gap = 24
combined = PdfWriter()
combined_page = combined.add_blank_page(width=width, height=height_a + height_b + gap)
combined_page.merge_transformed_page(panel_a, Transformation().scale(scale).translate(0, height_b + gap))
combined_page.merge_page(panel_b)
labels = io.BytesIO()
label_canvas = canvas.Canvas(labels, pagesize=(width, height_a + height_b + gap))
label_canvas.setFont('S9Sans-Bold', 22)
label_canvas.drawString(8, height_a + height_b + gap - 24, 'A')
label_canvas.drawString(8, height_b - 24, 'B')
label_canvas.save()
combined_page.merge_page(PdfReader(labels).pages[0])
combined.add_metadata({'/Title': 'Figure S9 - Breast Xenium Composition and Spatial References'})
with (folder/'Figure_S9.pdf').open('wb') as stream:
    combined.write(stream)
assert len(PdfReader(folder/'Figure_S9.pdf').pages) == 1
print(folder/'Figure_S9.pdf')
