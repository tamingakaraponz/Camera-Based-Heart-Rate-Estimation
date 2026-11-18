"""Detection/geometry-only A/B/C overlay audit. No RGB pulse, PPG, POS or FFT."""
import json,time
from pathlib import Path
from datetime import datetime
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ippg_v2_refactored as core

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent/'dataset for ipph'
COLORS={'A':(40,190,220),'B':(200,150,30),'C':(90,200,60)}
def polygons(points):
    # Project-specific candidate geometry, not an exact paper mask reproduction.
    ex=points[263]-points[33];scale=float(np.linalg.norm(ex));ex=ex/scale;ey=np.array([-ex[1],ex[0]])
    if np.dot(ey,points[152]-points[10])<0:ey=-ey
    forehead_height=float(np.dot(points[9]-points[10],ey));center=points[10]+ey*(.52*forehead_height)
    hw=.25*scale;hh=.16*forehead_height
    a=np.array([center-ex*hw-ey*hh,center+ex*hw-ey*hh,center+ex*hw+ey*hh,center-ex*hw+ey*hh])
    result={'A':a}
    # Existing array names are image-side conventions: map labels to subject side.
    for label,ids in [('B',core.RIGHT_CHEEK_LANDMARKS),('C',core.LEFT_CHEEK_LANDMARKS)]:
        p=points[ids].astype(float);center=p.mean(axis=0);result[label]=center+(p-center)*.78-ey*(.035*scale)
    return {k:np.rint(p).astype(np.int32) for k,p in result.items()}
def draw(frame,poly):
    output=frame.copy()
    for label,p in poly.items():
        mask=np.zeros(frame.shape[:2],np.uint8);cv2.fillPoly(mask,[p],255);sel=mask>0;output[sel]=(.75*output[sel]+.25*np.array(COLORS[label])).astype(np.uint8);cv2.polylines(output,[p],True,COLORS[label],2);c=np.rint(p.mean(axis=0)).astype(int);cv2.putText(output,label,tuple(c),cv2.FONT_HERSHEY_SIMPLEX,.65,COLORS[label],2)
    return output

def main():
    out=ROOT/'outputs'/('roi_overlay_abc_'+datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir();results=[]
    for n in (3,5,8,9,10,11):
        subject=f'subject{n}';video=next(DATA.glob(f'{subject}-*/{subject}/vid.avi'));cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);tracker=core.MediaPipeRegionTracker(core.IPPGConfig(show_preview=False));target=[2,10,20,30,40,50];saved={};records=[];miss=0;frames=0;last=None;maxshift=0.;wall=time.perf_counter()
        try:
            while True:
                ok,frame=cap.read()
                if not ok:break
                t=frames/fps
                if t>50.1:break
                start=time.perf_counter();regions=tracker.detect(frame,int(round(t*1000)));ms=(time.perf_counter()-start)*1000;frames+=1
                if regions is None:miss+=1;continue
                points=tracker._smoothed_points.copy();poly=polygons(points);areas={k:float(cv2.contourArea(p)) for k,p in poly.items()};centers=np.array([p.mean(axis=0) for p in poly.values()]);shift=0. if last is None else float(np.max(np.linalg.norm(centers-last,axis=1)));last=centers;maxshift=max(maxshift,shift)
                records.append(dict(time=t,detect_ms=ms,areas=areas,center_step_px=shift,in_frame=all(np.all((p[:,0]>=0)&(p[:,0]<frame.shape[1])&(p[:,1]>=0)&(p[:,1]<frame.shape[0])) for p in poly.values())))
                for sec in target:
                    if sec<=t<sec+1/fps and sec not in saved:
                        old={'A':np.rint(points[core.FOREHEAD_LANDMARKS]).astype(np.int32),'B':np.rint(points[core.RIGHT_CHEEK_LANDMARKS]).astype(np.int32),'C':np.rint(points[core.LEFT_CHEEK_LANDMARKS]).astype(np.int32)}
                        x1,y1,x2,y2=regions.face_box;pad=15;x1=max(0,x1-pad);y1=max(0,y1-pad);x2=min(frame.shape[1],x2+pad);y2=min(frame.shape[0],y2+pad)
                        saved[sec]=(frame[y1:y2,x1:x2].copy(),draw(frame,old)[y1:y2,x1:x2],draw(frame,poly)[y1:y2,x1:x2]);cv2.imwrite(str(out/f'{subject}_{sec}s_candidate.png'),draw(frame,poly));np.savez_compressed(out/f'{subject}_{sec}s_points.npz',landmarks=points,**poly)
        finally:cap.release();tracker.close()
        fig,axs=plt.subplots(6,3,figsize=(10,17),layout='constrained')
        for i,sec in enumerate(target):
            for j,title in enumerate(('Original','Existing regions','Candidate A/B/C')):
                ax=axs[i,j];ax.axis('off')
                if sec in saved:ax.imshow(saved[sec][j][:,:,::-1])
                else:ax.text(.5,.5,'Unavailable',ha='center',va='center')
                ax.set_title(f'{sec}s | {title}',fontsize=9)
        fig.suptitle(subject+' | geometry only | B/C = subject left/right');fig.savefig(out/f'{subject}_comparison.png',dpi=130);plt.close(fig)
        info=dict(subject=subject,frames_checked=frames,detection_missing=miss,snapshots=list(saved),all_candidate_vertices_in_frame=all(r['in_frame'] for r in records),nonpositive_area_frames=sum(any(v<=0 for v in r['areas'].values()) for r in records),detect_ms_median=float(np.median([r['detect_ms'] for r in records])),detect_ms_p95=float(np.percentile([r['detect_ms'] for r in records],95)),max_center_step_px=maxshift,wall_s=time.perf_counter()-wall)
        results.append(info);(out/f'{subject}_geometry.json').write_text(json.dumps(records),encoding='utf-8');(out/'manifest.json').write_text(json.dumps(dict(scope='Detection and ROI geometry only. No pulse/BPM/PPG computed. Candidate geometry is an anatomical approximation, not published mask replication. Hair exclusion not guaranteed by landmarks.',subjects=results),indent=2),encoding='utf-8');print(subject,info,flush=True)
    print('OUTPUT',out,flush=True)
if __name__=='__main__':main()
