"""Sparse numerical labels for decoder magnitude contours."""
import numpy as np
import matplotlib.patheffects as pe
from scipy.ndimage import map_coordinates


def label_magnitude(ax, contours, bounds, fontsize=6.5, halo_width=.7, alpha=.85,
                    level_stride=3, first_level=1, min_separation=.17, background=False,
                    value_format='.2g', edge_band=None, adaptive_color=False,
                    fontweight='normal', region_labels=None, feature_colors=None):
    """Label selected log1p-magnitude levels in raw norm units."""
    x0, x1, y0, y1 = bounds
    placed, records = [], []
    for index in range(first_level, len(contours.levels), level_stride):
        candidates = []
        for segment_index, segment in enumerate(contours.allsegs[index]):
            if len(segment) < 30:
                continue
            for point in segment[::max(1, len(segment)//80)]:
                uv = (point-[x0,y0])/[x1-x0,y1-y0]
                edge = np.min(np.r_[uv,1-uv])
                separation = min([np.linalg.norm(uv-p) for p in placed], default=1.)
                edge_ok = edge>.09 if edge_band is None else edge_band[0]<=edge<=edge_band[1]
                if edge_ok and separation>min_separation:
                    score = min(separation,.4)+.25*np.linalg.norm(uv-.5)+.0001*min(len(segment),1000)
                    if edge_band is not None:
                        score -= 2*abs(edge-np.mean(edge_band))
                    candidates.append((score,point,uv,segment_index))
        if not candidates:
            continue
        _, point, uv, segment_index = max(candidates,key=lambda c:c[0])
        value = float(np.expm1(contours.levels[index]))
        text_color = '#242424'
        if adaptive_color and region_labels is not None and feature_colors is not None:
            row = int(np.clip(round(uv[1]*(region_labels.shape[0]-1)),0,region_labels.shape[0]-1))
            col = int(np.clip(round(uv[0]*(region_labels.shape[1]-1)),0,region_labels.shape[1]-1))
            rgb = feature_colors[region_labels[row,col]]
            linear = np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
            luminance = float(linear @ np.array([.2126,.7152,.0722]))
            text_color = '#fafafa' if luminance<.179 else '#202020'
        text = ax.text(*point,format(value,value_format),fontsize=fontsize,color=text_color,
                       alpha=alpha,ha='center',va='center',fontweight=fontweight,
                       bbox=dict(boxstyle='square,pad=0.08',facecolor='white',edgecolor='none',alpha=.85) if background else None)
        if halo_width:
            text.set_path_effects([pe.withStroke(linewidth=halo_width,foreground='white')])
        placed.append(uv)
        records.append(dict(position=point.tolist(),raw_value=value,
                            level_index=index,segment_index=segment_index,text_color=text_color))
    return records


def add_uphill_arrows(ax, contours, values, xs, ys, labels,
                      length_fraction=.018, offset_fraction=.035, linewidth=.55, alpha=.6):
    """Place small arrows near labels, verifying the field rises along each arrow."""
    span = np.array([xs[-1]-xs[0],ys[-1]-ys[0]])
    origin = np.array([xs[0],ys[0]])
    gy,gx = np.gradient(values,ys,xs)
    def sample(field, point):
        uv = (point-origin)/span
        return float(map_coordinates(field,[[uv[1]*(len(ys)-1)],[uv[0]*(len(xs)-1)]],order=1,mode='nearest')[0])
    records=[]
    for label in labels:
        path=contours.allsegs[label['level_index']][label['segment_index']]
        distance=np.linalg.norm((path-label['position'])/span,axis=1)
        uv=(path-origin)/span
        eligible=(distance>=offset_fraction)&(distance<=offset_fraction*1.7)&(uv>.04).all(1)&(uv<.96).all(1)
        for other in labels:
            eligible &= np.linalg.norm((path-other['position'])/span,axis=1)>.025
        indices=np.flatnonzero(eligible)
        if not len(indices):continue
        start=path[indices[np.argmin(distance[indices])]]
        gradient=np.array([sample(gx,start),sample(gy,start)])
        if np.linalg.norm(gradient)<1e-10:continue
        delta=gradient/np.linalg.norm(gradient)*span.min()*length_fraction
        for _ in range(4):
            end=start+delta
            samples=[sample(values,start+t*delta) for t in np.linspace(0,1,9)]
            if np.all(np.diff(samples)>0) and np.all(end>origin) and np.all(end<origin+span):break
            delta*=.65
        else:continue
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color='#333333',
                    lw=linewidth,alpha=alpha,mutation_scale=5,shrinkA=0,shrinkB=0))
        records.append(dict(start=start.tolist(),end=end.tolist(),
                            raw_start=float(np.expm1(samples[0])),raw_end=float(np.expm1(samples[-1]))))
    return records
