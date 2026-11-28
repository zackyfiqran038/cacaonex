import math, random
from shapely.geometry import Polygon, Point
from shapely.ops import transform
from pyproj import CRS, Transformer

def plan(vertices, spacing, seed):
    if not 2 <= spacing <= 1000 or len(vertices)<3 or len(vertices)>500: raise ValueError('Batas atau interval tidak valid.')
    if any(not (-12<=lat<=7 and 94<=lon<=142) for lat,lon in vertices): raise ValueError('Gunakan koordinat wilayah Indonesia.')
    p=Polygon([(lon,lat) for lat,lon in vertices])
    if not p.is_valid or p.area==0: raise ValueError('Polygon berpotongan atau tidak valid.')
    c=p.centroid; zone=min(60,max(1,int((c.x+180)//6)+1)); crs=CRS.from_epsg((32600 if c.y>=0 else 32700)+zone)
    f=Transformer.from_crs(4326,crs,always_xy=True).transform; inv=Transformer.from_crs(crs,4326,always_xy=True).transform
    q=transform(f,p); a,b,d,e=q.bounds
    if (d-a)*(e-b)>100000000 or (math.ceil((d-a)/spacing)+1)*(math.ceil((e-b)/spacing)+1)>100000: raise ValueError('Area terlalu besar atau interval terlalu kecil.')
    rng=random.Random(seed); ox=rng.random()*spacing; oy=rng.random()*spacing; points=[]
    for j in range(math.ceil((e-b)/spacing)):
        for i in range(math.ceil((d-a)/spacing)):
            x=a+ox+i*spacing;y=b+oy+j*spacing
            if q.contains(Point(x,y)):
                lon,lat=inv(x,y);points.append(dict(code=f'P{len(points)+1:03}',lat=lat,lon=lon))
    if len(points)>2000: raise ValueError('Lebih dari 2000 titik; perbesar interval.')
    return dict(area=q.area,spacing=spacing,seed=seed,vertices=vertices,points=points)
