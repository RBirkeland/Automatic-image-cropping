import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt
import time
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

import shapely.geometry as geometry
import math
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay

if(len(sys.argv) != 4):
    print("Please provide 3 arguments: \nFile path (String) \nThreshold (Int) \nNumber of clusters (Int)")
    sys.exit(0)

print("File path: " + sys.argv[1])
print("Threshold: " + sys.argv[2])
print("Number of clusters: " + sys.argv[3])

### Variables
threshold = int(sys.argv[2])
num_of_clusters = int(sys.argv[3])

img = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(img,0)

start_time = time.time()
surf = cv2.xfeatures2d.SURF_create(threshold)
kp = surf.detect(gray,None)
points = []

for i in range(0,len(kp)):
    points.append(kp[i].pt)

print("Took: %s seconds to find %i points" % ((time.time() - start_time), len(points)))
points = np.array(points)

#img=cv2.drawKeypoints(img,kp, None, color=255)
plt.imshow(img)

def initialize_centroids(points, k):
    """returns k centroids from the initial points"""
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]

def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

c = initialize_centroids(points, num_of_clusters)

def move_centroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

def findBoundingBox(cluster):
    x1 = np.amin(cluster[:,0])
    y1 = np.amin(cluster[:,1])
    x2 = np.amax(cluster[:,0])
    y2 = np.amax(cluster[:,1])
    return patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=1,edgecolor='r',facecolor='none')

start_time = time.time()
#K-means
run=True
i = 0
while(run):
    new_c = move_centroids(points, closest_centroid(points, c), c)
    if(np.array_equal(c, new_c)):
        run = False
    else:
        c = new_c
	i += 1
print('K-means iterations: %i' % i)

#Make scatterplot
closest = closest_centroid(points, c)

print("K-means took: %s seconds" % (time.time() - start_time))


#Put clusters into dict
clusters = {}

for i in range(0, len(points)):
    if clusters.has_key(closest[i]):
	clusters[closest[i]] = np.append(clusters[closest[i]], [points[i]], axis=0)
    else:
	clusters[closest[i]] = [points[i]]

#print(clusters[0])
#print(clusters[0][:,0])

#poly = patches.Polygon(clusters[0], False, fill=True)
#print(poly)

#Find and draw bounding boxes per cluster
fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)

for i in range(0,len(clusters)):
    rect = findBoundingBox(clusters[i])
    ax.add_patch(rect)

#Draw points
plt.scatter(points[:, 0], points[:, 1], c=closest, s=8)
plt.scatter(c[:, 0], c[:, 1], c='r', s=8)





'''

#Not Working stuff!
def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
            """
            Add a line between the i-th and j-th points,
            if not in the list already
            """
            if (i, j) in edges or (j, i) in edges:
                # already added
                return
            edges.add( (i, j) )
            edge_points.append(coords[ [i, j] ])
            coords = np.array([point.coords[0] for point in points])
    	    tri = Delaunay(coords)
    	    edges = set()
    	    edge_points = []
    	    # loop over triangles:
    	    # ia, ib, ic = indices of corner points of the
    	    # triangle
    	    for ia, ib, ic in tri.vertices:
                pa = coords[ia]
                pb = coords[ib]
                pc = coords[ic]
                # Lengths of sides of triangle
                a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
                b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
                c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
                # Semiperimeter of triangle
                s = (a + b + c)/2.0
                # Area of triangle by Heron's formula
                area = math.sqrt(s*(s-a)*(s-b)*(s-c))
                circum_r = a*b*c/(4.0*area)
                # Here's the radius filter.
                #print circum_r
                if circum_r < 1.0/alpha:
            	    add_edge(edges, edge_points, coords, ia, ib)
            	    add_edge(edges, edge_points, coords, ib, ic)
            	    add_edge(edges, edge_points, coords, ic, ia)
    	    m = geometry.MultiLineString(edge_points)
    	    triangles = list(polygonize(m))
    	    return cascaded_union(triangles), edge_points

def plot_polygon(polygon):
    fig = pl.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    margin = .3
    x_min, y_min, x_max, y_max = polygon.bounds
    ax.set_xlim([x_min-margin, x_max+margin])
    ax.set_ylim([y_min-margin, y_max+margin])
    patch = PolygonPatch(polygon, fc='#999999',
                         ec='#000000', fill=True,
                         zorder=-1)
    ax.add_patch(patch)
    return fig




from shapely.geometry import Polygon
from shapely.geometry import Polygon, mapping
import fiona

poly = Polygon(points)
maap = mapping(poly)


# Define a polygon feature geometry with one attribute
schema = {
    'geometry': 'MultiPolygon',
    'properties': {'id': 'int'},
}

# Write a new Shapefile
with fiona.open('my_shp2.shp', 'w', 'ESRI Shapefile', schema) as c:
    ## If there are multiple geometries, put the "for" loop here
    c.write({
        'geometry': mapping(poly),
        'properties': {'id': 123},
    })	

input_shapefile = 'my_shp2.shp'
shapefile = fiona.open(input_shapefile)
points = [geometry.shape(point['geometry'])
          for point in shapefile]

print("Read in file")


alpha = .4
concave_hull, edge_points = alpha_shape(points, alpha=alpha)

plot_polygon(concave_hull)
_ = pl.plot(x,y,'o', color='#f16824')
plot_polygon(concave_hull.buffer(1))
_ = pl.plot(x,y,'o', color='#f16824')


def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
'''

plt.show()








