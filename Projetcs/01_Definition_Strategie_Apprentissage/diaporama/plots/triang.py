#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from basics import *

# from tools_glb.dic import tools as dic_tls ;
from vision._tools import calc_R_xy ;

plts.InvBWfig() ;
path_print = tls.Path() / '..' / 'Figures' ;
orange = (0.9,0.5,0) ;
# ------------------------------------- #

plts.set_font_size( 12 ) ;
width = 12 ;
height = 10 ;

l = 40. ;  L = 100. ;
x0 = np.linspace( -l, l, 10 ) ;
y0 = np.linspace( -L, L, 17 ) ;

x, y = np.meshgrid( x0, y0 ) ;
z = np.zeros_like(x) ;


tta = 17. / 180. * np.pi ;
d = 160. ;
ttax = -15. / 180. * np.pi ;
ttay = -3. / 180. * np.pi ;

i = 6 ; j = 4 ;  ind = i * x.shape[1] + j ;
n = np.empty( (3,2), dtype="float64" ) ;
n[0] = x[i,j] ;  n[1] = y[i,j] ;  n[2,0] = z[i,j] ;
n[2,1] = -200. ;

color_image = "0.4" ;

# ------------------------------------- #
def xyz_to_xy( xyz, R, T ):
  xy = xyz.astype("float64").copy() ;
  R = R.astype("float64") ; T = T.astype("float64") ;
  xy[0] += T[0] ; xy[1] += T[1] ; xy[2] += T[2] ;
  xy = R @ xy ;
  xy[0] /= xy[2] ;  xy[1] /= xy[2] ;
  return xy ;
# ------------------------------------- #

# ------------------------------------- #
def xy_to_xyz( xy, R, T, d ):
  xyz = xy.copy() ;
  xyz[0] *= d ;  xyz[1] *= d ;  xyz[2] = d ;
  xyz = R.transpose() @ xyz ;
  xyz[0] -= T[0] ; xyz[1] -= T[1] ; xyz[2] -= T[2] ;
  return xyz ;
# ------------------------------------- #

# ------------------------------------- #
def plot_wire_2d( ax, xy, clr, zorder=None ):
  print(xy.shape) ;
  for i in range(xy.shape[1]):
    ax.plot( xy[0,i], xy[1,i], clr, zorder=zorder ) ;
  # ---- #
  for i in range(xy.shape[2]):
    ax.plot( xy[0,:,i], xy[1,:,i], clr, zorder=zorder ) ;
  # ---- #
# ------------------------------------- #


R = np.array( [ [np.cos(tta), 0., -np.sin(tta)], [0., 1., 0.], [np.sin(tta), 0., np.cos(tta)] ]  ) ;
T = R.transpose() @ np.array( [0., 0., 400.] ).reshape( 3, 1 ) ;
T = T.flatten() ;

T2 = T.copy() ;
T2[0] = -T[0] ;

R_cam = calc_R_xy( ttax , ttay )
T_cam = np.array([0., 0., 800.], dtype="float64").reshape(3,1) ;
T_cam = R_cam.transpose() @ T_cam ;
T_cam = T_cam.flatten() ;
print("\nT_cam", T_cam ) ;
print("T", T ) ;

# fig, ax = plts.buildFigAx( width, height, b3D=False ) ;
fig = plts.Figure_Manager( width, height ) ;
ax = fig.ax ;

ax.plot([0], [0], color_image, label="image of the object") ;

xyz = np.empty( (3, x.size) ) ;
xyz[0] = x.flatten() ; xyz[1] = y.flatten() ;  xyz[2] = z.flatten() ;
xy = xyz_to_xy( xyz, R_cam, T_cam ).reshape(3, x.shape[0], x.shape[1]) ;
plot_wire_2d( ax, xy, "w" ) ;
# plot_wire = ax.plot_wireframe( x, y, z, color="k", rstride=1, cstride=1 ) ;

# PROJECTION ON CAMERA 1 #

xy = xyz_to_xy( xyz, R, T ) ;
xyz_2 = xy_to_xyz( xy, R, T, d) ;
xy = xyz_to_xy( xyz_2, R_cam, T_cam ).reshape(3, x.shape[0], x.shape[1]) ;
plot_wire_2d( ax, xy, color_image, zorder=2 ) ;
# plot_wire = ax.plot_wireframe( xyz_2[0].reshape(x.shape), xyz_2[1].reshape(x.shape), xyz_2[2].reshape(x.shape), color="r", rstride=1, cstride=1 ) ;
# print("xyz_2\n", xyz_2[:,:3]) ;



xyz_3 = np.empty( (3, 2), dtype="float64" ) ;
xyz_3[0,0] = xyz_2[0,ind] ;  xyz_3[1,0] = xyz_2[1,ind] ;  xyz_3[2,0] = xyz_2[2,ind] ;
xyz_3[0,1] = x[i,j] ;  xyz_3[1,1] = y[i,j] ;  xyz_3[2,1] = z[i,j] ;
xy = xyz_to_xy( xyz_3, R_cam, T_cam ) ;
ax.plot( xy[0], xy[1], "--", color="0.5", zorder=1 ) ;
ax.plot( xy[0], xy[1], "o", color=orange, markersize=5, zorder=4 ) ;

xyz_3[:,1] = -T ;
xy = xyz_to_xy( xyz_3, R_cam, T_cam ) ;
ax.plot( xy[0], xy[1], "--", color="0.5", zorder=3 ) ;


# PROJECTION ON CAMERA 2 #

xy = xyz_to_xy( xyz, R, T ) ;
xyz_2 = xy_to_xyz( xy, R.transpose(), T2, d) ;
xy = xyz_to_xy( xyz_2, R_cam, T_cam ).reshape(3, x.shape[0], x.shape[1]) ;
plot_wire_2d( ax, xy, color_image, zorder=2 ) ;


xyz_3 = np.empty( (3, 2), dtype="float64" ) ;
xyz_3[0,0] = xyz_2[0,ind] ;  xyz_3[1,0] = xyz_2[1,ind] ;  xyz_3[2,0] = xyz_2[2,ind] ;
xyz_3[0,1] = x[i,j] ;  xyz_3[1,1] = y[i,j] ;  xyz_3[2,1] = z[i,j] ;
xy = xyz_to_xy( xyz_3, R_cam, T_cam ) ;
ax.plot( xy[0], xy[1], "--", color="0.5", zorder=1, label="perspective beams" ) ;
ax.plot( xy[0], xy[1], "o", color=orange, markersize=5, zorder=4 ) ;

xyz_3[:,1] = -T2 ;
xy = xyz_to_xy( xyz_3, R_cam, T_cam ) ;
ax.plot( xy[0], xy[1], "--", color="0.5", zorder=3 ) ;


# NORMAL TO THE SURFACE #
# xy = xyz_to_xy( n, R_cam, T_cam ) ;
# ax.plot( xy[0], xy[1], "-", color="r", linewidth=2, zorder=3, label="normal to the surface" ) ;





ax.legend( loc='upper right') ;

# CAMERA CENTERS #

xy = xyz_to_xy( -T.reshape(3,1), R_cam, T_cam ) ;
ax.plot( xy[0], xy[1], "wo" ) ;
# ax.plot( -T[0], -T[1], -T[2], "ko" ) ;


xy = xyz_to_xy( -T2.reshape(3,1), R_cam, T_cam ) ;
ax.plot( xy[0], xy[1], "wo" ) ;
# ax.plot( -T2[0], -T2[1], -T2[2], "ko" ) ;

# XYZ CAMERAS #
w = 25. ; h = 60. ;
xyz_cam_0 = np.array( [
            [w, -w, -w, 0., -w ],
            [h, h, -h, 0., h],
            [d, d, d, 0., d]
] )
xyz_cam_02 = -xyz_cam_0 ;
xyz_cam_02[2] = xyz_cam_0[2] ;



# CAMERA 1 #
xyz_cam = R.transpose() @ xyz_cam_0 ;
xyz_cam_2 = R.transpose() @ xyz_cam_02 ;

xyz_cam[0] -= T[0] ; xyz_cam[1] -= T[1] ; xyz_cam[2] -= T[2] ;
xyz_cam_2[0] -= T[0] ; xyz_cam_2[1] -= T[1] ; xyz_cam_2[2] -= T[2] ;
xy = xyz_to_xy( xyz_cam, R_cam, T_cam ) ;
ax.plot( xy[0], xy[1], "w" ) ;
# ax.plot( xyz_cam[0], xyz_cam[1], xyz_cam[2], "w" ) ;


xy = xyz_to_xy( xyz_cam_2, R_cam, T_cam ) ;
ax.plot( xy[0], xy[1], "w", zorder=100 ) ;
# ax.plot( xyz_cam_2[0], xyz_cam_2[1], xyz_cam_2[2], "w" ) ;



# CAMERA 2 #
xyz_cam = R @ xyz_cam_0 ;
xyz_cam_2 = R @ xyz_cam_02 ;

xyz_cam[0] -= T2[0] ; xyz_cam[1] -= T2[1] ; xyz_cam[2] -= T2[2] ;
xyz_cam_2[0] -= T2[0] ; xyz_cam_2[1] -= T2[1] ; xyz_cam_2[2] -= T2[2] ;
xy = xyz_to_xy( xyz_cam, R_cam, T_cam ) ;
ax.plot( xy[0], xy[1], "w" ) ;
# ax.plot( xyz_cam[0], xyz_cam[1], xyz_cam[2], "w" ) ;


xy = xyz_to_xy( xyz_cam_2, R_cam, T_cam ) ;
ax.plot( xy[0], xy[1], "w", zorder=100 ) ;
# ax.plot( xyz_cam_2[0], xyz_cam_2[1], xyz_cam_2[2], "w" ) ;



# ax.annotate( "camera 1", [-0.22, -0.08], ha="left", va="center" ) ;
# ax.annotate( "camera 2", [0.32, -0.08], ha="left", va="center" ) ;

arrowprops = {'arrowstyle':'->', 'color':'white', 'lw':2. }
ax.annotate( 'image planes', [0.02, -0.25], ha='center', va='center' ) ;

ax.annotate( '', [-0.06, -0.20], [-0.01, -0.24], arrowprops=arrowprops ) ;
ax.annotate( '', [0.096, -0.20], [0.04, -0.24], arrowprops=arrowprops ) ;

ax.axis("equal") ;
ax.axis("off") ;
# ax.axis("off") ;
# ax.set_xlim( [-500, 500] ) ;
# ax.set_ylim( [-500, 500] ) ;
# ax.set_xlim( [0, 100] ) ;
# plts.show(True)
fig.tight_layout( pad=0 ) ;
fig.savefig( path_print + "triangulation.pdf" ) ;


quit() ;


I = plts.plt.imread( "I0.jpg" ).astype("float64").copy() ;
# I = I[:10,:10] ;
print(I.shape) ;

# fig, ax = plts.plt.subplots() ;
# ax.imshow(I, cmap="gray") ;
# plts.show(True)

xyz = np.empty( (3, I.size), dtype="float64" ) ;
x, y = np.meshgrid( np.arange(I.shape[1]), np.arange(I.shape[0]) ) ;
xyz[0] = x.flatten() ;  xyz[1] = y.flatten() ;
xyz[2] = 0. ;

x = xyz[0].reshape( I.shape ) ;
y = xyz[1].reshape( I.shape ) ;
z = xyz[2].reshape( I.shape ) ;
print("x\n", x[:3,:3]) ;
print("y\n", y[:3,:3]) ;




print("Imax", I.max())
norm = matplotlib.colors.Normalize(vmin=0, vmax=I.max()) ;


fig, ax = plts.buildFigAx( width, height, b3D=True ) ;
ax.plot_surface( x, y, z, facecolors=plts.plt.cm.gray(norm(I)), rstride=3, cstride=3, shade=False, edgecolors="none" ) ;
# ax.plot_surface(x, y, z, facecolors=plt.cm.gray(norm(I)), shade=False, cstride=1)
# ax.axis("off") ;
ax.set_xlim( [0, 1000] ) ;
ax.set_ylim( [0, 1000] ) ;
# ax.set_xlim( [0, 100] ) ;
plts.show(True)
fig.tight_layout( ) ;
plts.savefig( fig, "Figures/triang.pdf" ) ;


# ------------------------------------- #
