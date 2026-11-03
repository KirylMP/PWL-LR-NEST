"""Feature matrix with constant C thresholds; geom format matches sloped model."""
import numpy as np
def basis(y,g,geom):
 y=np.atleast_2d(y);g=np.broadcast_to(g,len(y));C,h,I=y.T
 c0,i1,width,height=geom;top=c0+height
 v=np.maximum(g,0)**.7
 return np.column_stack([np.ones(len(y)),C,h,I,v/(v+.02**.7),v/(v+.5**.7),v/(v+3**.7),np.maximum(C-c0,0),np.maximum(C-top,0)])
