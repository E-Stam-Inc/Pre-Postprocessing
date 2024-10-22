'''
Interactive processing of image files from a specified folder 
with demonstration of results in (matplotlib) plots format.

Includes:

1) preprocessing - preparing the image for box detection,
2) postprocessing - calculating the boxes on the image with results validation.

Running:

The program runs in the terminal in interactive mode, examples of run:

python3 -i *.py -J./data/					- processing files from a folder './data/'
python3 -i *.py -J./data/ -s1800			- processing files from folder './data/' with resize
python3 -i *.py -J./data/cm/ -s1800 -i		- processing files from folder './data/cm/' with resize and inversion of pixel intensities
python3 -i *.py -J./data/cm/ -i -n76 		- processing files from folder './data/cm/', starting from n-flie (#76) and with inversion of pixel intensities

Usage interative mode (control keys description):

After preprocessing the current file, it shows the timing of the basic operations 
and waits for a control key to be pressed to continue:

- Enter 	- processing the next file in the folder;
- 'q'+Enter - exit to terminal;
- 'a'+Enter - run advanced postprocessing (building boxes, binding boxes to lines, validating results);
- 'b'+Enter - run simple postprocessing (only building boxes);
- 's'+Enter - enabling streaming (without waiting for the control command), only preprocessing.

Libraries used:

numpy		- all matrix operations
opencv		- read files and special functions
matplotlib	- plot results
scipy		- zoom images

'''

import numpy as np				
import cv2 			

import argparse
import os,sys,math,time,datetime,warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

import scipy.ndimage


#███████████████████████████████████████████████████████████████████ 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pcolors = ['red','green','blue','yellow','magenta','cyan','grey','red','green','blue','orange','lightgray','gray','darkgray','lightblue','teal','seagreen','mediumseagreen','orange','tomato','snow','steelblue','red','black']
pcolors1 = ['grey','steelblue','slategray','royalblue','magenta','teal','black','blue','darkgray']
pcolors2 = ['green','blue','orange']

warnings.simplefilter("ignore")
warnings.filterwarnings('ignore',category=FutureWarning) 


#███████████████████████████████████████████████████████████████████████ statistics


'''

r = extractpixcntSB(agray,cnts0,cnts,title='agray: '+name)
r = extractpixcntSB(rs1,cnts0,cnts,title='rs: '+name)

#-----------------------------------------------------------------------
24643 (good)
stats for agray: 	24643 [0.68, 0.13] *** B: [253  19   4 138] ,*** S: [  35 1555   39  255] 253-35=218
stats for rs: 		24643 [0.68, 0.13] *** B: [254  24   4 106] ,*** S: [  26 1382   37  254] 254-26=228

29508 (noised by surface defects)
stats for agray: 	29508 [0.61, 0.14] *** B: [233 282  16 255] ,*** S: [  20 1577   39  255] 233-20=213
stats for rs: 		29508 [0.61, 0.14] *** B: [252 102  10 202] ,*** S: [  18 1394   37  255] 252-18=234

29634 (noised by glare)
stats for agray: 	29634 [0.68, 0.13] *** B: [218  65   8 227] ,*** S: [  52 996    31  208] 218-52=166
stats for rs: 		29634 [0.68, 0.13] *** B: [253  42   6 246] ,*** S: [  35 1567   39  244] 253-35=218
#-----------------------------------------------------------------------

'''	
def extractpixcntSB(img,cnts0,cnts,vi=900,title='BS: ',cmap='ocean',k=5,k1=(2,1)):
	#------------------------------------------------------------------- background
	mask1 = np.zeros_like(img,dtype=np.uint8)
	cv2.drawContours(mask1,[np.array(c,dtype=np.int32) for c in cnts0 ], -1, (255,0,0), -1)	
	if not k is None:
		kernel = np.ones((k,k),np.uint8)
		mask1 = cv2.dilate(mask1,kernel,iterations = k1[0])						
	m1 = mask1==0
	pixs1 = img[m1]
	#------------------------------------------------------------------- letters
	mask2 = np.zeros_like(img,dtype=np.uint8)
	cv2.drawContours(mask2,[np.array(c,dtype=np.int32) for c in cnts ], -1, (255,0,0), -1)	
	if not k is None:
		kernel = np.ones((k,k),np.uint8)
		mask2 = cv2.erode(mask2,kernel,iterations = k1[1])						
	m2 = mask2>0
	pixs2 = img[m2]
	#------------------------------------------------------------------- 
	t1 = np.int16((pixs1.mean(),pixs1.var(),pixs1.std(),pixs1.ptp()))
	t2 = np.int16((pixs2.mean(),pixs2.var(),pixs2.std(),pixs2.ptp()))
	s = np.prod(m1.shape)
	pp = np.round(np.array((m1.sum(),m2.sum()))/s,2).tolist()
	print('\n*** Stats: ',title.rjust(15,' '),pp,'*** B:',t1,',*** S:',t2,'::: delta=',t1[0]-t2[0])
	#-------------------------------------------------------------------
	if vi:
		implotn((img,m1*255,m2*255),cmap=cmap,fig=vi,titles=('src','B:'+str(len(cnts0)),'S:'+str(len(cnts))),title=title+str(pp)+'%, B'+str(t1)+', S'+str(t2))
		vi = vi+1
		implotn((mask1,mask2),fig=vi,titles=('Bmask (==0)','Smask (=255)'))
	return pixs1,pixs2	
	

#███████████████████████████████████████████████████████████████████████ plots


def xymark(x,y,fig=100,title=None,pcolor=None,lw = 1,col=None,alpha=1,label=None,ls='solid',invert=None):
	fig = plt.figure(fig) 
	if not title is None: plt.title(title) 
	if not pcolor is None: fig.patch.set_facecolor(pcolor);
	if invert: plt.gca().invert_yaxis();
	if not col is None:
		plt.plot(x,y,lw = lw,color=col, alpha=alpha, label=label, ls=ls)
	else:
		plt.plot(x,y,lw = lw, alpha=alpha, label=label, ls=ls)
	if not label is None: plt.legend()
	return fig

def xyplot( x, y, color = None, pcolor=None, title=None,fig = 3, lw = 3,alpha=1,sign = '.',label=None, picker=False, invert=False ):
	fig = plt.figure( fig )
	if not title is None: plt.title(title) 
	if not pcolor is None: fig.patch.set_facecolor(pcolor);
	if not color is None:
		plt.plot( x, y, sign, color = color, linewidth = lw, picker=picker, label=label,alpha=alpha ) 
	else:
		plt.plot( x, y, sign, linewidth = lw, picker=picker, label=label,alpha=alpha) 
	if not label is None: plt.legend()
	if invert: plt.gca().invert_yaxis() 
	return fig 

def implot(img,fig=7555, title=None, cmap='gray',color=None, label=None):
	fig = plt.figure( fig ); plt.clf();
	if not color is None: fig.patch.set_facecolor(color);
	if not title is None: plt.title(title)  
	plt.imshow(img,cmap=cmap); 
	if not label is None: 
		plt.legend(prop={'size': 10},loc='upper left',bbox_to_anchor=(1,1),handles=label)
	plt.axis('off')
	plt.show()
	return fig

def implotn(images,fig=7055, title=None, titles=None, color=None,tcolor='black', cmap='gray',label=None):
	fig = plt.figure( fig ); plt.clf(); plt.axis('off');
	if not color is None: fig.patch.set_facecolor(color);
	if not title is None: plt.title(title,color=tcolor)  
	n = len(images)
	grid = plt.GridSpec(1,n)
	axs = []
	for i in range(n):
		ax = fig.add_subplot(grid[0,i]) 
		ax.axis('off')
		ax.imshow(images[i],cmap= cmap[i] if (isinstance(cmap,(list, tuple, np.ndarray))) else cmap )
		if not titles is None:
			ax.set_title(titles[i]);
		axs.append( ax )
	if not label is None: 
		plt.legend(prop={'size': 10},loc='upper left',bbox_to_anchor=(1,1),handles=label)
	plt.show()	
	return axs

def gplot(a, xk=1.,color=None, fig=5555, lw=1, title=None, label=None, ls='solid',fclear=False,alpha=1,labels=None,invert=False):
	fig = plt.figure( fig )
	ax = fig.gca();
	if not title is None: plt.title(title)
	if fclear: plt.clf()
	if invert: plt.gca().invert_yaxis();
	plt.plot(np.arange(len(a))*xk, a, color=color, lw = lw, label=label, ls=ls, alpha=alpha); 
	plt.gcf().set_size_inches((11.2,5.1))
	if not label is None: plt.legend()
	if labels:
		plt.xlabel(labels[0])
		plt.ylabel(labels[1])
	plt.show()
	return fig,ax


#███████████████████████████████████████████████████████████████████████ tools


def pinbox(b,p,shape,offw=0,offh=0):									# b = (x,y,w,h)
	h,w = shape
	return max(0,b[0]-offw) < p[0] < min(w-1,b[0]+b[2]+offw) and max(0,b[1]-offh) < p[1] < min(h-1,b[1]+b[3]+offh)

def normalarray(a):
	amin,amax = np.min(a), np.max(a)
	d = (amax - amin)
	if d>0:
		return (a - amin) / (amax - amin)
	else:
		return a  

def pdist( p1, p2 ):
    ((x1,y1),(x2,y2)) = (p1,p2)
    return ((x2-x1)**2+(y2-y1)**2)**0.5


'''
group function for array of functions & array of thresholds
'''
def groupff(va,ff=[],tt=[]): return group( np.arange(len(va)), lambda a, b: np.all([ (ff[i](va[a],va[b])<tt[i]) for i in range(len(ff))]) ) 

'''
grouping
'''
def group( a, r ):
    def trygroup( g ):
        for t in g:
            if r( t, e ):
                return True
        return False
    
    ga = []

    for e in a:

        gx = [g for g in ga if trygroup( g )]

        if len( gx ):
            for g in gx[1:]:
                gx[0].extend( g )
                
                ga.remove( g )

            gx[0].append( e )
        else:
            ga.append( [e] )

    return ga	


def pa2ca( pa ):	return pa[:,0] + 1j * pa[:,1];
def ca2pa( ca ):   return np.vstack( (ca.real, ca.imag) ).T;  

def ca2ta( ca ):  return [(c.real, c.imag) for c in ca]
def tuple2c( t ):  return complex( t[0], t[1] )
def c2tuple( c ):  return (c.real, c.imag)
def cn( c ):  return c / abs( c )


#███████████████████████████████████████████████████████████████████████ simple contours detector

'''
'''
def line2points(p1,p2,n=None):
	if n is None:  n = np.round( np.sqrt( (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 ) ) + 1
	n = int(n)
	x = np.linspace( p1[0], p2[0], n )
	y = np.linspace( p1[1], p2[1], n )
	return np.array((x,y)).T

'''
'''
def cnt2points(cnt,n=None,eps=0.1):
	p1 = cnt[0]
	cnew = []
	for i in range(1,len(cnt)):
		cnew.append( line2points(p1,cnt[i],n=n) )
		p1 = cnt[i]
	cnew = np.concatenate(cnew)
	#-------------------------------------------------------------------
	if not eps is None:
		l = np.sqrt(((np.diff(cnew,axis=0))**2).sum(axis=1))
		i0  = np.argwhere(l < eps).ravel() + 1
		cnew = np.delete(cnew,i0,axis=0)
	#-------------------------------------------------------------------
	return cnew

'''
'''	
def roi2cntsS(e,thrs=(7,100,5),fext=True,fnorm=True):
	#------------------------------------------------------------------- total external cnts
	caa = cv2.findContours(e,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]			
	n = len(caa)
	boxcnts = np.array([ cv2.boundingRect(np.uint(np.round(c))) for c in caa ])	
	sboxs = np.prod(boxcnts[:,2:],axis=1)											
	#------------------------------------------------------------------- filtered by len, area values
	h,w = e.shape
	t,ts,tr = thrs 
	ts1 = (h*w)/20
	ca = [ c.reshape(-1,2) for i,c in enumerate(caa) if ((max(boxcnts[i][2:])/min(boxcnts[i][2:]))<tr) and len(c)>t and cv2.contourArea(c)>t and sboxs[i]>ts and sboxs[i]<ts1 ]					# close cnts
	#------------------------------------------------------------------- extend ech cnt
	if fext:
		ca = [ cnt2points(c) for c in ca ]
	#------------------------------------------------------------------- set start contour point
	if fnorm:
		ca1 = []
		for c in ca:
			j0 = np.argsort(c[:,1])[:len(c)//2]
			i0 = np.argmin(c[:,0][j0])
			j1 = j0[i0] # - 1
			i1 = np.arange(max(0,j1-5),min(j1+5,len(c[:,1])-1))
			j2 = np.argmin( c[:,1][i1] )
			jj = i1[j2]
			ca1.append( np.roll(c,-jj,axis=0) )
		ca = ca1
	return ca	


#███████████████████████████████████████████████████████████████████████ geometry detector


'''
function for special padding
'''
def custom_padding(arr, pad_width, iaxis,kwargs):
	p = kwargs.get('perc', 50)
	o = kwargs.get('offs', None)
	if (o is None) or (o==0):
		v1 = v2 = np.percentile(arr[pad_width[0]:-pad_width[1]],p,axis=0)
	else:
		v1 = v2 = np.percentile(arr[pad_width[0]+o:-pad_width[1]-o],p,axis=0)
	arr[:pad_width[0]] = v1
	arr[-pad_width[1]:] = v2

'''
special padding
'''		
def sldgrayf(agray,K=100,p=30,offs=None,k=255,vi='SldG',mode='median',cmap='Blues'):
	a = np.pad(agray,K,custom_padding,perc=p,offs=offs); a[K:-K,K:-K] = agray if (k is None) else k
	if not vi is None:
		implotn((agray,a),titles=('src','ext'),fig=vi,cmap=cmap)
	return a
	
'''
horizontal & vertical lines detector (geometry of document)
'''
def geometry1(picture,k=20,K=100,p=20,offs=None,thrl=100,vi=100):
	img = picture.copy()
	#------------------------------------------------------------------- 
	h0,w0 = img.shape
	if not k is None:
		t = 255
		img[:k,:] = t; img[h0-k:,:] = t
		img[:,:k] = t; img[:,w0-k:] = t
	#------------------------------------------------------------------- 
	a = normalarray(sldgrayf(img,K=K,p=p,offs=offs,k=None,vi=None))
	h,w = a.shape
	#------------------------------------------------------------------- 
	aright = a[K:-K,-K:]		
	aup = a[:K,K:-K]			
	#------------------------------------------------------------------- 
	y = aright[:,1] 
	xx = np.where(y<0.5)[0]
	laa = group(xx, lambda a,b: np.abs(a-b)<10)
	if len(laa)>10:
		yR = np.array([ ii[0] for ii in laa if len(ii)>20 ])
		yR0 = yR.copy()
		dd = np.diff(yR)		# diffs between yR0
		dd = dd[dd>90]
	else:
		yR = None
		yR0 = None
	#------------------------------------------------------------------- xL,xR
	y = aup[1,:]
	xx = np.where(y<0.5)[0]
	if len(xx)>=2:
		xL,xR = xx[[0,-1]]
	else:
		xL,xR = k,w0-k
	#------------------------------------------------------------------- 
	yG = None			
	#------------------------------------------------------------------- yR(s) filter
	if not(yR is None) and len(yR)>1:
			global T; T = yR,thrl,img,a,K
			t = np.median(dd)*0.6
			laa = group(yR, lambda a,b: np.abs(a-b)<t)
			if len(laa)>10:
				yR = np.array([ ii[0] for ii in laa ])					
				yG = np.array([ (ii[0],ii[-1]) for ii in laa ])
				d = np.diff(yR).max()
				while yR[-1]<h0: yR = np.append(yR,yR[-1]+d)
				while yR[0]>0: yR = np.insert(yR,0,yR[0]-d)
				if yR[-1]>h0: yR = np.delete(yR,-1)
				if yR[0]<0: yR = np.delete(yR,0)
			else:
				yR,yG = None,None
	#------------------------------------------------------------------- plot
	if vi:
		axs = implotn((img,a),title='Geometry for '+name+', hor lines:'+str(len(yR))+', vert lines:'+str([xL,xR]),titles=('src','padding'),fig=vi)
		if not (xR is None):
			axs[0].plot((xR,xR),(0,h0),color='blue')
		if not (xL is None):
			axs[0].plot((xL,xL),(0,h0),color='green')	
		if not (yR is None):
			_= [ axs[0].plot((0,w0),(y,y),color='red') for y in yR ]
			_= [ axs[1].plot((K,w-K),(y,y),color='red') for y in yR+K ]
			_= [ axs[1].plot((K,w-K),(y,y),color='orange',ls='dotted') for y in yR0+K ]		
	#-------------------------------------------------------------------
	return yR,yG,xL,xR


#███████████████████████████████████████████████████████████████████████ subscript symbols to lines linker
	
'''
subscript symbols to lines linker
'''	
def lowerbyline(iinboxes,rii):
	if len(iinboxes)<1 or (rii is None): return None,None
	#-------------------------------------------------------------------
	lowerbyline = []
	#-------------------------------------------------------------------
	for i in iinboxes:
		for j,ii in enumerate(rii):
			if i in ii[3]:
				lowerbyline.append((i,j))
	#-------------------------------------------------------------------
	if len(lowerbyline):
		lowerbyline = np.array(lowerbyline)
		ln = [ (lowerbyline[:,1]==i).sum() for i in range(len(rii)) ]
	else:
		return None,None
	#-------------------------------------------------------------------
	return lowerbyline,ln	


#███████████████████████████████████████████████████████████████████████ subscript symbols detector
	
'''
subscript symbols (children) detector
'''		
def lowersymboldetect(img,boxcnts,cbox,yL,yR,d=None,color='red',vi=None):
	shape = img.shape
	#-------------------------------------------------------------------
	if d is None:
		if not (yL is None):
			d = np.diff(yL).mean() * 2 / 3
		if not (yR is None):
			d = np.diff(yR).mean() * 2 / 3
		else:
			d = 50
	#-------------------------------------------------------------------
	ij = []
	for i in range(len(cbox)-1):
		cb1 = cbox[i]
		b1 = boxcnts[i]
		for j in range(1,len(cbox)):
			cb2 = cbox[j]
			b2 = boxcnts[j]
			#-------------------------------- центр бокса ребенка ниже (У-ось) центра бокса родителя 
			f1 = cb1[1] > cb2[1]
			if not f1: continue
			#-------------------------------- центр бокса ребенка левее (Х-ось) центра бокса родителя
			f2 = cb1[0] < cb2[0]
			if not f2: continue
			#-------------------------------- центр бокса ребенка отстоит ниже (У-ось) центра бокса родителя не более чем d (на расстояние между строками)
			f3 = (cb1[1] - cb2[1]) < d
			if not f3: continue
			#-------------------------------- площадь бокса ребенка меньше площади родителя
			f4 = np.prod(boxcnts[i][2:4]) < np.prod(boxcnts[j][2:4])
			if not f4: continue
			#-------------------------------- середина верхней грани бокса ребенка внутри бокса родителя
			p = (cb1[0],b1[1])
			f5 = pinbox(b2,p,shape,offw=0,offh=0)
			if not f5: continue
			#--------------------------------
			# f = f1 * f2 * f3 * f4 * f5
			# if f:
			#	print([i,j],'***',cb1,cb2)
			ij.append( (i,j) )
	#-------------------------------------------------------------------
	ij = np.array(ij)
	if vi:
			iinboxes,jinboxes = np.array(ij).T
			xyplot(*cbox[iinboxes].T,color=color,sign='o',fig=vi)	
	return ij	


#███████████████████████████████████████████████████████████████████████ boxes to lines linker

fabsdiff = lambda a,b: np.abs(a-b)


'''
Boxes to lines linker

Inputs
	K1 - maximum spread box centers in yours line
	K2 - maximum spread box centers to line
Outputs
	rii=i,j,y,ii - индекс в группе (внутренний параметр), индекс линии, У-линии, массив индексов боксов, привязанных к линии j
'''
def boxes2lineslink(cbox,boxcnts,sbox,yR,yL,K1=20,K2=50,K3=0.8,vi=None):
	ldefault = (38,40,41,32,39,32,30,42,34,39,33,34,36,39,41,37,39,32,35,38,37,7)
	#----------------------------------------------------------- groups indexes boxes centers OK::: s = np.array([ len(ii) for ii in laa ]).sum()
	laa = groupff(cbox[:,1],ff=[fabsdiff],tt=[K1])				
	la = list(map(len,laa))
	#----------------------------------------------------------- (laai to linej with value y and lla[i] )
	if not (yR is None):
				# rii = [ (i,j,y,ii) for i,ii in enumerate(laa) for j,y in enumerate(yR) if np.all(np.abs(cbox[ii][:,1]-y)<K2)] # 
				rii = [ (i,j,y,ii) for i,ii in enumerate(laa) for j,y in enumerate(yR) if len(ii)>5 and (np.abs(cbox[ii][:,1]-y)<K2).sum()>=(len(ii)*K3) ] # 
				if vi:
					s = 0
					for i,j,y,ii in rii: 
						s += len(ii)
						# print('group Right_laa #',i,[len(ii)],'==',[ldefault[j]],'and line #',j,'with y==',y)
						print('group Right_laa #',i,[len(ii)],'== line #',j,'with y==',y)
					print([ "{: 3d}".format(i) for i,_,_,ii in rii[::-1]],'*',len(rii))
					print('-----------------------------------------------------------------------------------------------------------------------------------------------')
					print([ "{: 3d}".format(len(ii)) for i,_,_,ii in rii[::-1]],'*',s)
					print('-----------------------------------------------------------------------------------------------------------------------------------------------')
					print([ "{: 3d}".format(n) for n in ldefault])
					print('\nfounded groups boxes by spread: K1=',K1,'groups-by-line=',[len(laa)],'==',[np.sum(la)])
	elif not (yL is None):
				rii = [ (i,j,y,ii) for i,ii in enumerate(laa) for j,y in enumerate(yL) if len(ii)>5 and (np.abs(cbox[ii][:,1]-y)<K2).sum()>=(len(ii)*K3) ] #  # (laai to linej with value y and lla[i] )
				if vi:
					s = 0
					for i,j,y,ii in rii: 
						s += len(ii)
						# print('group Left_laa #',i,[len(ii)],'==',[ldefault[j]],'and line #',j,'with y==',y)
						print('group Left_laa #',i,[len(ii)],'== line #',j,'with y==',y)
					print([ "{: 3d}".format(i) for i,_,_,ii in rii[::-1]])
					print('----------------------------------------------------------------------------------------------------------')
					print([ "{: 3d}".format(len(ii)) for i,_,_,ii in rii[::-1]])
					print('-----------------------------------------------------------------------------------------------------------------------------------------------')
					print([ "{: 3d}".format(n) for n in ldefault])
					print('\nfounded groups boxes by spread: K1=',K1,'lines 22==',[len(laa)],',775==',[np.sum(la)],',62==',[len(ij)])
	else:
				c = np.array((np.arange(len(cbox)),cbox[:,1])).T;
				smooth1,rang1 = 10505,2
				c1 = splinecnt(c,ns=None,vi=None,smooth=smooth1,k=rang1)
				if vi:
					gplot(c[:,1],label='Yc',fig='Ystairway',title='Y(c) & smooth(Y(c)) stairway',invert=1); gplot(c1[:,1],label='YsmoothC'+str([smooth1,rang1]),fig='Ystairway')
				# d1 = normalarray(c[:,1]-c1[:,1])
				#-------------------------------------------------------
				c = np.array((np.arange(len(boxcnts)),boxcnts[:,1])).T; 	
				smooth2,rang2 = 10505,2
				c2 = splinecnt(c,ns=None,vi=None,smooth=smooth2,k=rang2)
				if vi: 
					gplot(c[:,1],label='Y',fig='Ystairway'); gplot(c2[:,1],label='Ysmooth:'+str([smooth2,rang2]),fig='Ystairway')
				# d2 = normalarray(c[:,1]-c2[:,1])
				a,b = np.diff(c2[:,1]), np.diff(normalarray(sbox))			
				ilines = np.where((a<-3) & (b<0))[0]
				#-------------------------------------------------------
				yR = cbox[ilines][:,1]
				yL = None
				rii = None
	#-------------------------------------------------------------------
	return yR,yL,rii,laa


#███████████████████████████████████████████████████████████████████████ pixels filters	
	
'''
base opencv 2d filter
'''
def filter2d(img,kernel,cmap=None,vi=None):
	src = np.where(img != 0, 1., 0.)
	src = np.int16(src)
	v = cv2.filter2D(src, ddepth=cv2.CV_16S, kernel=kernel,anchor=(-1, -1), delta=0., borderType=cv2.BORDER_DEFAULT,)
	if vi:
		implotn((img,v),titles=('src','v'),title='f2d:'+str(kernel.tolist()),fig=vi,cmap=cmap)
	return v	

'''
alone pixels filtering
'''
def mnfilterM(m,kernel,n=3,t=[7],cmap=None,vi=100):
	m1 = m.copy()
	h,w = m.shape
	s0 = 0 
	for i in range(n):
		v=filter2d(m1,kernel,vi=None,cmap=None)
		mi = np.isin(v,t)
		y,x = np.where(mi)
		m1[mi] = [ np.median(m1[max(0,j-1):min(h-1,j+2),max(0,i-1):min(w-1,i+2)]) for j,i in zip(y,x) ]
		s1 = mi.sum()
		if s1==s0:
			break
		s0 = s1
	v=filter2d(m1,kernel,vi=None,cmap=None)
	if vi:
		implotn((m,m1,v),titles=('m','m1:'+str(i),'lastv:'+str(np.isin(v,t).sum())),title='mnfilterI for:'+str([n,kernel.tolist(),t]),fig=vi,cmap=cmap)
	return m1,v	


#███████████████████████████████████████████████████████████████████████ boxes filters

'''
std-based boxes filtering
'''
def badboxesi(img,boxcnts,o=0,thr=100,vi=None):
	stds = np.array([ img[y-o:y+h+1+o*2,x-o:x+w+1+o*2].std() for x,y,w,h in boxcnts ])
	ibad = np.where(stds<thr)[0]
	if vi: 
		_=[ xymark((x,x+w,x+w,x,x),(y,y,y+h,y+h,y), col='red',fig=vi) for (x,y,w,h) in boxcnts[ibad]]
		vi = vi + 1
		gplot(stds,fig=vi); gplot([thr]*len(std),label='thr:'+str(thr))
	return ibad


#███████████████████████████████████████████████████████████████████████ shining (antiglare)

'''
Recovering grayscale (8bit) image
'''
def shining(img,sigmaX=99,sigmaY=99,K=5,vi=None):
	# statnoises removing
	b = cv2.GaussianBlur(img, (0,0), sigmaX=sigmaX, sigmaY=sigmaY)
	d = cv2.divide(img,b,scale=255)
	# normalization to float
	an = cv2.normalize(np.float32(d),None,0.,1.,cv2.NORM_MINMAX,dtype=cv2.CV_32F) 
	# intensity spreading by koefficient
	s = 1 - cv2.pow(np.float64(1-an),K/10)	
	# normalization to uint8											
	r = cv2.normalize(s,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
	if vi:
		implotn((img,d,r),titles=('src','d','r'),cmap=None,fig=vi)
	return d,r	



#███████████████████████████████████████████████████████████████████████ 				
#███████████████████████████████████████████████████████████████████████  cl
#███████████████████████████████████████████████████████████████████████ 

if os.environ.get('DISPLAY','') == '':
		print('no display found. Using non-interactive Agg backend');
		mpl.use('Agg');	 	
else:
		mpl.use('TkAgg')	
plt.ion()															# plt.ioff()
np.set_printoptions(formatter={'float_kind':"{:.4f}".format})

#███████████████████████████████████████████████████████████████████████ 


ap = argparse.ArgumentParser()

ap.add_argument( "-J", "--Jpath",	type = str,	default=None,	help='fullpath to JJ dataset [None]')
ap.add_argument( "-s", "--maxH",	type = int,	default= None,	help='max H for image processing (for resize)')
ap.add_argument( "-n", "--rn",		type = int,	default =0,		help='start frame number from path (begin from n-frame)')
ap.add_argument( "-i", "--invert",	action = "store_true", default=False, help="inverting source image flag" )
#----------------------------------------------------------
args = vars(ap.parse_args())
#---------------------------------------------------------- 

Jpath = args["Jpath"]
maxH = args["maxH"]
rowseek = int(args["rn"])
invertimage = args["invert"]


spar = '[J]='+str(Jpath)+'][S]='+str(maxH)+'][start]='+str(rowseek)+'][invert]='+str(invertimage)+']'
	
print("\n-----------------------------------------------------------------------------------------")	
print("שלום","  t" if sys.flags.interactive else "m",spar)
print("    *** python3 -i *.py -J./data/")
print("    *** python3 -i *.py -J./data/ -s1800")
print("    *** python3 -i *.py -J./data/cm/ -s1800 -i")
print("    *** python3 -i *.py -J./data/cm/ -i -n76 (for 29634)") 
print("    *** python3 -i *.py -J./data/cm/ -i -n23 (for 29508)")
print("    *** python3 -i *.py -J./data/cm3/")
print("-----------------------------------------------------------------------------------------\n")


#███████████████████████████████████████████████████████████████████████ start



if not Jpath is None:	
	#------------------------------------------------------------------- list files in path											
	names = sorted(os.listdir(Jpath)) 
	#------------------------------------------------------------------- global flags
	
	fstream = None														# autostreaming
	froiselection = 0													# roi selection
	roi = None 								
	
	#------------------------------------------------------------------- seek to file (if necessary)
	if rowseek>0:
		names = names[rowseek:]
	#------------------------------------------------------------------- file by file processing	
	#-------------------------------------------------------------------
	
	N = len(names)			# total number of files in folder
	I = 0					# index of current file
	while I<N:
		
		#---------------------------------------------------------------
		#--------------------------------------------------------------- Read and transform (if necessary) image
		#---------------------------------------------------------------
		
		time00 = time.time()
		
		#--------------------------------------------------------------- read grayscale
		name = names[I]													
		image0 = cv2.imread(Jpath+name,0) 
		if image0 is None: 
			I += 1; continue
		image = image0.copy() if len(image0.shape)<3 else cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
		#--------------------------------------------------------------- resize
		if not maxH is None:
			h,w = image.shape
			k = maxH / h
			image = scipy.ndimage.zoom(image, k) 
		#--------------------------------------------------------------- invert
		
		agray = 255 - image if invertimage else image
		
		#---------------------------------------------------------------
		
		time01 = time.time()
		
				
		if not roi is None:
			
			#-----------------------------------------------------------
			#----------------------------------------------------------- Postprocessing
			#-----------------------------------------------------------
			
			
			e = Q[roi[0]:roi[1],roi[2]:roi[3]]; 						# image for postprocessing
			
			
			#----------------------------------------------------------- 1) cnts detector
			
			cnts_open = roi2cntsS(e)														# opened cnts
									
			cnts0 = [ np.append(c,[c[0]],axis=0) for c in cnts_open];						# closing
						
			boxcnts0 = np.array([ cv2.boundingRect(np.uint(np.round(c))) for c in cnts0 ])	# bounded boxes
			
			#----------------------------------------------------------- 2) cnts & boxes filter
			
			img = rs1 
			thrstd=60  
			ibad = badboxesi(img,boxcnts0,thr=thrstd)
			if len(ibad):
				cnts = np.delete(np.array(cnts0,dtype=object),ibad,axis=0)
				boxcnts = np.delete(boxcnts0,ibad,axis=0)
			else:
				cnts = cnts0
				boxcnts = boxcnts0
			
			print('\n*** Contours:',len(cnts0),'--->',len(cnts))
			
			#----------------------------------------------------------- 3) statistics (before & after antiglare)
			
			r = extractpixcntSB(agray,cnts0,cnts,title='agray: '+name,vi=None)
			r = extractpixcntSB(rs1,cnts0,cnts,title='rs: '+name,vi=None)
			
			#----------------------------------------------------------- 4) by symbols processing
			
			cbox0 = np.array((boxcnts0[:,0]+boxcnts0[:,2]/2,boxcnts0[:,1]+boxcnts0[:,3]/2)).T 
			cbox = np.array((boxcnts[:,0]+boxcnts[:,2]/2,boxcnts[:,1]+boxcnts[:,3]/2)).T 	# centers
			sbox = np.prod(boxcnts[:,2:],axis=1)											# size boxes
			
			if key=='a':												# full processing or only boxes
						
				#------------------------------------------------------- subscripts detection
			
				ij = lowersymboldetect(agray,boxcnts,cbox,yL,yR)
				iinboxes,jinboxes = ij.T									
				print('\n*** Subscripts:',len(ij))
			
				#------------------------------------------------------- boxes to lines linker
				K1 = 20 if yR is None else np.diff(yR).mean()*0.25
				K2 = 90 if yR is None else np.diff(yR).mean()*0.6 
				yR,yL,rii,laa = boxes2lineslink(cbox,boxcnts,sbox,yR,yL,K1=K1,K2=K2,vi=None)
				if not (rii is None):
					rii = np.array(rii,dtype=object)
					_,ii = np.unique(rii[:,2],return_index=1)
					rii = rii[ii]
					print('\n*** Boxes2lines:',[len(rii),len(laa)])
					iirii,ninboxes = lowerbyline(iinboxes,rii)
					print('    Subscripts2lines:',[len(iirii),len(ninboxes)])
				else:
					rii = []
					print('\n*** Boxes2lines:',[len(rii),len(laa)])
					_= [ print(i,len(ii)) for i,ii in enumerate(laa) ]
			else:
				iinboxes,rii = [],[]
			#-----------------------------------------------------------
			#----------------------------------------------------------- plots
			#-----------------------------------------------------------
			
			img = rs1 
			fig = implot(img,fig='Report1',title='Bboxes for:'+name+', total cnts= '+str([len(cnts0),len(boxcnts)])+', small= '+str([len(iinboxes)])+', lines:'+str(len(rii)))
			h,w = agray.shape
			ax = fig.gca() 
			
			_=[ xymark((x,x+w,x+w,x,x),(y,y,y+h,y+h,y), col='blue',fig='Report1') for x,y,w,h in boxcnts]				# boxes
			
			if len(ibad):																								# deleted boxes
				xyplot(*cbox0[ibad].T,color='red',fig='Report1')
				
			xyplot(*cbox.T,color='orange',fig='Report1')																# centers
			
			# ax = fig.gca(); _=[ ax.text(x,y,str(i),color='cyan', fontsize=11) for i,(x,y) in enumerate(cbox) ]		# numbers of boxes
			
			if key=='a':		
			
				# linked boxes
				_=[ [ xymark((x,x+w,x+w,x,x),(y,y,y+h,y+h,y),fig='Report1',col=pcolors2[i % len(pcolors2)]) for x,y,w,h in boxcnts[ii]] for i,ii in enumerate(laa) ]
				
				# horizontal lines
				if not (yR is None):
					_=[ xymark((0,agray.shape[1]-1),(y,y), col='blue',fig='Report1',ls='dotted') for y in yR ]			
			
				# visual validation test
				if len(rii):
					ldefault = (38,40,41,32,39,32,30,42,34,39,33,34,36,39,41,37,39,32,35,38,37,7)				# np.sum(ldefault) == 775
					lldefault = (6,4,4,0,3,0,2,7,0,3,2,2,5,6,7,2,1,0,1,2,4,1)									# np.sum(lldefault) == 62
					lln = ninboxes 
					if 1: # len(rii)== len(ldefault):
						_=[ ax.text(w-50,y+10,str(len(ii))+str([ldefault[k],lldefault[k]])+str(lln[k]),color='orange' if ((ldefault[k]==len(ii)) and (lln[k]==lldefault[k])) else 'red', fontsize=11) for k,(i,j,y,ii) in enumerate(rii[:len(ldefault)]) ]
			
				# vertical lines (left & right)
				if not Xright is None:
					ax.plot((Xright,Xright),(0,h-1),color='green',ls='dotted')
				if not Xleft is None:
					ax.plot((Xleft,Xleft),(0,h-1),color='blue',ls='dotted')	
			
				# subscripts boxes and their parents
				xyplot(*cbox[jinboxes].T,color='magenta',sign='o',fig='Report1')	# parents
				xyplot(*cbox[iinboxes].T,color='blue',sign='o',fig='Report1')		# subscripts
			
			#----------------------------------------------------------- end of postprocessing
			#-----------------------------------------------------------
			roi1 = roi.copy()											
			
			if froiselection:
				froiselection = 0
				roi = None
			
		else:
			
			#-----------------------------------------------------------
			#----------------------------------------------------------- Preprocessing
			#-----------------------------------------------------------
		
			#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! shining
			
			D,R = shining(agray,vi=None)
			
			time02 = time.time()
			
			#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! add filter
			
			kernel1 = np.array([[1, 1, 1],[1, 7, 1],[1, 1, 1]], dtype=np.int16);
			rs1,_ = mnfilterM(R,kernel1,n=20,t=[7,8],cmap=None,vi=None)
			
			time03 = time.time()

			#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! bynarization
			
			rs1 = cv2.fastNlMeansDenoising(rs1,None,40,7,7)
			
			q = cv2.adaptiveThreshold(rs1,255,cv2.ADAPTIVE_THRESH_MEAN_C,1,21,5)		
			
			#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! cut borders
						
			h,w = q.shape						
			k = 20 
			q[:k,:] = 0; q[h-k:,:] = 0
			q[:,:k] = 0; q[:,w-k:] = 0
			Q = q				
			
			time04 = time.time()
			
			#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! geometry test
			
			yR,_,Xright,Xleft = geometry1(rs1,K=300,p=20,offs=0,thrl=50,vi='ALines')
			yL = None
			
			time05 = time.time()
			
			print('\n*** Geometry,  [xR,xL]:',[Xright,Xleft],', yR total:', [len(yR),np.diff(yR)] if (not yR is None) else None)	
	
									
			#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! timing
			
			ttime = np.round((time01-time00,time02-time01,time03-time02,time04-time03,time05-time04,time05-time00),2)
			
			print('\n*** Timing:',ttime)
			
			#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! plots
						
			DA = np.int16(agray)-np.int16(D)
			DA1 = np.int16(D)-np.int16(R)
			DA0 = np.int16(agray)-np.int16(R)
			
			implotn((agray,DA,DA1,DA0),title='Diffs for '+name,titles=('src','diff1','diff2','total'),cmap='ocean',fig='Diffs')
						
			axs0 = implotn((agray,D,R,rs1),title='Recover for '+name+str(ttime),titles=('src','D','R','rs1'),cmap='ocean',fig='Recover')
			axs = implotn((agray,Q),cmap=None,title='For postprocessing: '+str(I)+str([name]),titles=('src','Q'),fig='Q')
			
			#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			
		#--------------------------------------------------------------- end of preprocessing
		#---------------------------------------------------------------
		
		#--------------------------------------------------------------- stream manager
		
		if not fstream is None:
				print('\nFrame#',I,[N],image.shape,name,'***',ttime)
				plt.pause(fstream)
		else:
				key = input('\nFrame: '+str(I)+str([N])+' '+name+' '+str(image0.shape)+str(image.shape)+' [q,a,b,s,Enter] ??? #')
				if key=='q': break
				elif key=='s': fstream=0.0000001;
				elif key=='a' or key=='b':						
					froiselection = 1
					h,w = agray.shape
					roi = [0,h,0,w]
					I = I -1
				elif key=='r':
					froiselection = 1
					selector3 = RectangleSelector(axs[0], lambda *args: None)
					key = input('...select rectangle on plot #'+str(vi+2)+'...and press anykey...')
					roi = np.roll(np.intp(selector3.extents),2)
					I = I -1
		print("\n-----------------------------------------------------------------------------------------")
		I = I + 1 
	#-------------------------------------------------------------------
	A = B


'''

'''

