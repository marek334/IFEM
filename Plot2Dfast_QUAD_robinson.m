%https://www.mathworks.com/help/aerotbx/examples/visualizing-geoid-height-for-earth-geopotential-model-1996.html 
tic
clear

coordinates = dlmread('quad_ellipsoid_WGS84_BLH_7.dat'); 
Solution = '131_T_WGS84_BLH_7_triang_inf_512x128x57.dat'; 

T=split(Solution,'_');
D=split(T(end),'.');
R=split(D(1),'x');
nx=str2num(char(R(1)))+1;
ny=str2num(char(R(2)))+1;

[Z]=read_v2(Solution);

coast = load('coast');
geoc_lon=linspace(0,360,nx);
geoc_lat=linspace(-90,90,ny);
[X,Y] = meshgrid(geoc_lon,geoc_lat);

Z = griddata(coordinates(:,2),coordinates(:,1),Z,X,Y,'nearest');
RRR = makerefmat('RasterSize',size(Z),'Latlim', [-90 90], 'Lonlim', [0 360] );
GeoidPlot2D(RRR,Z,coast,0,360,-90,90);

Zmin=min(Z,[],'all');
Zmax=max(Z,[],'all');
caxis([Zmin Zmax])
c=colorbar('location','EastOutside');
c.Label.String = 'T [m^2s^{-2}]';

a=split(Solution,'.');
tmp=char(a{1});
print(tmp, '-dpng', '-r500');

tmp=char(join({a{1},'.png'}));
tmp=strrep(tmp,' ', '');

I = imread(tmp);
IMG=size(I);
r = centerCropWindow2d(IMG,[IMG(1)-800 IMG(2)]);
J = imcrop(I,r);
imwrite(J,tmp)

close all


function [matrix]=read_v2(namefile)

grdfile=fopen(namefile,'r');    
[matrix] = textscan(grdfile, '%f');
matrix=matrix{1};

fclose(grdfile);
end


function GeoidPlot2D(RRR,geoidResults,coast,xmin,xmax,ymin,ymax)
% ast2DGeoidPlot: this is a helper function for astvrViewGeoidHeight to
% create a meshm plot.

%   Copyright 2011 The MathWorks, Inc.

% Define map axis
axesm('MapProjection','robinson','MapLatLimit',[ymin ymax],...
    'MapLonLimit',[xmin xmax],'Grid','off','Frame','on',...
    'MeridianLabel','on','LabelFormat','none',...
    'MLabelParallel','South','ParallelLabel','on',...
    'PLabelLocation',30,'MLabelLocation',90);

% Plot geoid data
meshm(geoidResults,RRR);
colormap('jet');
% Plot coast outline
geoshow(coast.lat,coast.long,'Color', 'white')
axis off
tightmap
end

